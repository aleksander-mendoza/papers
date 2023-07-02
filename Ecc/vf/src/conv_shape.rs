use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::{Step, Sum};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Range, Rem, Sub};
use std::process::Output;
use num_traits::{AsPrimitive, MulAdd, Num, One, PrimInt, Zero};
use crate::shape::Shape;
use crate::{ArrayCast, conv, vec_range, VecCast, VectorField, VectorFieldAddAssign, VectorFieldOne, VectorFieldPartialOrd, VectorFieldSub};
use crate::arr_concat::concat;
use crate::xyzw::{xy3, xy_z3, xy_z_w4, xy_zw4, xyz3, z3};
use crate::from_usize::FromUsize;
use crate::init::InitEmptyWithCapacity;
use crate::norm::{card, normalize_mat_columns};

/**[height, width, channels]->[height, width]*/
pub fn grid<X>(arr: &[X; 3]) -> &[X; 2] {
    xy3(arr)
}

/**[height, width, channels]->width*/
pub fn width<X>(arr: &[X; 3]) -> &X {
    &arr[1]
}

/**[height, width, channels]->height*/
pub fn height<X>(arr: &[X; 3]) -> &X {
    &arr[0]
}

/**[height, width, channels]->channels*/
pub fn channels<X>(arr: &[X; 3]) -> &X {
    &arr[2]
}


#[inline]
pub fn idx<Idx: Debug + Mul<Output=Idx> + Add<Output=Idx> + Copy + Ord>(output_idx: Idx, idx_within_kernel_column: Idx, output_volume: Idx) -> Idx {
    debug_assert!(output_idx < output_volume);
    idx_within_kernel_column * output_volume + output_idx
}

#[inline]
pub fn idx_<Idx: Debug + num_traits::PrimInt>(input_pos: &[Idx; 3], kernel_offset: &[Idx; 2], output_idx: Idx, kernel_column: &[Idx; 3], output_volume: Idx) -> Idx {
    let position_within_kernel_column = sub_kernel_offset(input_pos, kernel_offset);
    idx(output_idx, kernel_column.idx(&position_within_kernel_column), output_volume)
}

pub fn sub_kernel_offset<Idx: Debug + Copy + Sub<Output=Idx>>(input_pos: &[Idx; 3], offset: &[Idx; 2]) -> [Idx; 3] {
    xy_z3(xy3(input_pos).sub(offset), *z3(input_pos))
}


#[derive(Clone, Debug)]
pub struct ConvShape<Idx: Debug + PrimInt> {
    /**[in_height, in_width, in_channels]*/
    input_shape: [Idx; 3],
    /**[out_height, out_width, out_channels]*/
    output_shape: [Idx; 3],
    /**[kernel_height, kernel_width]*/
    kernel: [Idx; 2],
    /**[height, width]*/
    stride: [Idx; 2],
}

impl<Idx: Debug + PrimInt> ConvShape<Idx> {
    /**[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn w_shape(&self) -> [Idx; 6] {
        let [kernel_height, kernel_width] = self.kernel().clone();
        let [out_height, out_width, out_channels] = self.output_shape();
        [kernel_height, kernel_width, self.in_channels(), out_height, out_width, out_channels]
    }
    /**[out_height, out_width, out_channels, out_channels]*/
    pub fn u_shape(&self) -> [Idx; 4] {
        let [out_height, out_width, out_channels] = self.output_shape();
        [out_height, out_width, out_channels, out_channels]
    }
    /**[out_channels, out_channels]*/
    pub fn minicolumn_u_shape(&self) -> [Idx; 2] {
        [self.out_channels(), self.out_channels()]
    }
    /**[out_height, out_width, out_channels]*/
    pub fn out_shape(&self) -> &[Idx; 3] {
        &self.output_shape
    }
    /**[in_height, in_width, in_channels]*/
    pub fn in_shape(&self) -> &[Idx; 3] {
        &self.input_shape
    }
    /**[kernel_height, kernel_width]*/
    pub fn kernel(&self) -> &[Idx; 2] {
        &self.kernel
    }
    /**[height, width]*/
    pub fn stride(&self) -> &[Idx; 2] {
        &self.stride
    }
    /**[out_height, out_width, out_channels]*/
    pub fn output_shape(&self) -> [Idx; 3] {
        self.out_shape().clone()
    }
    /**[in_height, in_width, in_channels]*/
    pub fn input_shape(&self) -> [Idx; 3] {
        self.in_shape().clone()
    }
    /**[kernel_height, kernel_width, in_channels, out_channels]*/
    pub fn minicolumn_w_shape(&self) -> [Idx; 4] {
        xy_z_w4(self.kernel().clone(), self.in_channels(), self.out_channels())
    }
    /**[kernel_height, kernel_width, in_channels]*/
    pub fn kernel_column_shape(&self) -> [Idx; 3] {
        xy_z3(self.kernel().clone(), self.in_channels())
    }
    /**kernel_height * kernel_width*/
    pub fn kernel_column_area(&self) -> Idx where Idx: Mul<Output=Idx> + One {
        self.kernel().product()
    }
    /**kernel_height * kernel_width * in_channels*/
    pub fn kernel_column_volume(&self) -> Idx where Idx: Mul<Output=Idx> + One {
        self.kernel_column_area() * self.in_channels()
    }
    /**[in_height, in_width]*/
    pub fn in_grid(&self) -> &[Idx; 2] {
        grid(self.in_shape())
    }
    /**[out_height, out_width]*/
    pub fn out_grid(&self) -> &[Idx; 2] {
        grid(self.out_shape())
    }
    pub fn out_width(&self) -> Idx {
        *width(self.out_shape())
    }
    pub fn out_height(&self) -> Idx {
        *height(self.out_shape())
    }
    pub fn out_channels(&self) -> Idx {
        *channels(self.out_shape())
    }
    pub fn in_width(&self) -> Idx {
        *width(self.in_shape())
    }
    pub fn in_height(&self) -> Idx {
        *height(self.in_shape())
    }
    pub fn in_channels(&self) -> Idx {
        *channels(self.in_shape())
    }
    /**out_height * out_width*/
    pub fn out_area(&self) -> Idx where Idx: Mul<Output=Idx> + One {
        self.out_grid().product()
    }
    /**in_height * in_width*/
    pub fn in_area(&self) -> Idx where Idx: Mul<Output=Idx> + One {
        self.in_grid().product()
    }
    /**out_height * out_width * out_channels*/
    pub fn out_volume(&self) -> Idx where Idx: Mul<Output=Idx> + One {
        self.out_shape().product()
    }
    /**in_height * in_width * in_channels*/
    pub fn in_volume(&self) -> Idx where Idx: Mul<Output=Idx> + One {
        self.in_shape().product()
    }
    pub fn kernel_offset(&self, output_pos: &[Idx; 3]) -> [Idx; 2] where Idx: Mul<Output=Idx> {
        conv::in_range_begin(grid(output_pos), self.stride())
    }
    pub fn pos_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> [Idx; 3] {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(vec_range::contains(&conv::in_range(grid(output_pos), self.stride(), self.kernel()), grid(input_pos)));
        debug_assert!(vec_range::contains(&conv::out_range_clipped(grid(input_pos), self.stride(), self.kernel()), grid(output_pos)));
        sub_kernel_offset(input_pos, &self.kernel_offset(output_pos))
    }
    pub fn idx_within_kernel(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        self.kernel_column_shape().idx(&self.pos_within_kernel(input_pos, output_pos))
    }
    pub fn in_range(&self, output_column_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        assert!(output_column_pos.all_lt(self.out_grid()));
        conv::in_range(output_column_pos, self.stride(), self.kernel())
    }
    pub fn out_range(&self, input_pos: &[Idx; 2]) -> Range<[Idx; 2]> {
        conv::out_range_clipped_both_sides(input_pos, self.stride(), self.kernel(), self.out_grid())
    }
    pub fn idx(&self, input_pos: &[Idx; 3], output_pos: &[Idx; 3]) -> Idx {
        debug_assert!(output_pos.all_lt(self.out_shape()));
        debug_assert!(input_pos.all_lt(self.in_shape()));
        debug_assert!(vec_range::contains(&conv::in_range(grid(output_pos), self.stride(), self.kernel()), grid(input_pos)));
        debug_assert!(vec_range::contains(&conv::out_range_clipped(grid(input_pos), self.stride(), self.kernel()), grid(output_pos)));
        idx(self.out_shape().idx(output_pos), self.idx_within_kernel(input_pos, output_pos), self.out_volume())
    }
    /**conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]*/
    pub fn normalize_minicolumn<D: DivAssign + Copy>(&self, conv_tensor: &mut [D], norm_with_stride: impl Fn(&[D], usize) -> D) where Idx: AsPrimitive<usize> {
        let in_v = self.kernel_column_volume().as_();
        let out_v = self.out_channels().as_();
        let v = in_v * out_v;
        assert_eq!(conv_tensor.len(), v);
        normalize_mat_columns(out_v, conv_tensor, norm_with_stride)
    }
    /**conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn normalize_kernel_columns<D: DivAssign + Copy>(&self, conv_tensor: &mut [D], norm_with_stride: impl Fn(&[D], usize) -> D) where Idx: AsPrimitive<usize> {
        let in_v = self.kernel_column_volume().as_();
        let out_v = self.out_volume().as_();
        let v = in_v * out_v;
        assert_eq!(conv_tensor.len(), v);
        normalize_mat_columns(out_v, conv_tensor, norm_with_stride)
    }
    /**rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels].
        dot_product_output is of shape [out_height, out_width, out_channels]*/
    pub fn sparse_dot_slice<D: AddAssign + Copy + Zero>(&self, lhs_tensor: &[Idx], rhs_conv_tensor: &[D]) -> Vec<D> where Idx: AsPrimitive<usize> + Step + Hash {
        let mut dot_product_output = vec![D::zero(); self.out_volume().as_()];
        self.sparse_dot_slice_(lhs_tensor, rhs_conv_tensor, &mut dot_product_output);
        dot_product_output
    }
    /**rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels].
         dot_product_output is of shape [out_height, out_width, out_channels]*/
    pub fn sparse_dot_slice_<D: AddAssign + Copy>(&self, lhs_tensor: &[Idx], rhs_conv_tensor: &[D], dot_product_output: &mut [D]) where Idx: AsPrimitive<usize> + Step + Hash {
        debug_assert_eq!(rhs_conv_tensor.len(), self.w_shape().product().as_());
        debug_assert_eq!(dot_product_output.len(), self.out_volume().as_());
        self.sparse_dot(lhs_tensor, |output_idx, w_index| dot_product_output[output_idx.as_()] += rhs_conv_tensor[w_index.as_()])
    }
    pub fn sparse_dot(&self, lhs_tensor: &[Idx], mut target: impl FnMut(Idx, Idx)) where Idx: Step + Hash {
        let kernel_column = self.kernel_column_shape();
        let v = self.out_volume();
        let mut used_w = HashSet::new();
        for &input_idx in lhs_tensor {
            let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
            let r = self.out_range(grid(&input_pos));
            vec_range::foreach2d(&r, |output_pos| {
                let kernel_offset = conv::in_range_begin(&output_pos, self.stride());
                for p2 in Idx::zero()..self.out_channels() {
                    let output_pos = xy_z3(output_pos.clone(), p2);
                    let output_idx = self.out_shape().idx(&output_pos);
                    let w_index = idx_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    debug_assert_eq!(w_index, self.idx(&input_pos, &output_pos));
                    debug_assert!(used_w.insert(w_index), "{:?}", w_index);
                    target(output_idx, w_index);
                }
            });
        }
    }
    /**rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]. dot_product_output is of shape [out_height, out_width, out_channels]*/
    pub fn sparse_dot_repeated_slice<D: AddAssign + Copy + Zero>(&self, lhs_tensor: &[Idx], rhs_conv_tensor: &[D]) -> Vec<D> where Idx: AsPrimitive<usize> + Step {
        let mut dot_product_output = vec![D::zero(); self.out_volume().as_()];
        self.sparse_dot_repeated_slice_(lhs_tensor, rhs_conv_tensor, &mut dot_product_output);
        dot_product_output
    }
    /**rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]. dot_product_output is of shape [out_height, out_width, out_channels]*/
    pub fn sparse_dot_repeated_slice_<D: AddAssign + Copy>(&self, lhs_tensor: &[Idx], rhs_conv_tensor: &[D], dot_product_output: &mut [D]) where Idx: AsPrimitive<usize> + Step {
        debug_assert_eq!(rhs_conv_tensor.len(), self.minicolumn_w_shape().product().as_());
        debug_assert_eq!(dot_product_output.len(), self.out_volume().as_());
        self.sparse_dot_repeated(lhs_tensor, |output_idx, w_index| dot_product_output[output_idx.as_()] += rhs_conv_tensor[w_index.as_()])
    }
    pub fn sparse_dot_repeated(&self, lhs_tensor: &[Idx], mut target: impl FnMut(Idx, Idx)) where Idx: Step {
        let kernel_column = self.kernel_column_shape();
        for &input_idx in lhs_tensor {
            let input_pos: [Idx; 3] = self.in_shape().pos(input_idx);
            let r = self.out_range(grid(&input_pos));
            vec_range::foreach2d(&r, |output_pos| {
                let kernel_offset = conv::in_range_begin(&output_pos, self.stride());
                for p2 in Idx::zero()..self.out_channels() {
                    let output_pos = xy_z3(output_pos.clone(), p2);
                    let output_idx = self.out_shape().idx(&output_pos);
                    let w_index = idx_(&input_pos, &kernel_offset, p2, &kernel_column, self.out_channels());
                    target(output_idx, w_index);
                }
            });
        }
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                        except that this function works with convolutional topology of weights W and the vectors
                        are sparse binary. Here input is X, output is Y, fold generalizes W
         W is expected to be of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels].*/
    pub fn sparse_conjugate<V>(&self, input: &[Idx], output: &[Idx],
                               fold_init: V,
                               mut fold_per_weight_in_kernel_column: impl FnMut(V, Idx) -> V,
                               mut fold_per_kernel_column: impl FnMut(V, Idx) -> V) -> V {
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        let v = self.out_volume();
        let kernel_column = self.kernel_column_shape();
        let mut value = fold_init;
        for &output_idx in output {
            let output_pos = self.out_shape().pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = self.in_range(grid(&output_pos));
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if vec_range::contains(&input_range, grid(input_pos)) {
                    let w_index = idx_(&input_pos, &kernel_offset, output_idx, &kernel_column, v);
                    value = fold_per_weight_in_kernel_column(value, w_index);
                }
            }
            value = fold_per_kernel_column(value, output_idx)
        }
        value
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                            except that this function works with convolutional topology of weights W and the vectors
                            are sparse binary. Here input is X, output is Y, fold generalizes W.
         W is expected to be of shape [kernel_height, kernel_width, in_channels, out_channels].*/
    pub fn sparse_conjugate_repeated<V>(&self, input: &[Idx], output: &[Idx],
                                        fold_init: V,
                                        mut fold_per_weight_in_kernel_column: impl FnMut(V, Idx) -> V,
                                        mut fold_per_kernel_column: impl FnMut(V, Idx) -> V) -> V {
        let input_pos: Vec<[Idx; 3]> = input.iter().map(|&i| self.in_shape().pos(i)).collect();
        let v = self.out_channels();
        let kernel_column = self.kernel_column_shape();
        let mut value = fold_init;
        for &output_idx in output {
            let output_pos = self.out_shape().pos(output_idx);
            let kernel_offset = self.kernel_offset(&output_pos);
            let input_range = self.in_range(grid(&output_pos));
            for (&input_idx, input_pos) in input.iter().zip(input_pos.iter()) {
                if vec_range::contains(&input_range, grid(input_pos)) {
                    let w_index = idx_(&input_pos, &kernel_offset, output_pos[2], &kernel_column, v);
                    value = fold_per_weight_in_kernel_column(value, w_index);
                }
            }
            value = fold_per_kernel_column(value, output_idx)
        }
        value
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                                except that this function works with convolutional topology of weights W and the vectors
                                are sparse binary. Here input is X, output is Y, w_slice is W of shape self.minicolumn_w_shape()==[kernel_height, kernel_width, in_channels, out_channels]*/
    pub fn sparse_unbiased_increment_repeated<D: Copy + Div<Output=D> + AddAssign + FromUsize>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        assert_eq!(w_slice.len(), self.minicolumn_w_shape().product().as_());
        let w_to_increment: Vec<Idx> = Vec::with_capacity(input.len());
        self.sparse_conjugate_repeated(input, output, w_to_increment, |mut w_to_increment, w_index| {
            w_to_increment.push(w_index);
            w_to_increment
        }, |mut w_to_increment, output_idx| {
            let plasticity = epsilon / D::from_usize(w_to_increment.len());
            for w_index in w_to_increment.iter().cloned() {
                w_slice[w_index.as_()] += plasticity;
            }
            w_to_increment.clear();
            w_to_increment
        });
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                            except that this function works with convolutional topology of weights W and the vectors
                            are sparse binary. Here input is X, output is Y, w_slice is W of shape self.kernel_columns_shape()==[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn sparse_unbiased_increment<D: Copy + Div<Output=D> + AddAssign + FromUsize>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        assert_eq!(w_slice.len(), self.w_shape().product().as_());
        let w_to_increment: Vec<Idx> = Vec::with_capacity(input.len());
        self.sparse_conjugate(input, output, w_to_increment, |mut w_to_increment, w_index| {
            w_to_increment.push(w_index);
            w_to_increment
        }, |mut w_to_increment, output_idx| {
            let plasticity = epsilon / D::from_usize(w_to_increment.len());
            for w_index in w_to_increment.iter().cloned() {
                w_slice[w_index.as_()] += plasticity;
            }
            w_to_increment.clear();
            w_to_increment
        });
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                                except that this function works with convolutional topology of weights W and the vectors
                                are sparse binary. Here input is X, output is Y, w_slice is W of shape self.kernel_columns_shape()==[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn sparse_biased_increment<D: Copy + AddAssign>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        assert_eq!(w_slice.len(), self.w_shape().product().as_());
        self.sparse_conjugate(input, output, (), |(), idx| w_slice[idx.as_()] += epsilon, |_, _| ())
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                                    except that this function works with convolutional topology of weights W and the vectors
                                    are sparse binary. Here input is X, output is Y, w_slice is W of shape self.minicolumn_w_shape()==[kernel_height, kernel_width, in_channels, out_channels]*/
    pub fn sparse_biased_increment_repeated<D: Copy + AddAssign>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        assert_eq!(w_slice.len(), self.minicolumn_w_shape().product().as_());
        self.sparse_conjugate_repeated(input, output, (), |(), idx| w_slice[idx.as_()] += epsilon, |_, _| ())
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                                except that this function works with convolutional topology of weights W and the vectors
                                are sparse binary. Here input is X, output is Y, w_slice is W of shape self.kernel_columns_shape()==[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn sparse_mul_assign<D: Copy + MulAssign>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        assert_eq!(w_slice.len(), self.w_shape().product().as_());
        self.sparse_conjugate(input, output, (), |(), idx| w_slice[idx.as_()] *= epsilon, |_, _| ())
    }
    /**It works like XWY where X is a row vector, W is a matrix and Y is a column vector,
                                    except that this function works with convolutional topology of weights W and the vectors
                                    are sparse binary. Here input is X, output is Y, w_slice is W of shape self.minicolumn_w_shape()==[kernel_height, kernel_width, in_channels, out_channels]*/
    pub fn sparse_mul_assign_repeated<D: Copy + MulAssign>(&self, w_slice: &mut [D], epsilon: D, input: &[Idx], output: &[Idx]) where Idx: AsPrimitive<usize> {
        assert_eq!(w_slice.len(), self.minicolumn_w_shape().product().as_());
        self.sparse_conjugate_repeated(input, output, (), |(), idx| w_slice[idx.as_()] *= epsilon, |_, _| ())
    }

    pub fn compose(&self, next: &Self) -> Self {
        assert_eq!(self.out_shape(), next.in_shape());
        let (kernel, stride) = conv::compose(self.stride(), self.kernel(), next.stride(), next.kernel());
        Self {
            input_shape: self.input_shape(),
            output_shape: next.output_shape(),
            kernel,
            stride,
        }
    }

    pub fn new_identity(shape: [Idx; 3]) -> Self {
        Self {
            input_shape: shape.clone(),
            output_shape: shape,
            kernel: [Idx::one(); 2],
            stride: [Idx::one(); 2],
        }
    }
    /**This convolution is in fact just a dense linear layer with certain number of inputs and outputs.*/
    pub fn new_linear(input: Idx, output: Idx) -> Self {
        Self {
            input_shape: [Idx::one(), Idx::one(), input],
            output_shape: [Idx::one(), Idx::one(), output],
            kernel: [Idx::one(); 2],
            stride: [Idx::one(); 2],
        }
    }
    pub fn new(output: [Idx; 2], kernel: [Idx; 2], stride: [Idx; 2], in_channels: Idx, out_channels: Idx) -> Self {
        Self::new_out(in_channels, xy_z3(output, out_channels), kernel, stride)
    }
    pub fn concat(layers: &[Self]) -> Self where Idx: Sum {
        assert_ne!(layers.len(), 0, "No layers provided!");
        let first_layer = &layers[0];
        let mut out_shape = first_layer.output_shape();
        let in_shape = first_layer.input_shape();
        let kernel = first_layer.kernel.clone();
        let stride = first_layer.stride.clone();
        assert!(layers.iter().all(|a| a.in_shape().all_eq(&in_shape)), "All concatenated layers must have the same input shape!");
        assert!(layers.iter().all(|a| a.out_grid().all_eq(grid(&out_shape))), "All concatenated layers must have the same output width and height!");
        assert!(layers.iter().all(|a| a.stride().all_eq(&stride)), "All concatenated layers must have the same stride!");
        assert!(layers.iter().all(|a| a.kernel().all_eq(&kernel)), "All concatenated layers must have the same kernel!");
        let concatenated_sum: Idx = layers.iter().map(|a| a.out_channels()).sum();
        out_shape[2] = concatenated_sum;
        Self {
            input_shape: in_shape,
            output_shape: out_shape,
            kernel,
            stride,
        }
    }

    pub fn new_in(input_shape: [Idx; 3],
                  out_channels: Idx,
                  kernel: [Idx; 2],
                  stride: [Idx; 2]) -> Self {
        Self {
            input_shape,
            output_shape: xy_z3(conv::out_size(&grid(&input_shape), &stride, &kernel), out_channels),
            kernel,
            stride,
        }
    }
    pub fn new_out(in_channels: Idx,
                   output_shape: [Idx; 3],
                   kernel: [Idx; 2],
                   stride: [Idx; 2]) -> Self {
        Self {
            input_shape: xy_z3(conv::in_size(&grid(&output_shape), &stride, &kernel), in_channels),
            output_shape,
            kernel,
            stride,
        }
    }
    pub fn set_stride(&mut self, new_stride: [Idx; 2]) {
        let input = conv::in_size(self.out_grid(), &new_stride, self.kernel());
        let input = xy_z3(input, self.in_channels());
        self.input_shape = input;
        self.stride = new_stride;
    }
    /**Input weights are of shape [kernel_height, kernel_width, in_channels, out_channels]. Output is
        [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]*/
    pub fn repeat_minicolumn<D: Copy>(&self, weights: &[D]) -> Vec<D> where Idx: AsPrimitive<usize> {
        let i_volume = self.kernel_column_volume().as_();
        let o_area = self.out_area().as_();
        let channels = self.out_channels().as_();
        assert_eq!(weights.len(), i_volume * channels);
        let mut conv_weights = Vec::<D>::empty(i_volume * o_area * channels);
        for i in 0..i_volume {
            for o in 0..o_area {
                for c in 0..channels {
                    conv_weights[(i * o_area + o) * channels + c] = weights[i * channels + c]
                }
            }
        }
        conv_weights
    }
    /**Input u_weights are of shape [out_height, out_width, out_channels, out_channels].
    y_dense and s are of shape [out_height, out_width, out_channels].
    y_sparse is a sparse vector encoding y_dense.
     The formula is a lambda function that takes |s[y,x,k],s[y,x,j],y[y,x,j],u[y,x,k,j]| and produces new value of u[y,x,k,j]. This function can
     safely assume that y[y,x,k]==true. The updates should be Hebbian, thus they do not need access to any other variables than those listed as lambda parameters above.*/
    pub fn update_u<D: Copy>(&self, s: &[D], y_sparse: &[Idx], y_dense: impl Fn(usize)->bool, u_weights: &mut [D], formula:impl Fn(D,D,bool,D)->D) where Idx: AsPrimitive<usize> {
        let o_area = self.out_area().as_();
        let channels = self.out_channels().as_();
        let o_vol = o_area * channels;
        assert_eq!(u_weights.len(), o_vol * channels);
        assert_eq!(s.len(), o_vol);
        for k in y_sparse.iter().map(|k|k.as_()){
            let k_yx = k / channels;
            let sk = s[k];
            for c in 0..channels{
                let j = k_yx*channels+c;
                let yj = y_dense(j);
                let sj = s[j];
                u_weights[k*channels + j] = formula(sk,sj,yj,u_weights[k*channels + j]);
            }
        }
    }
    /**Input u_weights are of shape [out_height, out_width, out_channels, out_channels].
            y is a sparse vector of output activations. k and s are of shape [out_height, out_width, out_channels]*/
    pub fn update_u_as_expected_sk_minus_sj<D: Copy+Add<Output=D>+Sub<Output=D>+Mul<Output=D>+One>(&self, epsilon:D, s: &[D], y: &[Idx], u_weights: &mut [D]) where Idx: AsPrimitive<usize> {
        let one_minus_eps = D::one() - epsilon;
        self.update_u(s,y,|_|false,u_weights,|sk,sj,yj,ukj|ukj*one_minus_eps+(sk-sj)*epsilon)
    }
    /**Input u_weights are of shape [out_channels, out_channels].
        y_dense and s are of shape [out_height, out_width, out_channels].
        y_sparse is a sparse vector encoding y_dense.
         The formula is a lambda function that takes |s[y,x,k],s[y,x,j],y[y,x,j],u[y,x,k,j]| and produces new value of u[y,x,k,j]. This function can
         safely assume that y[y,x,k]==true. The updates should be Hebbian, thus they do not need access to any other variables than those listed as lambda parameters above.
     The outputs of the formula are summed together using accumulate lambda.*/
    pub fn update_u_repeated<D: Copy+Zero+AddAssign>(&self, s: &[D], y_sparse: &[Idx], y_dense: impl Fn(usize)->bool, u_weights: &mut [D],
                                                                             formula:impl Fn(D,D,bool,D)->D,
                                                                             accumulate: impl Fn(D,D,usize)->D) where Idx: AsPrimitive<usize> {
        let o_area = self.out_area().as_();
        let channels = self.out_channels().as_();
        let o_vol = o_area * channels;
        let c2 = channels * channels;
        assert_eq!(u_weights.len(), c2, "U tensor has wrong shape");
        assert_eq!(s.len(), o_vol, "s tensor has wrong shape");
        let mut u_updates = vec![D::zero();c2];
        let mut win_count = vec![0usize; channels];
        for k in y_sparse.iter().map(|k|k.as_()){
            win_count[k] += 1;
            let k_yx = k / channels;
            let k_c = k % channels;
            let sk = s[k];
            for c in 0..channels{
                let j = k_yx*channels+c;
                let yj = y_dense(j);
                let sj = s[j];
                u_updates[k_c*channels + j] += formula(sk,sj,yj,u_weights[k_c*channels + j])
            }
        }
        for k in 0..channels{
            let wins = win_count[k];
            for j in 0..channels {
                let j = k*channels+j;
                u_weights[j] = accumulate(u_updates[j], u_weights[j], wins);
            }
        }

    }
    /**Input u_weights are of shape [out_channels, out_channels].
                y is a sparse vector of output activations. k and s are of shape [out_height, out_width, out_channels]*/
    pub fn update_u_as_expected_sk_minus_sj_repeated<D: Copy+AddAssign+Zero+Add<Output=D>+Sub<Output=D>+Mul<Output=D>+One+Div<Output=D>+FromUsize>(&self, epsilon:D, s: &[D], y: &[Idx], u_weights: &mut [D]) where Idx: AsPrimitive<usize> {
        let one_minus_eps = D::one() - epsilon;
        self.update_u_repeated(s,y,|_|false,u_weights,|sk,sj,yj,ukj|sk-sj,
                      |new_ukj_sum,old_ukj,k_win_count|old_ukj*one_minus_eps+(new_ukj_sum/D::from_usize(k_win_count))*epsilon)
    }
    /**[out_height,out_width,out_channels,kernel_height, kernel_width, in_channels]*/
    pub fn receptive_field_shape(&self)->[Idx;6]{
        let [oh,ow,oc] = self.output_shape;
        let [kh,kw] = self.kernel;
        [oh,ow,oc,kh,kw,self.in_channels()]
    }
    /**[out_channels,kernel_height, kernel_width, in_channels]*/
    pub fn minicolumn_receptive_field_shape(&self)->[Idx;4]{
        let [kh,kw] = self.kernel;
        [self.out_channels(),kh,kw,self.in_channels()]
    }
    /**minicolumn_receptive_field is of shape [out_channels,kernel_height, kernel_width, in_channels]*/
    pub fn add_to_receptive_field_repeated<D:AddAssign+One>(&self, minicolumn_receptive_field:&mut [D], x:&[Idx], y: &[Idx]) where Idx: AsPrimitive<usize> {
        let kc = self.kernel_column_shape().as_scalar::<usize>();
        let c = self.out_channels().as_();
        let kcv = kc.product();
        assert_eq!(minicolumn_receptive_field.len(),kcv*c);
        let is = self.in_shape().as_scalar::<usize>();
        let os = self.out_shape().as_scalar::<usize>();
        let stride = self.stride().as_scalar::<usize>();
        let kernel = self.kernel().as_scalar::<usize>();
        let x_pos:Vec<[usize;3]> = x.iter().map(|i|is.pos(i.as_())).collect();
        for j in y{
            debug_assert!(j.as_()<self.out_volume().as_());
            let j_pos = os.pos(j.as_());
            let i_range = conv::in_range(xy3(&j_pos),&stride,&kernel);
            for i_pos in &x_pos{
                debug_assert!(j.as_()<self.in_volume().as_());
                if vec_range::contains(&i_range,xy3(i_pos)){
                    let i_pos_within_kernel_column = sub_kernel_offset(i_pos,&i_range.start);
                    let i_within_kernel_column = kc.idx(&i_pos_within_kernel_column);
                    minicolumn_receptive_field[j_pos[2]*kcv + i_within_kernel_column] += D::one();
                }
            }
        }
    }

    // fn batch_sum_x<T, O>(&self, input: &[T], output: &[O], f: impl Fn(&T) -> &[Idx] + Send + Sync, of: impl Fn(&O) -> &[Idx] + Send + Sync) -> ConvTensor<f32>{
    //     let mut q: Vec<ConvTensor<f32>> = (0..num_cpus::get()).map(|_| ConvTensor::new(self.clone(),0.)).collect();
    //     parallel_iter_vectors(input,output,&mut q, |i,o,q|{
    //         let i = f(i);
    //         for &o in of(o).iter(){
    //             for &i in i.iter(){
    //                 q.as_slice_mut()[self.idx_within_kernel()]
    //             }
    //         }
    //     });
    //     let mut sum = q.pop().unwrap();
    //     for q in q{
    //         sum.add_assign(&q)
    //     }
    //     sum
    // }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use super::*;
    use rand::{random, SeedableRng};
    use crate::init_rand::InitRandWithCapacity;
    use crate::{rand_set, VectorFieldAddOwned, VectorFieldRemOwned};
    #[test]
    fn test1() {
        let S = 5;
        let K = 5;
        let shape = ConvShape::new_in([S, S, 3], 8, [K, K], [1, 1]);
        let kc = shape.kernel_column_shape();
        let kcv = kc.product();
        let is = shape.in_shape();
        let os = shape.out_shape();
        for _ in 0..50 {
            let s = Vec::<f32>::rand( shape.out_volume());
            let kernel_column = rand_set(8, 0..kcv);
            let offset = xy_z3(random::<[usize;2]>()._rem_scalar(S-K+1),0);
            let k = random::<usize>() % 8;
            let x:Vec<usize> = kernel_column.iter().map(|&i|is.idx(&kc.pos(i)._add(&offset))).collect();
            let y = vec![os.idx(&[offset[0],offset[1],k])];
            let rfv = shape.minicolumn_receptive_field_shape().product();
            let mut rf = vec![0.;rfv];
            shape.add_to_receptive_field_repeated(&mut rf, &x, &y);
            let mut kernel_column_dense = vec![0.;rfv];
            for i in kernel_column{
                kernel_column_dense[k*kcv+i]=1.;
            }
            assert_eq!(rf,kernel_column_dense,"k={}",k);
        }
    }
    #[test]
    fn test2() {
        let shape = ConvShape::new_in([5, 5, 3], 8, [5, 5], [1, 1]);
        let epsilon = 0.1;
        for _ in 0..50 {
            let s = Vec::<f32>::rand( shape.out_volume());
            let y = rand_set(8, 0..shape.out_volume());
            let mut u: Vec<f32> = Vec::rand( shape.u_shape().product());
            let mut u2: Vec<f32> = u.clone();
            shape.update_u_as_expected_sk_minus_sj_repeated(epsilon, &s, &y,&mut u);
            shape.update_u_as_expected_sk_minus_sj(epsilon, &s, &y,&mut u2);
            assert_eq!(u, u2);
        }
    }

    #[test]
    fn test3() {
        let shape = ConvShape::new_in([28, 28, 3], 3, [3, 3], [1, 1]);
        let epsilon = 2.;
        for _ in 0..50 {
            let x = rand_set(8, 0..shape.in_volume());
            let y = rand_set(8, 0..shape.out_volume());
            let mut mini_w: Vec<f32> = vec![1.; shape.minicolumn_w_shape().product()];
            let mut w: Vec<f32> = vec![1.; shape.w_shape().product()];
            shape.sparse_mul_assign_repeated(&mut mini_w, epsilon, &x, &y);
            shape.sparse_mul_assign(&mut w, epsilon, &x, &y);
            let oa = shape.out_area();
            for i in 0..shape.kernel_column_volume() {
                for c in 0..shape.out_channels() {
                    let mut sum = 1.;
                    for o in 0..shape.out_area() {
                        sum *= w[(i * oa + o) * shape.out_channels() + c];
                    }
                    assert_eq!(sum, mini_w[i * shape.out_channels() + c]);
                }
            }
        }
    }

    #[test]
    fn test4() {
        let shape = ConvShape::new_in([28, 28, 3], 3, [3, 3], [1, 1]);
        let epsilon = 0.01;
        for _ in 0..50 {
            let x = rand_set(8, 0..shape.in_volume());
            let y = rand_set(8, 0..shape.out_volume());
            let mut mini_w: Vec<f32> = vec![0.; shape.minicolumn_w_shape().product()];
            let mut w: Vec<f32> = vec![0.; shape.w_shape().product()];
            shape.sparse_unbiased_increment_repeated(&mut mini_w, epsilon, &x, &y);
            shape.sparse_unbiased_increment(&mut w, epsilon, &x, &y);
            let oa = shape.out_area();
            for i in 0..shape.kernel_column_volume() {
                for c in 0..shape.out_channels() {
                    let mut sum = 0.;
                    for o in 0..shape.out_area() {
                        sum += w[(i * oa + o) * shape.out_channels() + c];
                    }
                    assert_eq!(sum, mini_w[i * shape.out_channels() + c]);
                }
            }
        }
    }

    #[test]
    fn test5() {
        let mut rng: StdRng = SeedableRng::seed_from_u64(325);
        let shape = ConvShape::new_in([28, 28, 3], 3, [3, 3], [1, 1]);
        let epsilon = 0.01;
        for _ in 0..50 {
            let x = rand_set(8, 0..shape.in_volume());
            let y = rand_set(8, 0..shape.out_volume());
            let mut mini_w: Vec<f32> = vec![0.; shape.minicolumn_w_shape().product()];
            let mut w: Vec<f32> = vec![0.; shape.w_shape().product()];
            shape.sparse_biased_increment_repeated(&mut mini_w, epsilon, &x, &y);
            shape.sparse_biased_increment(&mut w, epsilon, &x, &y);
            let oa = shape.out_area();
            for i in 0..shape.kernel_column_volume() {
                for c in 0..shape.out_channels() {
                    let mut sum = 0.;
                    for o in 0..shape.out_area() {
                        sum += w[(i * oa + o) * shape.out_channels() + c];
                    }
                    assert_eq!(sum, mini_w[i * shape.out_channels() + c]);
                }
            }
        }
    }

    #[test]
    fn test6() {
        let mut rng: StdRng = SeedableRng::seed_from_u64(325);
        let shape = ConvShape::new_in([28, 28, 3], 3, [3, 3], [1, 1]);
        let mini_w: Vec<f32> = Vec::rand(shape.minicolumn_w_shape().product());
        let w: Vec<f32> = shape.repeat_minicolumn(&mini_w);
        for _ in 0..50 {
            let x = rand_set(8, 0..shape.in_volume());
            let o0 = shape.sparse_dot_repeated_slice(&x, &mini_w);
            let o1 = shape.sparse_dot_slice(&x, &w);
            assert_eq!(o0, o1);
        }
    }
}