#![feature(slice_flatten)]

mod util;

use rand_distr::Distribution;
use std::ops::{Deref, DerefMut, Range};
use std::str::FromStr;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray6, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::{wrap_pyfunction, wrap_pymodule, PyNativeType};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use pyo3::types::PyList;
use rand::Rng;
use vf::soft_wta::*;
use vf::{ArrayCast, conv, VecCast, VectorField, VectorFieldDivAssign, VectorFieldMul, VectorFieldMulAssign, VectorFieldOne, VectorFieldZero};
use vf::{arr2, arr3, slice_as_arr, tup2, tup3, tup4, tup6};
use vf::ecc_layer::Layer;
use vf::init::InitEmptyWithCapacity;
use vf::top_k::argsort;
use crate::util::{arrX, pickle, py_any_as_numpy, unpickle};


#[pyfunction]
pub fn conv_out_size(py: Python, input_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let input_size = arrX(py, &input_size, 1, 1, 1)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let out_size = conv::out_size(&input_size, &stride, &kernel);
    Ok(out_size.to_vec())
}

#[pyfunction]
pub fn conv_in_size(py: Python, output_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let output_size = arrX(py, &output_size, 1, 1, 1)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let in_size = conv::in_size(&output_size, &stride, &kernel);
    Ok(in_size.to_vec())
}

#[pyfunction]
pub fn conv_in_range_with_custom_size(py: Python, output_pos: PyObject, output_size: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let output_pos = arrX(py, &output_pos, 0, 0, 0)?;
    let output_size = arrX(py, &output_size, 1, 1, 1)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let in_range = conv::in_range_with_custom_size(&output_pos, &output_size, &stride, &kernel);
    Ok((in_range.start.to_vec(), in_range.end.to_vec()))
}

#[pyfunction]
pub fn conv_in_range(py: Python, output_pos: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let output_pos = arrX(py, &output_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let in_range = conv::in_range(&output_pos, &stride, &kernel);
    Ok((in_range.start.to_vec(), in_range.end.to_vec()))
}

#[pyfunction]
pub fn conv_out_range(py: Python, input_pos: PyObject, stride: PyObject, kernel: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let input_pos = arrX(py, &input_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let out_range = conv::out_range_clipped(&input_pos, &stride, &kernel);
    Ok((out_range.start.to_vec(), out_range.end.to_vec()))
}

#[pyfunction]
pub fn conv_out_range_clipped_both_sides(py: Python, input_pos: PyObject, stride: PyObject, kernel: PyObject, max_bounds: PyObject) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let input_pos = arrX(py, &input_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let max_bounds = arrX(py, &max_bounds, 0, 0, 0)?;
    let out_range = conv::out_range_clipped_both_sides(&input_pos, &stride, &kernel, &max_bounds);
    Ok((out_range.start.to_vec(), out_range.end.to_vec()))
}

#[pyfunction]
pub fn conv_in_range_begin(py: Python, output_pos: PyObject, stride: PyObject) -> PyResult<Vec<u32>> {
    let output_pos = arrX(py, &output_pos, 0, 0, 0)?;
    let stride = arrX(py, &stride, 1, 1, 1)?;
    let begin = conv::in_range_begin(&output_pos, &stride);
    Ok(begin.to_vec())
}

#[pyfunction]
pub fn conv_stride(py: Python, input_size: PyObject, output_size: PyObject, kernel: PyObject) -> PyResult<Vec<u32>> {
    let input_size = arrX(py, &input_size, 1, 1, 1)?;
    let output_size = arrX(py, &output_size, 1, 1, 1)?;
    let kernel = arrX(py, &kernel, 1, 1, 1)?;
    let stride = conv::stride(&input_size, &output_size, &kernel);
    Ok(stride.to_vec())
}

#[pyfunction]
///strides:[(int,int,int?)],kernels:[(int,int,int?)]  -> stride:(int,int,int), kernel:(int,int,int)
pub fn conv_compose_array(py: Python, strides: Vec<PyObject>, kernels: Vec<PyObject>) -> PyResult<((u32, u32, u32), (u32, u32, u32))> {
    assert_eq!(strides.len(), kernels.len());
    let (mut kernel, mut stride) = ([1; 3], [1; 3]);
    for (s, k) in strides.into_iter().zip(kernels.into_iter()) {
        let s = arrX(py, &s, 1, 1, 1)?;
        let k = arrX(py, &k, 1, 1, 1)?;
        (stride, kernel) = conv::compose(&stride, &kernel, &s, &k);
    }
    Ok((vf::tup3(stride), vf::tup3(kernel)))
}

#[pyfunction]
pub fn conv_compose(py: Python, stride1: PyObject, kernel1: PyObject, stride2: PyObject, kernel2: PyObject) -> PyResult<((u32, u32, u32), (u32, u32, u32))> {
    let stride1 = arrX(py, &stride1, 1, 1, 1)?;
    let kernel1 = arrX(py, &kernel1, 1, 1, 1)?;
    let stride2 = arrX(py, &stride2, 1, 1, 1)?;
    let kernel2 = arrX(py, &kernel2, 1, 1, 1)?;
    let (stride, kernel) = conv::compose(&stride1, &kernel1, &stride2, &kernel2);
    Ok((vf::tup3(stride), vf::tup3(kernel)))
}

#[pyfunction]
pub fn conv_compose_weights<'py>(stride1: [usize; 2], weights1: &'py PyArray4<f32>, bias1: &'py PyArray1<f32>,
                                 stride2: [usize; 2], weights2: &'py PyArray4<f32>, bias2: &'py PyArray1<f32>) -> PyResult<([usize; 2], &'py PyArray4<f32>, &'py PyArray1<f32>)> {
    if let [self_out_channels, self_in_channels, self_kernel0, self_kernel1] = *weights1.shape() {
        if let [next_out_channels, next_in_channels, next_kernel0, next_kernel1] = *weights2.shape() {
            assert_eq!(self_out_channels, next_in_channels, "self_out_channels != next_in_channels");
            let self_kernel = [self_kernel0, self_kernel1];
            let self_weights: &[f32] = unsafe { weights1.as_slice()? };
            let self_bias: &[f32] = unsafe { bias1.as_slice()? };
            let next_kernel = [next_kernel0, next_kernel1];
            let next_weights: &[f32] = unsafe { weights2.as_slice()? };
            let next_bias: &[f32] = unsafe { bias2.as_slice()? };
            assert_eq!(self_out_channels, self_bias.len(), "self_out_channels != self_bias.len()");
            assert_eq!(next_out_channels, next_bias.len(), "next_out_channels != next_bias.len()");
            let py = weights1.py();
            let (comp_stride, comp_kernel) = conv::compose(&stride1, &self_kernel, &stride2, &next_kernel);
            let mut comp_weigths = unsafe { PyArray4::<f32>::new(py, [next_out_channels, self_in_channels, comp_kernel[0], comp_kernel[1]], false) };
            let mut comp_bias = unsafe { PyArray1::<f32>::new(py, [next_out_channels], false) };
            let w_comp = unsafe { comp_weigths.as_slice_mut()? };
            w_comp.fill(0.);
            conv::compose_weights2d(self_in_channels, &stride1, &self_kernel, self_weights, self_bias,
                                    &next_kernel, next_weights, next_bias,
                                    &comp_kernel, w_comp, unsafe { comp_bias.as_slice_mut()? });
            return Ok((comp_stride, comp_weigths, comp_bias));
        }
    }
    unreachable!();
}


type Idx = u32;

///
/// ConvShape(output: (int,int), kernel: (int,int), stride: (int,int), in_channels: int, out_channels: int)
///
#[pyclass]
pub struct ConvShape {
    pub(crate) cs: vf::conv_shape::ConvShape<Idx>,
}

#[pymethods]
impl ConvShape {
    ///[out_height, out_width, out_channels, out_channels]
    #[getter]
    pub fn u_shape(&self) -> (Idx, Idx, Idx, Idx) { tup4(self.cs.u_shape()) }
    ///[out_channels, out_channels]
    #[getter]
    pub fn minicolumn_u_shape(&self) -> (Idx, Idx) { tup2(self.cs.minicolumn_u_shape()) }
    ///[kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    #[getter]
    pub fn w_shape(&self) -> (Idx, Idx, Idx, Idx, Idx, Idx) { tup6(self.cs.w_shape()) }
    ///[out_height, out_width, out_channels]
    #[getter]
    pub fn out_shape(&self) -> (Idx, Idx, Idx) { tup3(self.cs.output_shape()) }
    ///[in_height, in_width, in_channels]
    #[getter]
    pub fn in_shape(&self) -> (Idx, Idx, Idx) { tup3(self.cs.input_shape()) }
    ///[kernel_height, kernel_width]
    #[getter]
    pub fn kernel(&self) -> (Idx, Idx) { tup2(self.cs.kernel().clone()) }
    #[getter]
    pub fn kernel_height(&self) -> Idx { self.cs.kernel_height() }
    #[getter]
    pub fn kernel_width(&self) -> Idx { self.cs.kernel_width() }
    pub fn rand_dense_input(&self, cardinality: Idx) -> Vec<bool> {
        self.cs.rand_dense_input(cardinality)
    }
    pub fn rand_dense_output(&self, cardinality: Idx) -> Vec<bool> {
        self.cs.rand_dense_output(cardinality)
    }
    pub fn rand_sparse_input(&self, cardinality: Idx) -> Vec<Idx> {
        self.cs.rand_sparse_input(cardinality)
    }
    pub fn rand_sparse_output(&self, cardinality: Idx) -> Vec<Idx> {
        self.cs.rand_sparse_output(cardinality)
    }
    ///[height, width]
    #[getter]
    pub fn stride(&self) -> (Idx, Idx) { tup2(self.cs.stride().clone()) }
    ///[kernel_height, kernel_width, in_channels]
    /// Kernel column is the shape of receptive field of each output neuron. Don't confuse it with
    /// minicolumn which consists of all the output neurons that have the same receptive field.
    #[getter]
    pub fn kernel_column_shape(&self) -> (Idx, Idx, Idx) { tup3(self.cs.kernel_column_shape()) }
    ///[kernel_height, kernel_width, in_channels, out_channels]
    /// This is the shape of weight tensor that constitutes weights of a single minicolumn.
    #[getter]
    pub fn minicolumn_w_shape(&self) -> (Idx, Idx, Idx, Idx) { tup4(self.cs.minicolumn_w_shape()) }
    /// kernel_height * kernel_width
    #[getter]
    pub fn kernel_column_area(&self) -> Idx { self.cs.kernel_column_area() }
    /// kernel_height * kernel_width * in_channels
    #[getter]
    pub fn kernel_column_volume(&self) -> Idx { self.cs.kernel_column_volume() }
    ///[in_height, in_width]
    #[getter]
    pub fn in_grid(&self) -> (Idx, Idx) { tup2(self.cs.in_grid().clone()) }
    #[getter]
    ///[out_height, out_width]
    pub fn out_grid(&self) -> (Idx, Idx) { tup2(self.cs.out_grid().clone()) }
    #[getter]
    pub fn out_width(&self) -> Idx { self.cs.out_width() }
    #[getter]
    pub fn out_height(&self) -> Idx { self.cs.out_height() }
    #[getter]
    pub fn out_channels(&self) -> Idx { self.cs.out_channels() }
    #[getter]
    pub fn in_width(&self) -> Idx { self.cs.in_width() }
    #[getter]
    pub fn in_height(&self) -> Idx { self.cs.in_height() }
    #[getter]
    pub fn in_channels(&self) -> Idx { self.cs.in_channels() }
    #[getter]
    pub fn out_area(&self) -> Idx { self.cs.out_area() }
    #[getter]
    pub fn in_area(&self) -> Idx { self.cs.in_area() }
    #[getter]
    pub fn out_volume(&self) -> Idx { self.cs.out_volume() }
    #[getter]
    pub fn in_volume(&self) -> Idx { self.cs.in_volume() }
    pub fn kernel_offset(&self, output_pos: (Idx, Idx, Idx)) -> (Idx, Idx) { tup2(self.cs.kernel_offset(&arr3(output_pos))) }
    pub fn pos_within_kernel(&self, input_pos: (Idx, Idx, Idx), output_pos: (Idx, Idx, Idx)) -> (Idx, Idx, Idx) { tup3(self.cs.pos_within_kernel(&arr3(input_pos), &arr3(output_pos))) }
    pub fn idx_within_kernel(&self, input_pos: (Idx, Idx, Idx), output_pos: (Idx, Idx, Idx)) -> Idx { self.cs.idx_within_kernel(&arr3(input_pos), &arr3(output_pos)) }
    #[getter]
    ///[out_height,out_width,out_channels,kernel_height, kernel_width, in_channels]
    pub fn receptive_field_shape(&self) -> (Idx, Idx, Idx, Idx, Idx, Idx) {
        tup6(self.cs.receptive_field_shape())
    }
    #[getter]
    ///[out_channels,kernel_height, kernel_width, in_channels]
    pub fn minicolumn_receptive_field_shape(&self) -> (Idx, Idx, Idx, Idx) {
        tup4(self.cs.minicolumn_receptive_field_shape())
    }
    ///((start_y,start_x),(end_y,end_x))
    pub fn in_range(&self, output_column_pos: (Idx, Idx)) -> ((Idx, Idx), (Idx, Idx)) {
        let Range { start, end } = self.cs.in_range(&arr2(output_column_pos));
        (tup2(start), tup2(end))
    }
    ///((start_y,start_x),(end_y,end_x))
    pub fn out_range(&self, input_pos: (Idx, Idx)) -> ((Idx, Idx), (Idx, Idx)) {
        let Range { start, end } = self.cs.out_range(&arr2(input_pos));
        (tup2(start), tup2(end))
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    pub fn normalize_kernel_columns(&self, conv_tensor: &PyArray6<f32>, norm: usize) {
        assert_eq!(conv_tensor.shape(), self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let rhs = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        self.cs.normalize_kernel_columns(rhs, vf::l(norm));
    }

    /// indices:[int], output_pos:(int,int)
    pub fn sparse_kernel_column_input_subset<'py>(&'py self, indices: &'py PyArray1<Idx>, output_pos: (Idx, Idx)) -> PyResult<&'py PyArray1<Idx>> {
        let output_pos = [output_pos.0, output_pos.1];
        let i = unsafe { indices.as_slice()? };
        let x = self.cs.sparse_kernel_column_input_subset(i, &output_pos);
        let o = PyArray1::from_vec(indices.py(), x);
        Ok(o)
    }
    /// indices:[int], output_pos:(int,int)
    pub fn sparse_kernel_column_input_subset_reindexed<'py>(&'py self, indices: &'py PyArray1<Idx>, output_pos: (Idx, Idx)) -> PyResult<&'py PyArray1<Idx>> {
        let output_pos = [output_pos.0, output_pos.1];
        let i = unsafe { indices.as_slice()? };
        let x = self.cs.sparse_kernel_column_input_subset_reindexed(i, &output_pos);
        let o = PyArray1::from_vec(indices.py(), x);
        Ok(o)
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    pub fn normalize_minicolumn(&self, conv_tensor: &PyArray4<f32>, norm: usize) {
        assert_eq!(conv_tensor.shape(), self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let rhs = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        self.cs.normalize_minicolumn(rhs, vf::l(norm));
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    pub fn sparse_repeated_normalize(&self, conv_tensor: &PyArray4<f32>, y: &PyArray1<u32>, norm: usize) {
        assert_eq!(conv_tensor.shape(), self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let rhs = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        let y = unsafe { y.as_slice_mut() }.expect("Output sparse tensor is not continuous");
        self.cs.sparse_repeated_normalize(rhs, y, vf::l(norm));
    }
    pub fn idx(&self, input_pos: (Idx, Idx, Idx), output_pos: (Idx, Idx, Idx)) -> Idx { self.cs.idx(&arr3(input_pos), &arr3(output_pos)) }
    /// rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels].
    /// lhs_tensor is a sparse binary vector (list of indices).
    /// dot_product_output is of shape [out_height, out_width, out_channels]
    pub fn sparse_dot<'py>(&self, lhs_tensor: &'py PyArray1<Idx>, rhs_conv_tensor: &'py PyArray6<f32>, dot_product_output: Option<&'py PyArray3<f32>>) -> &'py PyArray3<f32> {
        assert_eq!(rhs_conv_tensor.shape(), self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let lhs = unsafe { lhs_tensor.as_slice() }.expect("Lhs input tensor is not continuous");
        let rhs = unsafe { rhs_conv_tensor.as_slice() }.expect("Convolutional weights tensor is not continuous");
        let out_shape = self.cs.out_shape().as_scalar::<usize>();
        let out_tensor: &'py PyArray3<f32> = dot_product_output.unwrap_or_else(|| PyArray3::zeros(lhs_tensor.py(), out_shape, false));
        assert_eq!(out_tensor.shape(), &out_shape, "Output tensor shape is wrong");
        let out = unsafe { out_tensor.as_slice_mut() }.expect("Output tensor is not continuous");
        self.cs.sparse_dot_slice_(lhs, rhs, out);
        out_tensor
    }
    /// rhs_conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels].
    /// lhs_tensor is a sparse binary vector (list of indices).
    /// dot_product_output is of shape [out_height, out_width, out_channels]
    pub fn sparse_dot_repeated<'py>(&self, lhs_tensor: &'py PyArray1<Idx>, rhs_conv_tensor: &'py PyArray4<f32>, dot_product_output: Option<&'py PyArray3<f32>>) -> &'py PyArray3<f32> {
        assert_eq!(rhs_conv_tensor.shape(), self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let lhs = unsafe { lhs_tensor.as_slice() }.expect("Lhs input tensor is not continuous");
        let rhs = unsafe { rhs_conv_tensor.as_slice() }.expect("Convolutional weights tensor is not continuous");
        let out_shape = self.cs.out_shape().as_scalar::<usize>();
        let out_tensor: &'py PyArray3<f32> = dot_product_output.unwrap_or_else(|| PyArray3::zeros(lhs_tensor.py(), out_shape, false));
        assert_eq!(out_tensor.shape(), &out_shape, "Output tensor shape is wrong");
        let out = unsafe { out_tensor.as_slice_mut() }.expect("Output tensor is not continuous");
        self.cs.sparse_dot_repeated_slice_(lhs, rhs, out);
        out_tensor
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    /// input and output are sparse binary vectors (list of indices)
    pub fn sparse_mul_assign<'py>(&self, conv_tensor: &'py PyArray6<f32>, epsilon: f32, input: &'py PyArray1<Idx>, output: &'py PyArray1<Idx>) {
        assert_eq!(conv_tensor.shape(), self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe { input.as_slice() }.expect("Input tensor is not continuous");
        let out = unsafe { output.as_slice() }.expect("Output tensor is not continuous");
        let conv = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        self.cs.sparse_mul_assign(conv, epsilon, inp, out)
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    /// input and output are sparse binary vectors (list of indices)
    pub fn sparse_mul_assign_repeated<'py>(&self, conv_tensor: &'py PyArray4<f32>, epsilon: f32, input: &'py PyArray1<Idx>, output: &'py PyArray1<Idx>) {
        assert_eq!(conv_tensor.shape(), self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe { input.as_slice() }.expect("Input tensor is not continuous");
        let out = unsafe { output.as_slice() }.expect("Output tensor is not continuous");
        let conv = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        self.cs.sparse_mul_assign_repeated(conv, epsilon, inp, out)
    }

    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    pub fn sparse_increment<'py>(&self, conv_tensor: &'py PyArray6<f32>, epsilon: f32, input: &'py PyArray1<Idx>, output: &'py PyArray1<Idx>, biased: bool) {
        assert_eq!(conv_tensor.shape(), self.cs.w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe { input.as_slice() }.expect("Input tensor is not continuous");
        let out = unsafe { output.as_slice() }.expect("Output tensor is not continuous");
        let conv = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        if biased {
            self.cs.sparse_biased_increment(conv, epsilon, inp, out)
        } else {
            self.cs.sparse_unbiased_increment(conv, epsilon, inp, out)
        }
    }
    /// conv_tensor is of shape [kernel_height, kernel_width, in_channels, out_channels]
    pub fn sparse_increment_repeated<'py>(&self, conv_tensor: &'py PyArray4<f32>, epsilon: f32, input: &'py PyArray1<Idx>, output: &'py PyArray1<Idx>, biased: bool) {
        assert_eq!(conv_tensor.shape(), self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Convolutional tensor shape is wrong");
        let inp = unsafe { input.as_slice() }.expect("Input tensor is not continuous");
        let out = unsafe { output.as_slice() }.expect("Output tensor is not continuous");
        let conv = unsafe { conv_tensor.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        if biased {
            self.cs.sparse_biased_increment_repeated(conv, epsilon, inp, out)
        } else {
            self.cs.sparse_unbiased_increment_repeated(conv, epsilon, inp, out)
        }
    }
    /// u is of shape [out_height, out_width, out_channels, out_channels].
    /// y is a sparse vector of output activations. k and s are of shape [out_height, out_width, out_channels]
    pub fn update_u_as_expected_sk_minus_sj(&self, epsilon: f32, s: &PyArray3<f32>, y: &PyArray1<Idx>, u_weights: &PyArray4<f32>) {
        assert_eq!(u_weights.shape(), self.cs.u_shape().as_scalar::<usize>().as_slice(), "U tensor shape is wrong");
        assert_eq!(s.shape(), self.cs.out_shape().as_scalar::<usize>().as_slice(), "s tensor shape is wrong");
        let s = unsafe { s.as_slice() }.expect("Input tensor is not continuous");
        let y = unsafe { y.as_slice() }.expect("Output tensor is not continuous");
        let u = unsafe { u_weights.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        self.cs.update_u_as_expected_sk_minus_sj(epsilon, s, y, u)
    }
    /// u is of shape [out_channels, out_channels].
    /// y is a sparse vector of output activations. k and s are of shape [out_height, out_width, out_channels]
    pub fn update_u_as_expected_sk_minus_sj_repeated(&self, epsilon: f32, s: &PyArray3<f32>, y: &PyArray1<Idx>, u_weights: &PyArray2<f32>) {
        assert_eq!(u_weights.shape(), self.cs.minicolumn_u_shape().as_scalar::<usize>().as_slice(), "U tensor shape is wrong");
        assert_eq!(s.shape(), self.cs.out_shape().as_scalar::<usize>().as_slice(), "s tensor shape is wrong");
        let s = unsafe { s.as_slice() }.expect("Input tensor is not continuous");
        let y = unsafe { y.as_slice() }.expect("Output tensor is not continuous");
        let u = unsafe { u_weights.as_slice_mut() }.expect("Convolutional weights tensor is not continuous");
        self.cs.update_u_as_expected_sk_minus_sj_repeated(epsilon, s, y, u)
    }
    pub fn compose(&self, next: &Self) -> Self {
        Self { cs: self.cs.compose(&next.cs) }
    }
    #[staticmethod]
    pub fn new_identity(shape: (Idx, Idx, Idx)) -> Self {
        Self { cs: vf::conv_shape::ConvShape::new_identity(arr3(shape)) }
    }
    #[staticmethod]
    /**This convolution is in fact just a dense linear layer with certain number of inputs and outputs.*/
    pub fn new_linear(input: Idx, output: Idx) -> Self {
        Self { cs: vf::conv_shape::ConvShape::new_linear(input, output) }
    }
    #[new]
    pub fn new(output: (Idx, Idx), kernel: (Idx, Idx), stride: (Idx, Idx), in_channels: Idx, out_channels: Idx) -> Self {
        Self { cs: vf::conv_shape::ConvShape::new(arr2(output), arr2(kernel), arr2(stride), in_channels, out_channels) }
    }
    // #[staticmethod]
    // pub fn concat<'py>(layers: Vec<&'py ConvShape>) -> Self {
    //     let layers:Vec<vf::conv_shape::ConvShape<Idx>> = layers.iter().map(|cs|cs.cs.clone()).collect();
    //     Self{cs:vf::conv_shape::ConvShape::concat(layers.as_slice())}
    // }
    #[staticmethod]
    /// (input_shape:(int,int,int), out_channels:int, kernel:(int,int), stride:(int,int))
    /// where `input_shape==(height,width,channels)`
    pub fn new_in(input_shape: (Idx, Idx, Idx), out_channels: Idx, kernel: (Idx, Idx), stride: (Idx, Idx)) -> Self {
        Self { cs: vf::conv_shape::ConvShape::new_in(arr3(input_shape), out_channels, arr2(kernel), arr2(stride)) }
    }
    #[staticmethod]
    /// (in_channels:int, out_channels:(int,int,int), kernel:(int,int), stride:(int,int))
    /// where `out_channels==(height,width,channels)`
    pub fn new_out(in_channels: Idx, output_shape: (Idx, Idx, Idx), kernel: (Idx, Idx), stride: (Idx, Idx)) -> Self {
        Self { cs: vf::conv_shape::ConvShape::new_out(in_channels, arr3(output_shape), arr2(kernel), arr2(stride)) }
    }
    pub fn set_stride(&mut self, new_stride: (Idx, Idx)) {
        self.cs.set_stride(arr2(new_stride))
    }
    ///Input weights are of shape [kernel_height, kernel_width, in_channels, out_channels]. Output is [kernel_height, kernel_width, in_channels, out_height, out_width, out_channels]
    pub fn repeat_minicolumn<'py>(&'py self, weights: &'py PyArray4<f32>) -> PyResult<&'py PyArray6<f32>> {
        assert_eq!(weights.shape(), self.cs.minicolumn_w_shape().as_scalar::<usize>().as_slice(), "Weight tensor shape is wrong");
        let inp = unsafe { weights.as_slice()? };
        let out = self.cs.repeat_minicolumn(inp);
        let out = PyArray1::from_vec(weights.py(), out);
        let out = out.reshape(self.cs.w_shape().as_scalar::<usize>())?;
        Ok(out)
    }
    ///minicolumn_receptive_field is of shape [out_channels,kernel_height, kernel_width, in_channels]
    pub fn add_to_receptive_field_repeated(&self, minicolumn_receptive_field: &PyArray4<f32>, x: &PyArray1<Idx>, y: &PyArray1<Idx>) -> PyResult<()> {
        let minicolumn_receptive_field = unsafe { minicolumn_receptive_field.as_slice_mut()? };
        let x = unsafe { x.as_slice() }.expect("Input tensor is not continuous");
        let y = unsafe { y.as_slice() }.expect("Output tensor is not continuous");
        self.cs.add_to_receptive_field_repeated(minicolumn_receptive_field, x, y);
        Ok(())
    }
    fn __repr__(&self) -> String {
        format!("{:?}", &self.cs)
    }
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyfunction]
pub fn version<'py>() -> u32 {
    0
}

#[pyfunction]
/// Returns a pair of vectors (indices, offsets). First vector contains indices of all
/// true boolean values within each batch.
/// The second vector contains offsets to the first one. It works just like [[int]] but is flattened.
/// Batches are assumed to be laid out continuously in memory.
pub fn batch_sparse<'py>(bools: &'py PyArrayDyn<bool>) -> PyResult<(&'py PyArray1<Idx>, &'py PyArray1<usize>)> {
    assert!(bools.ndim() > 1, "Tensor must have at least 2 dimensions!");
    let b = unsafe { bools.as_slice()? };
    let batch_size = bools.shape()[1..].product();
    let (indices, offsets) = vf::batch_dense_to_sparse::<u32>(batch_size, b);
    let i = PyArray1::<u32>::from_vec(bools.py(), indices);
    let o = PyArray1::<usize>::from_vec(bools.py(), offsets);
    Ok((i, o))
}

#[pyfunction]
/// Returns a pair of vectors (indices, offsets). First vector contains indices of all
/// true boolean values within each batch.
/// The second vector contains offsets to the first one. It works just like `[[int]]` but is flattened.
/// Batches are assumed to be laid out continuously in memory.
pub fn join_sparse<'py>(py: Python<'py>, sparse: Vec<&'py PyArray1<Idx>>) -> PyResult<(&'py PyArray1<Idx>, &'py PyArray1<usize>)> {
    let slices: Result<Vec<&[Idx]>, _> = sparse.iter().map(|v| unsafe { v.as_slice() }).collect();
    let slices = slices?;
    let (indices, offsets) = vf::join_sets(&slices);
    let i = PyArray1::<u32>::from_vec(py, indices);
    let o = PyArray1::<usize>::from_vec(py, offsets);
    Ok((i, o))
}

#[pyfunction]
pub fn receptive_field<'py>(output_size: usize, output_indices: &'py PyArray1<Idx>, output_offsets: &'py PyArray1<usize>,
                            input_size: usize, input_indices: &'py PyArray1<Idx>, input_offsets: &'py PyArray1<usize>) -> PyResult<&'py PyArray2<f32>> {
    let oi = unsafe { output_indices.as_slice()? };
    let oo = unsafe { output_offsets.as_slice()? };
    let ii = unsafe { input_indices.as_slice()? };
    let io = unsafe { input_offsets.as_slice()? };
    let v = vf::ecc_layer::receptive_field(output_size, oi, oo, input_size, ii, io);
    PyArray1::from_vec(output_indices.py(), v).reshape((output_size, input_size))
}

#[pyfunction]
/// indices:[int], tensor_shape:(int,int,int), from_pos:(int,int,int), to_pos:(int,int,int)
pub fn sparse_subtensor<'py>(py: Python<'py>, indices: &'py PyArray1<Idx>, tensor_shape: (Idx, Idx, Idx), from_pos: (Idx, Idx, Idx), to_pos: (Idx, Idx, Idx)) -> PyResult<&'py PyArray1<Idx>> {
    let from_pos = [from_pos.0, from_pos.1, from_pos.2];
    let to_pos = [to_pos.0, to_pos.1, to_pos.2];
    let tensor_shape = [tensor_shape.0, tensor_shape.1, tensor_shape.2];
    let i = unsafe { indices.as_slice()? };
    let x = vf::sparse_subtensor(i, &tensor_shape, from_pos..to_pos);
    let o = PyArray1::from_vec(py, x);
    Ok(o)
}

#[pyfunction]
/// indices:[int], tensor_shape:(int,int,int), from_pos:(int,int,int), to_pos:(int,int,int)
pub fn sparse_subtensor_reindexed<'py>(py: Python<'py>, indices: &'py PyArray1<Idx>, tensor_shape: (Idx, Idx, Idx), from_pos: (Idx, Idx, Idx), to_pos: (Idx, Idx, Idx)) -> PyResult<&'py PyArray1<Idx>> {
    let from_pos = [from_pos.0, from_pos.1, from_pos.2];
    let to_pos = [to_pos.0, to_pos.1, to_pos.2];
    let tensor_shape = [tensor_shape.0, tensor_shape.1, tensor_shape.2];
    let i = unsafe { indices.as_slice()? };
    let x = vf::sparse_subtensor_reindexed(i, &tensor_shape, from_pos..to_pos);
    let o = PyArray1::from_vec(py, x);
    Ok(o)
}

#[pyfunction]
/// Returns a vector containing indices of all true boolean values
pub fn sparse(bools: &PyArrayDyn<bool>) -> PyResult<&PyArray1<u32>> {
    let b = unsafe { bools.as_slice()? };
    Ok(PyArray1::<u32>::from_vec(bools.py(), vf::dense_to_sparse(b)))
}

#[pyfunction]
/// Returns a vector containing indices of all true boolean values
pub fn dense(indices: &PyArrayDyn<u32>, length: usize) -> PyResult<&PyArray1<bool>> {
    let b = unsafe { indices.as_slice()? };
    Ok(PyArray1::<bool>::from_vec(indices.py(), vf::sparse_to_dense(b, length)))
}

#[pyfunction]
/// Returns a vector containing indices of all true boolean values
pub fn dense_(indices: &PyArrayDyn<u32>, output: &PyArrayDyn<bool>) -> PyResult<()> {
    let b = unsafe { indices.as_slice()? };
    let o = unsafe { output.as_slice_mut()? };
    vf::sparse_to_dense_(b, o);
    Ok(())
}

#[pyfunction]
/// Returns a boolean tensor randomly sampled according to probabilities contained in another tensor
pub fn sample(probabilities: &PyArrayDyn<f32>) -> PyResult<&PyArrayDyn<bool>> {
    let mut rng = rand::thread_rng();
    let b = unsafe { probabilities.as_slice()? };
    let mut d = unsafe { PyArrayDyn::new(probabilities.py(), probabilities.dims(), false) };
    let ds = unsafe { d.as_slice_mut() }.unwrap();
    for (&prob, sampled) in b.iter().zip(ds.iter_mut()) {
        *sampled = rng.gen::<f32>() < prob;
    }
    Ok(d)
}

#[pyfunction]
/// Returns a sparse boolean tensor of specified cardinality (number of ones) such that the top values get assigned 1.
/// Optionally (if std_dev is provided) the values can be treated as means of gaussian distributions with the provided standard deviation.
pub fn sample_of_cardinality(values: &PyArrayDyn<f32>, cardinality: usize, std_dev: Option<f32>) -> PyResult<&PyArrayDyn<bool>> {
    let p = unsafe { values.as_slice()? };
    let sorted_indices: Vec<usize> = if let Some(std_dev) = std_dev {
        let mut rng = rand::thread_rng();
        let mut dist = rand_distr::Normal::new(0f32, std_dev).map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        let tmp: Vec<f32> = p.iter().map(|v| v + dist.sample(&mut rng)).collect();
        argsort(&tmp, f32::total_cmp)
    } else {
        argsort(p, f32::total_cmp)
    };
    let mut d = PyArrayDyn::zeros(values.py(), values.dims(), false);
    let d_buff = unsafe { d.as_slice_mut() }.unwrap();
    for &i in sorted_indices.iter().take(cardinality) {
        d_buff[i] = true;
    }
    Ok(d)
}

#[pyfunction]
/// Returns a vector containing indices of all true boolean values
pub fn rand_sparse_k<'py>(py: Python<'py>, cardinality: u32, from_inclusive: u32, to_exclusive: u32) -> &'py PyArray1<u32> {
    PyArray1::from_vec(py, vf::rand_set(cardinality, from_inclusive..to_exclusive))
}

#[pyfunction]
/// Returns a vector containing indices of all true boolean values
pub fn rand_sparse_k_sorted<'py>(py: Python<'py>, cardinality: u32, from_inclusive: u32, to_exclusive: u32) -> &'py PyArray1<u32> {
    PyArray1::from_vec(py, vf::rand_set_sorted(cardinality, from_inclusive..to_exclusive))
}

#[pyfunction]
/// Returns a binary vector of specific length (number of 1s and 0s) and cardinality (number of 1s)
pub fn rand_dense_k<'py>(py: Python<'py>, cardinality: u32, length: u32) -> &'py PyArray1<bool> {
    PyArray1::from_vec(py, vf::rand_dense(cardinality, length))
}

#[pyfunction]
/// returns indices i at which `dense[i] > threshold`. Those indices are sorted by `dense[i]` in descending order.
pub fn sparse_gt<'py>(py: Python<'py>, dense: &'py PyArrayDyn<f32>, threshold: f32) -> PyResult<&'py PyArray1<u32>> {
    let p = unsafe { dense.as_slice()? };
    Ok(PyArray1::from_vec(py, vf::sparse_gt(p, threshold)))
}

#[pyfunction]
/// returns k indices corresponding to the k highest values in dense vector. Those indices i are sorted by dense[i] in descending order.
pub fn sparse_top_k<'py>(py: Python<'py>, dense: &'py PyArrayDyn<f32>, k: u32) -> PyResult<&'py PyArray1<u32>> {
    let p = unsafe { dense.as_slice()? };
    Ok(PyArray1::from_vec(py, vf::sparse_top_k(p, k)))
}

#[pyfunction]
/// returns k indices corresponding to the k highest values in dense vector. Those indices i are sorted by dense[i] in descending order.
pub fn dense_top_k<'py>(py: Python<'py>, dense: &'py PyArrayDyn<f32>, k: u32) -> PyResult<&'py PyArrayDyn<bool>> {
    let p = unsafe { dense.as_slice()? };
    PyArray1::from_vec(py, vf::sparse_to_dense(&vf::sparse_top_k(p, k), dense.len())).reshape(dense.shape())
}

#[pyfunction]
/// Dot product `x' * y` where `x` is a sparse binary vector. Both vectors `x` and `y` must be of equal length. Output is a scalar.
/// The input `sparse_vec` is a sprase representation (list of indices of turned-on bits) of `x`.
pub fn sparse_inner_product<'py>(sparse: &'py PyArray1<u32>, dense: &'py PyArray1<f32>) -> PyResult<f32> {
    let d = unsafe { dense.as_slice()? };
    let s = unsafe { sparse.as_slice()? };
    Ok(vf::dot_sparse_slice::inner_product(s, d))
}

#[pyfunction]
/// Dot product `x' * W` where `x` is a sparse binary vector. Shapes are `size(W)=(rows, cols)` and `size(x)=(rows)`. Output vector is of size `cols`.
/// The input `sparse_vec` is a sprase representation (list of indices of turned-on bits) of `x`.
pub fn sparse_dot<'py>(sparse: &'py PyArray1<u32>, dense: &'py PyArray2<f32>) -> PyResult<&'py PyArray1<f32>> {
    let shape = dense.shape();
    let cols = shape[1];
    let s = unsafe { sparse.as_slice()? };
    let d = unsafe { dense.as_slice()? };
    let output = vf::dot_sparse_slice::dot1(s, d, cols);
    Ok(PyArray1::from_vec(sparse.py(), output))
}

#[pyfunction]
/// Dot product `W * x` where `x` is a sparse binary vector. Shapes are `size(W)=(rows, cols)` and `size(x)=(cols)`. Output vector is of size `rows`.
/// The input `sparse_vec` is a sprase representation (list of indices of turned-on bits) of `x`.
pub fn sparse_dot_t<'py>(dense: &'py PyArray2<f32>, sparse: &'py PyArray1<u32>) -> PyResult<&'py PyArray1<f32>> {
    let shape = dense.shape();
    let rows = shape[0];
    let s = unsafe { sparse.as_slice()? };
    let d = unsafe { dense.as_slice()? };
    let output = vf::dot_sparse_slice::dot1_t(d, s, rows);
    Ok(PyArray1::from_vec(sparse.py(), output))
}


#[pyfunction]
/// Binary-valued ordered soft-winner-takes-all. V are the strengths of inhibitory connections, s are the delays at which neurons fired,
/// si is a list of indices of those neurons that fired sorted by the order in which they fired. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
pub fn ordered_swta_v<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>, si: &'py PyArray1<u32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = vf::soft_wta::ordered_top_slice(unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { si.as_slice()?.iter().cloned() });
    let w = winners.into_pyarray(v.py());
    Ok(w)
}

#[pyfunction]
/// Binary-valued ordered soft-winner-takes-all. V are the strengths of inhibitory connections, s are the delays at which neurons fired,
/// si is a list of indices of those neurons that fired sorted by the order in which they fired. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
pub fn ordered_swta_u<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, si: &'py PyArray1<u32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = vf::soft_wta::ordered_top_slice(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { si.as_slice()?.iter().cloned() });
    let w = winners.into_pyarray(u.py());
    Ok(w)
}

#[pyfunction]
/// v is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column).
pub fn swta_v<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = vf::soft_wta::top_slice(unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, |_| ());
    let w = winners.into_pyarray(v.py());
    Ok(w)
}

#[pyfunction]
/// u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column).
pub fn swta_u<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>) -> PyResult<&'py PyArray1<bool>> {
    let winners = vf::soft_wta::top_slice(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, |_| ());
    let w = winners.into_pyarray(u.py());
    Ok(w)
}

#[pyfunction]
/// u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column).
pub fn swta_u_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::top_slice_(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }, |_| ()))
}

#[pyfunction]
/// v is row-major. Element `v[k,j]==1` means neuron k (row) can inhibit neuron j (column).
pub fn swta_v_<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::top_slice_(unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }, |_| ()))
}

#[pyfunction]
/// u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column).
pub fn ordered_swta_u_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray1<f32>, si: &'py PyArray1<u32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::ordered_top_slice_(unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { si.as_slice()?.iter().cloned() }, unsafe { y.as_slice_mut()? }, |_| ()))
}

#[pyfunction]
/// v is row-major. Element `v[k,j]==1` means neuron k (row) can inhibit neuron j (column).
pub fn ordered_swta_v_<'py>(v: &'py PyArray2<bool>, s: &'py PyArray1<f32>, si: &'py PyArray1<u32>, y: &'py PyArray1<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::ordered_top_slice_(unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { si.as_slice()?.iter().cloned() }, unsafe { y.as_slice_mut()? }, |_| ()))
}


#[pyfunction]
/// u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of u is [height, width, channels, channels],
/// shape of y is [height, width, channels].
pub fn swta_u_conv_<'py>(u: &'py PyArray4<f32>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::top_conv_(slice_as_arr(y.shape()), unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }, |_| ()))
}

#[pyfunction]
/// v is row-major. Element `v[j0,j1,k,j]==1` means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of v is [height, width, channels, channels],
/// shape of y is [height, width, channels].
pub fn swta_v_conv_<'py>(v: &'py PyArray4<bool>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::top_conv_(slice_as_arr(y.shape()), unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }, |_| ()))
}


#[pyfunction]
/// u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of u is [channels, channels],
/// shape of y is [height, width, channels].
pub fn swta_u_repeated_conv_<'py>(u: &'py PyArray2<f32>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::top_repeated_conv_(slice_as_arr(y.shape()), unsafe { u.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }, |_| ()))
}

#[pyfunction]
/// v is row-major. Element `v[k,j]==1` means neuron k (row) can inhibit neuron j (column).
/// Shape of s is [height, width, channels], shape of v is [channels, channels],
/// shape of y is [height, width, channels].
pub fn swta_v_repeated_conv_<'py>(v: &'py PyArray2<bool>, s: &'py PyArray3<f32>, y: &'py PyArray3<u8>) -> PyResult<()> {
    Ok(vf::soft_wta::top_repeated_conv_(slice_as_arr(y.shape()), unsafe { v.as_slice()? }, unsafe { s.as_slice()? }, unsafe { y.as_slice_mut()? }, |_| ()))
}

#[pyfunction]
pub fn cyclic_group<'py>(py: Python<'py>, n: usize) -> PyResult<&'py PyArray2<usize>> {
    let (m, l) = vf::cayley::cyclic_group(n);
    let ll = m.len() / l;
    PyArray1::from_vec(py, m).reshape([ll, l])
}


#[pyfunction]
pub fn cyclic_monoid<'py>(py: Python<'py>, n: usize) -> PyResult<&'py PyArray2<usize>> {
    let (m, l) = vf::cayley::cyclic_monoid(n);
    let ll = m.len() / l;
    PyArray1::from_vec(py, m).reshape([ll, l])
}


#[pyfunction]
pub fn direct_product<'py>(a: &'py PyArray2<usize>, b: &'py PyArray2<usize>) -> PyResult<&'py PyArray2<usize>> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    assert_eq!(a_shape.len(), 2, "Shape should be [elements, generators]");
    assert_eq!(b_shape.len(), 2, "Shape should be [elements, generators]");
    let a_s = unsafe { a.as_slice()? };
    let b_s = unsafe { b.as_slice()? };
    let (m, g) = vf::cayley::direct_product(a_s, a_shape[1], b_s, b_shape[1]);
    PyArray1::from_vec(a.py(), m).reshape([a_shape[0] * b_shape[0], g])
}

#[pyfunction]
/// `w` are feedforward weights
/// returns recurrent weights `u` and new feedforward weights `w`
pub fn learn_uw<'py>(state_space: &'py PyArray2<usize>, w: &'py PyArray2<f32>) -> PyResult<(&'py PyArray3<f32>, &'py PyArray2<f32>)> {
    let state_space_shape = state_space.shape();
    let w_shape = w.shape();
    let a_len = state_space_shape[1];
    assert_eq!(state_space_shape.len(), 2, "State space shape should be [states, transitions]");
    assert_eq!(w_shape.len(), 2, "W shape should be [states, quotient_monoid_elements]");
    let sp = unsafe { state_space.as_slice()? };
    let w_slice = unsafe { w.as_slice()? };
    let quotient_monoid_elements = w_shape[1];
    let states = state_space_shape[0];
    let mut u = Vec::empty(a_len * quotient_monoid_elements * quotient_monoid_elements);
    vf::cayley::learn_u(sp, a_len, w_slice, quotient_monoid_elements, &mut u);
    for a in 0..a_len {
        for g in 0..quotient_monoid_elements {
            let offset = (a * quotient_monoid_elements + g) * quotient_monoid_elements;
            let row = &mut u[offset..offset + quotient_monoid_elements];
            row[g] = 0.;
            let inv_sum = 1. / row.sum();
            row.mul_scalar_(inv_sum);
        }
    }
    let mut new_w = vec![0.; states * quotient_monoid_elements];
    vf::cayley::learn_w(sp, a_len, w_slice, quotient_monoid_elements, &u, &mut new_w);
    let u = PyArray1::from_vec(w.py(), u);
    let new_w = PyArray1::from_vec(w.py(), new_w);
    let u = u.reshape([a_len, quotient_monoid_elements, quotient_monoid_elements])?;
    let new_w = new_w.reshape([states, quotient_monoid_elements])?;
    Ok((u, new_w))
}


///
/// HwtaL2Layer(shape: ConvShape, norm: int)
///
#[pyclass]
pub struct HwtaL2Layer {
    pub(crate) l: vf::ecc_layer::HwtaL2Layer<Idx>,
}

#[pymethods]
impl HwtaL2Layer {
    #[new]
    pub fn new(cs: &ConvShape, norm: usize) -> Self {
        Self { l: vf::ecc_layer::HwtaL2Layer::new(cs.cs.clone(), norm) }
    }
    /// returns the ConvShape object
    #[getter]
    fn shape(&self) -> ConvShape {
        ConvShape { cs: self.l.shape().clone() }
    }
    #[getter]
    fn n(&self) -> Idx {
        self.l.n()
    }
    #[getter]
    fn m(&self) -> Idx {
        self.l.m()
    }
    ///x is a sparse representation of binary vector of shape `[n]`, output is a sparse repr. of bin. vec. of shape `[m]`
    fn run<'py>(&'py self, x: &'py PyArray1<Idx>) -> PyResult<&'py PyArray1<Idx>> {
        let xi = unsafe { x.as_slice() }?;
        Ok(PyArray1::from_vec(x.py(), self.l.run_into_vec(xi)))
    }
    fn batch_run<'py>(&'py self, py: Python<'py>, indices: &'py PyArray1<Idx>, offsets: &'py PyArray1<usize>) -> PyResult<(&'py PyArray1<Idx>, &'py PyArray1<usize>)> {
        let i = unsafe { indices.as_slice() }?;
        let o = unsafe { offsets.as_slice() }?;
        let (indices, offsets) = vf::join_sets(&self.l.batch_run_into_vec(i, o));
        let i = PyArray1::<u32>::from_vec(py, indices);
        let o = PyArray1::<usize>::from_vec(py, offsets);
        Ok((i, o))
    }
    fn batch_run_conv<'py>(&'py self, py: Python<'py>, indices: &'py PyArray1<Idx>, offsets: &'py PyArray1<usize>) -> PyResult<(&'py PyArray1<Idx>, &'py PyArray1<usize>)> {
        let i = unsafe { indices.as_slice() }?;
        let o = unsafe { offsets.as_slice() }?;
        let (indices, offsets) = vf::join_sets(&self.l.batch_run_conv_into_vec(i, o));
        let i = PyArray1::<u32>::from_vec(py, indices);
        let o = PyArray1::<usize>::from_vec(py, offsets);
        Ok((i, o))
    }
    ///x is a sparse representation of binary vector of shape `[n]`, y is a sparse repr. of bin. vec. of shape `[m]`
    fn learn(&mut self, x: &PyArray1<Idx>, s: &PyArray1<f32>, y: &PyArray1<Idx>) -> PyResult<()> {
        let xi = unsafe { x.as_slice() }?;
        let yi = unsafe { y.as_slice() }?;
        let si = unsafe { s.as_slice() }?;
        Ok(self.l.learn(xi, si, yi))
    }
    ///x is a sparse representation of binary matrix of shape `[in_height, in_width, in_channels]`, output is a sparse repr. of bin. mat. of shape `[out_height, out_width, out_channels]`
    fn run_conv<'py>(&'py self, x: &'py PyArray1<Idx>) -> PyResult<&'py PyArray1<Idx>> {
        let xi = unsafe { x.as_slice() }?;
        Ok(PyArray1::from_vec(x.py(), self.l.run_conv_into_vec(xi)))
    }
    ///x is a sparse representation of binary matrix of shape `[kernel_height, kernel_width, in_channels]`, output is a sparse repr. of bin. mat. of shape `[out_channels]`
    fn train<'py>(&'py mut self, x: &'py PyArray1<Idx>) -> PyResult<&'py PyArray1<Idx>> {
        let xi = unsafe { x.as_slice() }?;
        Ok(PyArray1::from_vec(x.py(), self.l.train(xi)))
    }
    ///x is a sparse representation of binary matrix of shape `[in_height, in_width, in_channels]`, y is a sparse repr. of bin. mat. of shape `[out_height, out_width, out_channels]`
    fn learn_conv(&mut self, x: &PyArray1<Idx>, s: &PyArray1<f32>, y: &PyArray1<Idx>) -> PyResult<()> {
        let xi = unsafe { x.as_slice() }?;
        let yi = unsafe { y.as_slice() }?;
        let si = unsafe { s.as_slice() }?;
        Ok(self.l.learn_conv(xi, si, yi))
    }
    #[getter]
    fn W<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray4<f32>> {
        PyArray1::from_slice(py, self.l.W()).reshape(self.l.shape().minicolumn_w_shape().map(|i| i as usize))
    }
    #[getter]
    fn r<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        PyArray1::from_slice(py, self.l.r())
    }
    fn save(&self, path: String) -> PyResult<()> {
        pickle(&self.l, path)
    }
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        unpickle(path).map(|l| Self { l })
    }
}

///
/// SwtaLayer(shape: ConvShape, norm: int)
///
#[pyclass]
pub struct SwtaLayer {
    pub(crate) l: vf::ecc_layer::SwtaLayer<Idx>,
}

#[pymethods]
impl SwtaLayer {
    #[new]
    pub fn new(cs: &ConvShape, norm: usize) -> Self {
        Self { l: vf::ecc_layer::SwtaLayer::new(cs.cs.clone(), norm) }
    }
    /// returns the ConvShape object
    #[getter]
    fn shape(&self) -> ConvShape {
        ConvShape { cs: self.l.shape().clone() }
    }
    #[getter]
    fn get_use_abs(&self) -> bool {
        self.l.use_abs
    }
    #[setter]
    fn set_use_abs(&mut self, use_abs: bool) {
        self.l.use_abs = use_abs
    }
    #[getter]
    fn get_use_cos_sim(&self) -> bool {
        self.l.cos_sim
    }
    #[setter]
    fn set_use_cos_sim(&mut self, cos_sim: bool) {
        self.l.cos_sim = cos_sim
    }
    #[getter]
    fn get_conditional(&self) -> bool {
        self.l.conditional
    }
    #[setter]
    fn set_conditional(&mut self, conditional: bool) {
        self.l.conditional = conditional
    }
    #[getter]
    fn W<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray4<f32>> {
        PyArray1::from_slice(py, self.l.W()).reshape(self.l.shape().minicolumn_w_shape().map(|i| i as usize))
    }
    #[getter]
    fn U<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f32>> {
        PyArray1::from_slice(py, self.l.U()).reshape(self.l.shape().minicolumn_u_shape().map(|i| i as usize))
    }
    #[getter]
    fn n(&self) -> Idx {
        self.l.n()
    }
    #[getter]
    fn m(&self) -> Idx {
        self.l.m()
    }

    ///x is a sparse representation of binary vector of shape `[n]`, output is a sparse repr. of bin. vec. of shape `[m]`
    fn run<'py>(&'py self, x: &'py PyArray1<Idx>) -> PyResult<&'py PyArray1<Idx>> {
        let xi = unsafe { x.as_slice() }?;
        Ok(PyArray1::from_vec(x.py(), self.l.run_into_vec(xi)))
    }
    ///x is a sparse representation of binary vector of shape `[n]`, y is a sparse repr. of bin. vec. of shape `[m]`
    fn learn(&mut self, x: &PyArray1<Idx>, s: &PyArray1<f32>, y: &PyArray1<Idx>) -> PyResult<()> {
        let xi = unsafe { x.as_slice() }?;
        let yi = unsafe { y.as_slice() }?;
        let si = unsafe { s.as_slice() }?;
        Ok(self.l.learn(xi, si, yi))
    }
    ///x is a sparse representation of binary matrix of shape `[in_height, in_width, in_channels]`, output is a sparse repr. of bin. mat. of shape `[out_height, out_width, out_channels]`
    fn run_conv<'py>(&'py self, x: &'py PyArray1<Idx>) -> PyResult<&'py PyArray1<Idx>> {
        let xi = unsafe { x.as_slice() }?;
        Ok(PyArray1::from_vec(x.py(), self.l.run_conv_into_vec(xi)))
    }
    ///x is a sparse representation of binary matrix of shape `[kernel_height, kernel_width, in_channels]`, output is a sparse repr. of bin. mat. of shape `[out_channels]`
    fn train<'py>(&'py mut self, x: &'py PyArray1<Idx>) -> PyResult<&'py PyArray1<Idx>> {
        let xi = unsafe { x.as_slice() }?;
        Ok(PyArray1::from_vec(x.py(), self.l.train(xi)))
    }
    ///x is a sparse representation of binary matrix of shape `[in_height, in_width, in_channels]`, y is a sparse repr. of bin. mat. of shape `[out_height, out_width, out_channels]`
    fn learn_conv(&mut self, x: &PyArray1<Idx>, s: &PyArray1<f32>, y: &PyArray1<Idx>) -> PyResult<()> {
        let xi = unsafe { x.as_slice() }?;
        let yi = unsafe { y.as_slice() }?;
        let si = unsafe { s.as_slice() }?;
        Ok(self.l.learn_conv(xi, si, yi))
    }
    fn save(&self, path: String) -> PyResult<()> {
        pickle(&self.l, path)
    }
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        unpickle(path).map(|l| Self { l })
    }
}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn ecc_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<ConvShape>()?;
    m.add_class::<SwtaLayer>()?;
    m.add_class::<HwtaL2Layer>()?;
    m.add_function(wrap_pyfunction!(sample, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_range_with_custom_size, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_range, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_range, m)?)?;
    m.add_function(wrap_pyfunction!(conv_out_range_clipped_both_sides, m)?)?;
    m.add_function(wrap_pyfunction!(conv_in_range_begin, m)?)?;
    m.add_function(wrap_pyfunction!(conv_stride, m)?)?;
    m.add_function(wrap_pyfunction!(conv_compose_array, m)?)?;
    m.add_function(wrap_pyfunction!(conv_compose, m)?)?;
    m.add_function(wrap_pyfunction!(conv_compose_weights, m)?)?;
    m.add_function(wrap_pyfunction!(swta_u, m)?)?;
    m.add_function(wrap_pyfunction!(swta_v, m)?)?;
    m.add_function(wrap_pyfunction!(swta_u_, m)?)?;
    m.add_function(wrap_pyfunction!(swta_v_, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_subtensor_reindexed, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_subtensor, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_swta_u, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_swta_v, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_swta_u_, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_swta_v_, m)?)?;
    m.add_function(wrap_pyfunction!(swta_u_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(swta_v_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(swta_u_repeated_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(swta_v_repeated_conv_, m)?)?;
    m.add_function(wrap_pyfunction!(sparse, m)?)?;
    m.add_function(wrap_pyfunction!(dense, m)?)?;
    m.add_function(wrap_pyfunction!(dense_, m)?)?;
    m.add_function(wrap_pyfunction!(batch_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(join_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(rand_sparse_k, m)?)?;
    m.add_function(wrap_pyfunction!(rand_sparse_k_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(rand_dense_k, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_gt, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_top_k, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_dot, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_dot_t, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_group, m)?)?;
    m.add_function(wrap_pyfunction!(cyclic_monoid, m)?)?;
    m.add_function(wrap_pyfunction!(direct_product, m)?)?;
    m.add_function(wrap_pyfunction!(learn_uw, m)?)?;
    m.add_function(wrap_pyfunction!(sample_of_cardinality, m)?)?;
    m.add_function(wrap_pyfunction!(receptive_field, m)?)?;
    // m.add_function(wrap_pyfunction!(conv_variance, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat0() -> Result<(), String> {
        Ok(())
    }
}