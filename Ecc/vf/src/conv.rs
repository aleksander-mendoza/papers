use std::fmt::Debug;
use crate::*;
use std::ops::{Range, Mul, Add, Sub, Div, Rem};
use num_traits::{AsPrimitive, One, Zero};

pub fn in_range_begin<T: Mul<Output=T> + Copy, const DIM: usize>(out_position: &[T; DIM], stride: &[T; DIM]) -> [T; DIM] {
    out_position.mul(stride)
}

/**returns the range of inputs that connect to a specific output neuron*/
pub fn in_range<T: Copy + Mul<Output=T> + Add<Output=T> + Zero, const DIM: usize>(out_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    let from = in_range_begin(out_position, stride);
    let to = from.add(kernel_size);
    from..to
}

/**returns the range of inputs that connect to a specific patch of output neuron.
That output patch starts this position specified by this vector*/
pub fn in_range_with_custom_size<T: Copy + Mul<Output=T> + Add<Output=T> + Zero + One + Sub<Output=T> + PartialOrd, const DIM: usize>(out_position: &[T; DIM], output_patch_size: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    if output_patch_size.all_gt_scalar(T::zero()) {
        let from = in_range_begin(out_position, stride);
        let to = in_range_begin(&out_position.add(output_patch_size)._sub_scalar(T::one()), stride);
        let to = to._add(kernel_size);
        from..to
    } else {
        [T::zero(); DIM]..[T::zero(); DIM]
    }
}

/**returns the range of outputs that connect to a specific input neuron*/
pub fn out_range<T: Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + One, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    //out_position * stride .. out_position * stride + kernel
    //out_position * stride ..= out_position * stride + kernel - 1
    //
    //in_position_from == out_position * stride
    //in_position_from / stride == out_position
    //round_down(in_position / stride) == out_position_to
    //
    //in_position_to == out_position * stride + kernel - 1
    //(in_position_to +1 - kernel)/stride == out_position
    //round_up((in_position +1 - kernel)/stride) == out_position_from
    //round_down((in_position +1 - kernel + stride - 1)/stride) == out_position_from
    //round_down((in_position - kernel + stride)/stride) == out_position_from
    //
    //(in_position - kernel + stride)/stride ..= in_position / stride
    //(in_position - kernel + stride)/stride .. in_position / stride + 1
    let to = in_position.div(stride)._add_scalar(T::one());
    let from = in_position.add(stride)._sub(kernel_size)._div(stride);
    from..to
}

pub fn out_transpose_kernel<T: Copy + Div<Output=T> + Add<Output=T> + One + Sub<Output=T> + Ord, const DIM: usize>(kernel: &[T; DIM], stride: &[T; DIM]) -> [T; DIM] {
    // (in_position - kernel + stride)/stride .. in_position / stride + 1
    //  in_position / stride + 1 - (in_position - kernel + stride)/stride
    //  (in_position- (in_position - kernel + stride))/stride + 1
    //  (kernel - stride)/stride + 1
    debug_assert!(kernel.all_ge(stride));
    kernel.sub(stride)._div(stride)._add_scalar(T::one())
}

/**returns the range of outputs that connect to a specific input neuron.
output range is clipped to 0, so that you don't get overflow on negative values when dealing with unsigned integers.*/
pub fn out_range_clipped<T: Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + One + Ord, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> Range<[T; DIM]> {
    let to = in_position.div(stride)._add_scalar(T::one());
    let from = in_position.add(stride)._max(kernel_size)._sub(kernel_size)._div(stride);
    from..to
}

pub fn out_range_clipped_both_sides<T: Copy + Div<Output=T> + Add<Output=T> + One + Sub<Output=T> + Ord, const DIM: usize>(in_position: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM], max_bounds: &[T; DIM]) -> Range<[T; DIM]> {
    let mut r = out_range_clipped(in_position, stride, kernel_size);
    r.end.min_(max_bounds);
    r
}

pub fn out_size<T: Debug + Rem<Output=T> + Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Ord + Zero + One, const DIM: usize>(input: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(kernel_size.all_le(input), "Kernel size {:?} is larger than the input shape {:?} ", kernel_size, input);
    let input_sub_kernel = input.sub(kernel_size);
    assert!(input_sub_kernel.rem(stride).all_eq_scalar(T::zero()), "Convolution stride {:?} does not evenly divide the input shape {:?}-{:?}={:?} ", stride, input, kernel_size, input_sub_kernel);
    input_sub_kernel._div(stride)._add_scalar(T::one())
    //(input-kernel)/stride+1 == output
}

pub fn in_size<T: Debug + Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Zero + Ord + One, const DIM: usize>(output: &[T; DIM], stride: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(output.all_gt_scalar(T::zero()), "Output size {:?} contains zero", output);
    output.sub_scalar(T::one())._mul(stride)._add(kernel_size)
    //input == stride*(output-1)+kernel
}

pub fn stride<T: Debug + Rem<Output=T> + Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Ord + One + Zero, const DIM: usize>(input: &[T; DIM], out_size: &[T; DIM], kernel_size: &[T; DIM]) -> [T; DIM] {
    assert!(kernel_size.all_le(input), "Kernel size {:?} is larger than the input shape {:?}", kernel_size, input);
    let input_sub_kernel = input.sub(kernel_size);
    let out_size_minus_1 = out_size.sub_scalar(T::one());
    assert!(input_sub_kernel.rem_default_zero(&out_size_minus_1, T::zero()).all_eq_scalar(T::zero()), "Output shape {:?}-1 does not evenly divide the input shape {:?}", out_size, input);
    input_sub_kernel._div_default_zero(&out_size_minus_1, T::one())
    //(input-kernel)/(output-1) == stride
}

pub fn compose<T: Copy + Div<Output=T> + Add<Output=T> + Sub<Output=T> + Ord + One, const DIM: usize>(self_stride: &[T; DIM], self_kernel: &[T; DIM], next_stride: &[T; DIM], next_kernel: &[T; DIM]) -> ([T; DIM], [T; DIM]) {
    //(A-kernelA)/strideA+1 == B
    //(B-kernelB)/strideB+1 == C
    //((A-kernelA)/strideA+1-kernelB)/strideB+1 == C
    //(A-kernelA+(1-kernelB)*strideA)/(strideA*strideB)+1 == C
    //(A-(kernelA-(1-kernelB)*strideA))/(strideA*strideB)+1 == C
    //(A-(kernelA+(kernelB-1)*strideA))/(strideA*strideB)+1 == C
    //    ^^^^^^^^^^^^^^^^^^^^^^^^^^^                    composed kernel
    //                                   ^^^^^^^^^^^^^^^ composed stride
    let composed_kernel = next_kernel.sub_scalar(T::one())._mul(self_stride)._add(self_kernel);
    let composed_stride = self_stride.mul(next_stride);
    (composed_stride, composed_kernel)
}

/**First run `compose` to obtain `comp_stride` and `comp_kernel`. The shapes of tensors are
 ```
self_weights.shape==[self_out_channels, self_in_channels, self_kernel[0], self_kernel[1]]
next_weights.shape==[next_out_channels, next_in_channels, next_kernel[0], next_kernel[1]]
where
self_in_channels == comp_in_channels
self_bias.len() == self_out_channels == next_in_channels
next_bias.len() == next_out_channels == comp_out_channels
comp_stride, comp_kernel = compose(self_stride, self_kernel, next_stride, next_kernel)
```
and there is precondition
```
forall i: comp_weights[i]==0.
```
This specification is compatible wit PyTorch
 */
pub fn compose_weights2d<T: Copy + Mul<Output=T> + Add<Output=T> + One + AsPrimitive<usize>>(
    self_in_channels: T,
    self_stride: &[T; 2], self_kernel: &[T; 2], self_weights: &[f32], self_bias: &[f32],
    next_kernel: &[T; 2], next_weights: &[f32], next_bias: &[f32],
    comp_kernel: &[T; 2], comp_weights: &mut [f32], comp_bias: &mut [f32]) {
    let self_in_channels = self_in_channels.as_();
    let self_out_channels = self_bias.len();
    let next_in_channels = self_out_channels;
    let next_out_channels = next_bias.len();
    let self_ker_area = self_kernel.product().as_();
    let next_ker_area = next_kernel.product().as_();
    let comp_ker_area = comp_kernel.product().as_();
    assert_eq!(self_in_channels * self_out_channels * self_ker_area, self_weights.len(), "self_in_channels={}, self_out_channels={},self_ker_area={},  LHS convolution weight tensor has wrong size", self_in_channels, self_out_channels, self_ker_area);
    assert_eq!(next_in_channels * next_out_channels * next_ker_area, next_weights.len(), "next_in_channels={}, next_out_channels={}, next_ker_area={},  RHS convolution weight tensor has wrong size", next_in_channels, next_out_channels, next_ker_area);
    assert_eq!(self_in_channels * next_out_channels * comp_ker_area, comp_weights.len(), "self_in_channels={}, next_out_channels={}, comp_ker_area={}, composed convolution weight tensor has wrong size", self_in_channels, next_out_channels, comp_ker_area);
    assert_eq!(next_bias.len(), comp_bias.len(), "composed bias length doesn't match RHS bias length");
    // This is what happens in 1D case
    //
    //                 [w1, w2, w3]  -> y5 = b + w1*x5 + w2*x6 + w3*x7
    //             [w1, w2, w3]      -> y4 = b + w1*x4 + w2*x5 + w3*x6
    //         [w1, w2, w3]          -> y3 = b + w1*x3 + w2*x4 + w3*x5
    //     [w1, w2, w3]              -> y2 = b + w1*x2 + w2*x3 + w3*x4
    // [w1, w2, w3]                  -> y1 = b + w1*x1 + w2*x2 + w3*x3
    // [x1, x2, x3, x4, x5, x6, x7]  -> input x convolved with self (w)
    //
    //         [W1, W2]  -> z3 = B + W1*y3 + W2*y4
    //     [W1, W2]      -> z2 = B + W1*y2 + W2*y3
    // [W1, W2]          -> z1 = B + W1*y1 + W2*y2
    // [y1, y2, y3, y4]  -> input y convolved with next (W)
    //
    // Therefore composition does
    //  z1 = B + W1*y1 + W2*y2
    //     = B + W1*(b + w1*x1 + w2*x2 + w3*x3) + W2 * (b + w1*x2 + w2*x3 + w3*x4)
    //     = B + W1*b + W2*b + W1*w1*x1 + W1*w2*x2 + W1*w3*x3 + W2*w1*x2 + W2*w2*x3 + W2*w3*x4
    //     = (B + W1*b + W2*b) + W1*w1*x1 + (W1*w2 + W2*w1)*x2 + (W1*w3 + W2*w2)*x3 + W2*w3*x4
    //     = (B + W1*b + W2*b) + ([W2, W1] convolved with [0, w1, w2, w3, 0]) @ [x1, x2, x3, x4]^T
    // If there was stride 2 in first convolution then y vector would consist of [y1,y3,y5]
    //  z1 = B + W1*y1 + W2*y3
    //     = B + W1*(b + w1*x1 + w2*x2 + w3*x3) + W2*(b + w1*x3 + w2*x4 + w3*x5)
    //     = B + W1*b + W2*b + W1*w1*x1 + W1*w2*x2 + W1*w3*x3 + W2*w1*x3 + W2*w2*x4 + W2*w3*x5
    //     = B + W1*b + W2*b + W1*w1*x1 + W1*w2*x2 + (W1*w3 + W2*w1)*x3 + W2*w2*x4 + W2*w3*x5
    //     = (B + W1*b + W2*b) + ([W2, 0, W1] convolved with [0, 0, w1, w2, w3, 0,  0]) @ [x1, x2, x3, x4, x5]^T
    // The stride of second convolution has no impact on computation of individual z's. It only changes
    // which of the z's are evaluated. But composition of two convolutional layers is achieved by finding
    // the formula for a single z given the vector of x.

    /*next_bias is like uppercase B */
    comp_bias.copy_from_slice(next_bias);
    for next_out_channel in 0..next_out_channels {
        // We need to compute z1 (it could be any z but indexing makes sense when you take z1.
        // The other z's are just translations whose indexing is shifted accordingly).
        // We will need to sum over all channels in the hidden layer.
        for next_in_channel in 0..next_in_channels {
            let self_out_channel = next_in_channel; // just for better readability
            let w_next_offset = (next_out_channel * next_in_channels + next_in_channel) * next_ker_area;
            /*(next_x,next_y) is like the index of uppercase W*/
            for next_x in 0..next_kernel[0].as_() {
                for next_y in 0..next_kernel[1].as_() {
                    /*w_next is like uppercase W */
                    let w_next = next_weights[w_next_offset + next_x * next_kernel[1].as_() + next_y];
                    /*b_self is like lowercase b */
                    let b_self = self_bias[self_out_channel];
                    comp_bias[next_out_channel] += b_self * w_next;
                    /*(hidden_x,hidden_y) is like the index of y*/
                    let hidden_x = next_x * self_stride[0].as_();
                    let hidden_y = next_y * self_stride[1].as_();
                    /*(self_x,self_y) is like the index of lowercase w*/
                    for self_in_channel in 0..self_in_channels {
                        let w_comp_offset = (next_out_channel * self_in_channels + self_in_channel) * comp_ker_area;
                        let w_self_offset = (self_out_channel * self_in_channels + self_in_channel) * self_ker_area;
                        for self_x in 0..self_kernel[0].as_() {
                            for self_y in 0..self_kernel[1].as_() {
                                /* w_self is like lowercase w */
                                let w_self = self_weights[w_self_offset + self_x * self_kernel[1].as_() + self_y];
                                let w_comp = w_self * w_next;
                                /* (comp_x,comp_y) is like the index of x*/
                                let comp_x = hidden_x + self_x;
                                let comp_y = hidden_y + self_y;
                                comp_weights[w_comp_offset + comp_x * comp_kernel[1].as_() + comp_y] += w_comp;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn conv_nearest(img:&[u8], out:&mut [u8], kernel_height:usize, kernel_width:usize, height:usize, width:usize){
    assert_eq!(img.len(), height *width, "img length should be height*width={}*{}",height ,width);
    assert_eq!(out.len(), height*width, "output length should be height*width={}*{}",height,width);
    let bottom_half_kh = kernel_height as isize/2;
    let top_half_kh = kernel_height as isize-bottom_half_kh;
    let bottom_half_kw = kernel_width as isize/2;
    let top_half_kw = kernel_width as isize-bottom_half_kw;
    fn l2(x:isize,y:isize)->u8{
        let d = x*x+y*y;
        let d = d as f32;
        let d = d.sqrt().min(255.);
        d as u8
    }
    let mut order:Vec<(isize,isize,u8)> = (-bottom_half_kh..top_half_kh).flat_map(|y|(-bottom_half_kw..top_half_kw).map(move |x|(x,y,l2(x,y)))).collect();
    order.sort_by(|(_,_,d1),(_,_,d2)|d1.cmp(d2));
    for y in 0..height{
        'o: for x in 0..width{
            let offset = y*width+x;
            for &(kx,ky,d) in &order{
                let x = x as isize + kx;
                let y = y as isize + ky;
                if x >=0 && y >= 0 && (x as usize) < width  && (y as usize) < height && img[y as usize*width+x as usize]!=0{
                    out[offset] = d;
                    continue 'o;
                }
            }
            out[offset] = 255;
        }
    }
}

// pub fn conv_variance(img:&[f32], kernel:&[f32], out:&mut [f32], kernel_height:usize, kernel_width:usize, height:usize, width:usize, channels:usize){
//     assert_eq!(img.len(), height *width*channels, "img length should be height*width*channels={}*{}*{}",height ,width,channels);
//     assert_eq!(kernel.len(), kernel_width*kernel_height, "kernel length should be kernel_width*kernel_height={}*{}",kernel_width,kernel_height);
//     assert_eq!(out.len(), height*width, "output length should be height*width={}*{}",height,width);
//     let bottom_half_kh = kernel_height/2;
//     let top_half_kh = kernel_height-bottom_half_kh;
//     let bottom_half_kw = kernel_width/2;
//     let top_half_kw = kernel_width-bottom_half_kw;
//     for y in 0..height{
//         for x in 0..width{
//             let offset = y*width+x;
//             let c = &img[offset*channels..(offset+1)*channels];
//             let mut mean = 0.;
//             let mut mean_squared = 0.;
//             let from_kx = x as isize-bottom_half_kw as isize;
//             let from_ky = y as isize-bottom_half_kh as isize;
//             let to_kx = x+top_half_kw;
//             let to_ky = y+top_half_kh;
//             let range_y = from_ky.max(0) as usize..to_ky.min(height);
//             let range_x = from_kx.max(0) as usize..to_kx.min(width);
//             // let area = range_y.len()*range_x.len();
//             for ky in range_y{
//                 for kx in range_x{
//                     // println!("{from_ky}.min(0)={ky}..{to_ky} w={width} kx={kx} {channels}");
//                     let offset = (ky*width+kx) *channels;
//                     let kc = &img[offset..offset+channels];
//                     fn square(a:f32)->f32{a*a}
//                     // println!("{:?}",kc);
//                     // println!("{:?}",c);
//                     let distance_squared:f32 = c.iter().zip(kc.iter()).map(|(&a,&b)|square(a-b)).sum();
//                     let ker_x = (kx as isize - from_kx) as usize;
//                     let ker_y = (ky as isize - from_ky) as usize;
//                     let weight = kernel[ker_y*kernel_width+ker_x];
//                     mean += distance_squared;
//                     mean_squared += distance_squared*distance_squared;
//                 }
//             }
//             // mean_squared /= area as f32;
//             // mean /= area as f32;
//             out[offset] = mean_squared - mean*mean;
//         }
//     }
// }
// pub fn conv(&[])

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test17(){
        let kernel_height = 3;
        let kernel_width = 3;
        let height = 6;
        let width = 6;
        let channels = 3;
        let img:Vec<f32> = (0..height*width*channels).map(|_|rand::random::<f32>()).collect();
        let kernel:Vec<f32> = (0..kernel_height*kernel_width).map(|_|1./(kernel_height*kernel_width) as f32).collect();
        let mut out:Vec<f32> = vec![0.;height*width];
        conv_variance(&img, &kernel, &mut out,kernel_height ,
                      kernel_width ,
                      height ,
                      width,
                      channels)
    }
    #[test]
    fn test5() {
        for x in 1..3 {
            for y in 1..4 {
                for sx in 1..2 {
                    for sy in 1..2 {
                        for ix in 1..3 {
                            for iy in 1..4 {
                                let kernel = [x, y];
                                let stride = [x, y];
                                let output_size = [ix, iy];
                                let input_size = in_size(&output_size, &stride, &kernel);
                                assert_eq!(output_size, out_size(&input_size, &stride, &kernel));
                                for ((&expected, &actual), &out) in stride.iter().zip(super::stride(&input_size, &output_size, &kernel).iter()).zip(output_size.iter()) {
                                    if out != 1 {
                                        assert_eq!(expected, actual);
                                    } else {
                                        assert_eq!(1, actual);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test6() {
        for output_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let i = in_range(&[output_idx], &[sx], &[x]);
                    let i_r = i.start[0]..i.end[0];
                    for i in i_r.clone() {
                        let o = out_range(&[i], &[sx], &[x]);
                        let o_r = o.start[0]..o.end[0];
                        assert!(o_r.contains(&output_idx), "o_r={:?}, i_r={:?} output_idx={} sx={} x={}", o_r, i_r, output_idx, sx, x)
                    }
                }
            }
        }
    }

    #[test]
    fn test7() {
        for input_idx in 0..24 {
            for x in 1..5 {
                for sx in 1..5 {
                    let o = out_range(&[input_idx], &[sx], &[x]);
                    let o_r = o.start[0]..o.end[0];
                    for o in o_r.clone() {
                        let i = in_range(&[o], &[sx], &[x]);
                        let i_r = i.start[0]..i.end[0];
                        assert!(i_r.contains(&input_idx), "o_r={:?}, i_r={:?} input_idx={} sx={} x={}", o_r, i_r, input_idx, sx, x)
                    }
                }
            }
        }
    }
}