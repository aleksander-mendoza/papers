use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::iter::Step;
use std::path::Path;
use itertools::Itertools;
use more_asserts::debug_assert_lt;
use nalgebra::{DMatrix, DVector, Matrix};
use num_traits::{AsPrimitive, NumAssign, PrimInt};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use crate::{ArrayCast, l, normalize_mat_column, normalize_mat_columns, VectorFieldAddAssign, VectorFieldOne};
use crate::conv_shape::ConvShape;
use crate::dot_sparse_slice::{dot1, dot1_};
use crate::from_usize::FromUsize;
use crate::init::{InitEmptyWithCapacity, InitFilledCapacity};
use crate::init_rand::InitRandWithCapacity;
use crate::soft_wta::NULL;
use serde::{Serialize, Deserialize};


pub trait Layer<Idx: Debug + PrimInt + FromUsize + AsPrimitive<usize> + NumAssign + Send+Sync>: Send+Sync{
    fn shape(&self) -> &ConvShape<Idx>;
    /**Size of input when running in non-convolutional mode (`run` and `learn`)*/
    fn n(&self) -> Idx {
        self.shape().kernel_column_volume()
    }
    /**Size of output when running in non-convolutional mode (`run` and `learn`)*/
    fn m(&self) -> Idx {
        self.shape().out_channels()
    }
    fn new_s(&self) -> Vec<f32> {
        Vec::empty(self.m().as_())
    }
    fn new_s_conv(&self) -> Vec<f32> {
        Vec::empty(self.shape().out_volume().as_())
    }
    /**x is sparse representation of input tensor of shape `[kernel_height, kernel_width, in_channels]`,
             output is sparse representation of tensor of shape `[out_channels]`*/
    fn run_(&self, x: &[Idx], s: &mut [f32], winner_callback: impl FnMut(usize));
    fn run(&self, x: &[Idx], winner_callback: impl FnMut(usize)) {
        self.run_(x, self.new_s().as_mut_slice(), winner_callback)
    }

    /**x is sparse representation of input tensor of shape `[kernel_height, kernel_width, in_channels]`,
                 output is sparse representation of tensor of shape `[out_channels]`*/
    fn run_into_vec_(&self, x: &[Idx], s: &mut [f32]) -> Vec<Idx> {
        let mut v = Vec::new();
        self.run_(x, s, |k| v.push(Idx::from_usize(k)));
        v
    }
    fn run_into_vec(&self, x: &[Idx]) -> Vec<Idx> {
        self.run_into_vec_(x, self.new_s().as_mut_slice())
    }
    fn batch_run_into_vec(&self, indices: &[Idx], offsets:&[usize]) -> Vec<Vec<Idx>>{
        (0..offsets.len()-1).into_par_iter().map(|i|{
            let x:&[Idx] = &indices[offsets[i]..offsets[i+1]];
            self.run_into_vec(x)
        }).collect()
    }
    /**x is sparse representation of input tensor of shape `[kernel_height, kernel_width, in_channels]`,
         y is sparse representation of output tensor of shape `[out_channels]`*/
    fn learn(&mut self, x: &[Idx], s: &[f32], y: &[Idx]);
    fn train_(&mut self, x: &[Idx], s: &mut [f32]) -> Vec<Idx> {
        let y = self.run_into_vec_(x, s);
        self.learn(x, s, &y);
        y
    }
    fn train(&mut self, x: &[Idx]) -> Vec<Idx> {
        self.train_(x, self.new_s().as_mut_slice())
    }
    /**x is sparse representation of input tensor of shape `[in_height, in_width, in_channels]`,
             output is sparse representation of tensor of shape `[out_height, out_width, out_channels]`*/
    fn run_conv_(&self, x: &[Idx], s: &mut [f32], winner_callback: impl FnMut(usize));
    fn run_conv(&self, x: &[Idx], winner_callback: impl FnMut(usize)) {
        self.run_conv_(x, self.new_s_conv().as_mut_slice(), winner_callback)
    }
    /**x is sparse representation of input tensor of shape `[in_height, in_width, in_channels]`,
                 output is sparse representation of tensor of shape `[out_height, out_width, out_channels]`*/
    fn run_conv_into_vec_(&self, x: &[Idx], s: &mut [f32]) -> Vec<Idx> {
        let mut v = Vec::new();
        self.run_conv_(x, s, |k| v.push(Idx::from_usize(k)));
        v
    }
    fn run_conv_into_vec(&self, x: &[Idx]) -> Vec<Idx> {
        self.run_conv_into_vec_(x, self.new_s_conv().as_mut_slice())
    }
    fn batch_run_conv_into_vec(&self, indices: &[Idx], offsets:&[usize]) -> Vec<Vec<Idx>> {
        (0..offsets.len()-1).into_par_iter().map(|i|{
            let x:&[Idx] = &indices[offsets[i]..offsets[i+1]];
            self.run_conv_into_vec(x)
        }).collect()
    }
    /**x is sparse representation of input tensor of shape `[in_height, in_width, in_channels]`,
                 y is sparse representation of output tensor of shape `[out_height, out_width, out_channels]`*/
    fn learn_conv(&mut self, x: &[Idx], s: &[f32], y: &[Idx]);
    fn train_conv_(&mut self, x: &[Idx], s: &mut [f32]) -> Vec<Idx> {
        let y = self.run_conv_into_vec_(x, s);
        self.learn_conv(x, s, &y);
        y
    }
    fn train_conv(&mut self, x: &[Idx]) -> Vec<Idx> {
        self.train_conv_(x, self.new_s_conv().as_mut_slice())
    }
}
/**Returns tensor of shape `[output_size,input_size]`*/
pub fn receptive_field<Idx:AsPrimitive<usize>>(output_size:usize, output_indices: &[Idx], output_offsets:&[usize], input_size:usize, input_indices: &[Idx], input_offsets:&[usize])->Vec<f32>{
    assert_eq!(output_offsets.len(), input_offsets.len());
    let mut rf = vec![0f32;output_size*input_size];
    for i in 0..output_offsets.len()-1{
        let out = &output_indices[output_offsets[i]..output_offsets[i+1]];
        let inp = &input_indices[input_offsets[i]..input_offsets[i+1]];
        for out_channel in out{
            let out_channel= out_channel.as_();
            let rf_offset = out_channel*input_size;
            for inp_idx in inp{
                let inp_idx = inp_idx.as_();
                debug_assert_lt!(inp_idx, input_size);
                rf[rf_offset+inp_idx] += 1.;
            }
        }
    }
    rf
}
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct HwtaL2Layer<Idx: Debug + PrimInt + NumAssign + Send+Sync> {
    shape: ConvShape<Idx>,
    pub r_step: f32,
    /// of shape  `[kernel_height, kernel_width, in_channels, out_channels]`
    W: Vec<f32>,
    /// of shape `[out_channels]`
    r: Vec<f32>,
    norm: usize,
    pub w_step: f32,
    pub min_input_cardinality: u32,
}

impl<Idx: Debug + PrimInt + NumAssign+ Send+Sync> Debug for HwtaL2Layer<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hwta({:?})", self.shape)
    }
}

impl<Idx: Debug + PrimInt + AsPrimitive<usize> + NumAssign+ Send+Sync> HwtaL2Layer<Idx> {
    pub fn new(shape: ConvShape<Idx>, norm: usize) -> Self {
        let mut W = Vec::rand(shape.minicolumn_w_shape().product().as_());
        let r = Vec::full(shape.out_channels().as_(), 0f32);
        let width = shape.kernel_column_volume().as_();
        normalize_mat_columns(width, &mut W, l(norm));
        Self {
            shape,
            r_step: 0.002,
            W,
            r,
            norm,
            w_step: 0.01,
            min_input_cardinality: 1,
        }
    }
    pub fn W(&self)->&[f32]{
        &self.W
    }
    pub fn r(&self)->&[f32]{
        &self.r
    }
}

impl<Idx: Debug + PrimInt + AsPrimitive<usize> + Step + FromUsize + NumAssign + Send+Sync> Layer<Idx> for HwtaL2Layer<Idx> {
    fn shape(&self) -> &ConvShape<Idx> {
        &self.shape
    }

    fn run_(&self, x: &[Idx], s: &mut [f32], mut winner_callback: impl FnMut(usize)) {
        if x.len() >= self.min_input_cardinality as usize {
            debug_assert_lt!(x.iter().max().cloned().unwrap_or(Idx::zero()), self.n());
            debug_assert_eq!(self.W.len(), self.n().as_() * self.m().as_());
            // println!("n={} m={} max={}",self.n().as_(),self.m().as_(),x.iter().max().cloned().unwrap_or(Idx::zero()).as_());
            assert_eq!(s.len(), self.r.len());
            s.copy_from_slice(&self.r);
            dot1_(x, &self.W, s);
            let k = s.iter().cloned().position_max_by(f32::total_cmp).unwrap();
            winner_callback(k);
        }
    }

    fn learn(&mut self, x: &[Idx], s: &[f32], y: &[Idx]) {
        if let Some(k) = y.first() {
            assert_eq!(y.len(), 1);
            self.r[k.as_()] -= self.r_step;
            let c = self.shape.out_channels().as_();
            let w_step = self.w_step;
            for i in x {
                self.W[i.as_() * c + k.as_()] += w_step / x.len() as f32;
            }
            let width = self.shape.kernel_column_volume().as_();
            let n = self.norm;
            normalize_mat_column(width, k.as_(), &mut self.W, l(n));
        }
    }

    fn run_conv_(&self, x: &[Idx], s: &mut [f32], mut winner_callback: impl FnMut(usize)) {
        if x.len() >= self.min_input_cardinality as usize {
            ConvShape::<Idx>::copy_vector_repeated(&self.r, s);
            self.shape.sparse_dot_repeated_slice_(x, self.W.as_slice(), s);
            let mut offset = 0;
            let c = self.shape.out_channels().as_();
            while offset < s.len() {
                let end = offset + c;
                let k = s[offset..end].iter().cloned().position_max_by(f32::total_cmp).unwrap();
                winner_callback(offset + k);
                offset = end;
            }
        }
    }

    fn learn_conv(&mut self, x: &[Idx], s: &[f32], y: &[Idx]) {
        let Self { shape, r_step, W, r, norm, w_step, .. } = self;
        let n_columns = shape.kernel_column_volume().as_();
        let norm = norm.as_();
        shape.sparse_unbiased_increment_repeated(W, *w_step, x, y);
        shape.unique_fired_output_neurons(y, |k| {
            normalize_mat_column(n_columns, k.as_(), W, l(norm));
            r[k.as_()] -= *r_step;
        });
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SwtaLayer<Idx: Debug + PrimInt + NumAssign+ Send+Sync> {
    shape: ConvShape<Idx>,
    pub cos_sim: bool,
    pub conditional: bool,
    pub use_abs: bool,
    pub W_step: f32,
    pub U_step: f32,
    /// of shape  `[kernel_height, kernel_width, in_channels, out_channels]`
    W: Vec<f32>,
    /// of shape  `[out_channels, out_channels]`. U is row-major. Element `U[k,j]==0` means neuron k (row) can inhibit neuron j (column).
    U: Vec<f32>,
    pub threshold: f32,
    norm: usize,
}

impl<Idx: Debug + PrimInt + NumAssign+ Send+Sync> Debug for SwtaLayer<Idx> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Swta({:?})", self.shape)
    }
}

impl<Idx: Debug + PrimInt + AsPrimitive<usize> + NumAssign+ Send+Sync> SwtaLayer<Idx> {
    pub fn new(shape: ConvShape<Idx>, norm: usize) -> Self {
        let mut W = Vec::rand(shape.minicolumn_w_shape().product().as_());
        let U = Vec::rand(shape.minicolumn_u_shape().product().as_());
        let width = shape.kernel_column_volume().as_();
        normalize_mat_columns(width, &mut W, l(norm));
        Self {
            shape,
            cos_sim: true,
            conditional: false,
            use_abs: false,
            W_step: 0.01,
            U_step: 0.01,
            W,
            U,
            threshold: 0.0001,
            norm,
        }
    }
    pub fn W(&self)->&[f32]{
        &self.W
    }
    pub fn U(&self)->&[f32]{
        &self.U
    }
}

impl<Idx: Debug + PrimInt + AsPrimitive<usize> + Step + FromUsize + NumAssign+ Send+Sync> SwtaLayer<Idx> {
    fn _learn_<const COS_SIM: bool, const CONDITIONAL: bool, const USE_ABS: bool>(&mut self, x: &[Idx], s: &[f32], y: &[Idx]){
        let m = self.m().as_();
        let Self {
             W_step, U_step,
            W, U, norm, ..
        } = self;
        for i in x {
            for k in y {
                W[i.as_() * m + k.as_()] += *W_step;
            }
        }
        for k in y {
            normalize_mat_column(m, k.as_(), W, l(*norm));
        }
        for k in y {
            let k = k.as_();
            for j in 0..m {
                let kj = k * m + j;
                if !CONDITIONAL || s[k] > s[j] {
                    let sk_minus_sj = if COS_SIM { s[k] * s[j] } else { s[k] - s[j] };
                    let sk_minus_sj = if USE_ABS{sk_minus_sj.abs()}else{sk_minus_sj};
                    U[kj] = U[kj] * (1. - *U_step) +  sk_minus_sj * *U_step;
                }
            }
        }
    }
}

impl<Idx: Debug + PrimInt + AsPrimitive<usize> + Step + FromUsize + NumAssign + Send+Sync> Layer<Idx> for SwtaLayer<Idx> {
    fn shape(&self) -> &ConvShape<Idx> {
        &self.shape
    }

    fn run_(&self, x: &[Idx], s: &mut [f32], winner_callback: impl FnMut(usize)) {
        s.fill(0.);
        debug_assert_lt!(x.iter().max().cloned().unwrap_or(Idx::zero()).as_(), self.n().as_(), "{:?}", x);
        dot1_(x, &self.W, s);
        debug_assert_eq!(s.len(), self.m().as_());
        let mut y: Vec<u8> = s.iter().map(|&s_j| if s_j > self.threshold { NULL } else { 0 }).collect();
        debug_assert_eq!(y.len(), self.m().as_());
        crate::soft_wta::top_slice_(&self.U, &s, &mut y, winner_callback);
    }

    fn learn(&mut self, x: &[Idx], s: &[f32], y: &[Idx]) {
        if self.cos_sim{
            if self.conditional{
                if self.use_abs{
                    self._learn_::<true,true,true>(x,s,y);
                }else{
                    self._learn_::<true,true,false>(x,s,y);
                }
            }else{
                if self.use_abs{
                    self._learn_::<true,false,true>(x,s,y);
                }else{
                    self._learn_::<true,false,false>(x,s,y);
                }
            }
        }else{
            if self.conditional{
                if self.use_abs{
                    self._learn_::<false,true,true>(x,s,y);
                }else{
                    self._learn_::<false,true,false>(x,s,y);
                }
            }else{
                if self.use_abs{
                    self._learn_::<false,false,true>(x,s,y);
                }else{
                    self._learn_::<false,false,false>(x,s,y);
                }
            }
        }
    }

    fn run_conv_(&self, x: &[Idx], s: &mut [f32], mut winner_callback: impl FnMut(usize)) {
        s.fill(0.);
        self.shape.sparse_dot_repeated_slice_(x, self.W.as_slice(), s);
        let mut y: Vec<u8> = s.iter().map(|&s_j| if s_j > self.threshold { NULL } else { 0 }).collect();
        crate::soft_wta::top_repeated_conv_(&self.shape.out_shape().as_scalar(), &self.U, &s, &mut y, winner_callback);
    }

    fn learn_conv(&mut self, x: &[Idx], s: &[f32], y: &[Idx]) {
        unimplemented!();
    }
}


#[cfg(test)]
mod tests {
    use crate::{dense_to_sparse, rand_set};
    use super::*;

    #[test]
    fn test1() {
        let mut s1 = HwtaL2Layer::new(ConvShape::new_linear(10, 10), 2);
        let mut s2 = s1.clone();
        let x = rand_set(3, 0u32..10);
        let y1 = s1.train(&x);
        let y2 = s2.train_conv(&x);
        assert_eq!(y1, y2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test3() {
        let sc = ConvShape::new_in([32, 32, 6], 8, [4, 4], [1, 1]);
        let mut s1 = HwtaL2Layer::new(sc.clone(), 2);
        let mut s2 = s1.clone();
        let x = s1.shape().rand_dense_input(600);
        let mut y1 = Vec::<u32>::new();
        sc.conv(&x, |offset, kernel_column| {
            s1.run(&dense_to_sparse(kernel_column), |k| y1.push(offset + k as u32))
        });
        let y2 = s2.run_conv_into_vec(&dense_to_sparse(&x));
        assert_eq!(y1, y2);
    }

    #[test]
    fn test2() {
        let mut s1 = SwtaLayer::<u32>::new(ConvShape::new_linear(10, 8), 2);
        let mut s2 = s1.clone();
        let x = s1.shape().rand_sparse_input(3);
        let y1 = s1.run_into_vec(&x);
        let y2 = s2.run_conv_into_vec(&x);
        assert_eq!(y1, y2);
        // s1.learn(&x,&y1);
        // s2.learn(&x,&y2);
        // assert_eq!(s1,s2);
    }

    #[test]
    fn test4() {
        let sc = ConvShape::new_in([32, 32, 6], 8, [4, 4], [1, 1]);
        let mut s1 = SwtaLayer::<u32>::new(sc.clone(), 2);
        let mut s2 = s1.clone();
        let x = s1.shape().rand_dense_input(600);
        let mut y1 = Vec::<u32>::new();
        sc.conv(&x, |offset, kernel_column| {
            s1.run(&dense_to_sparse(kernel_column), |k| y1.push(offset + k as u32))
        });
        let y2 = s2.run_conv_into_vec(&dense_to_sparse(&x));
        assert_eq!(y1, y2);
    }
}