use std::fmt::{Debug, Formatter};
use std::iter::Step;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector, Matrix};
use num_traits::{AsPrimitive, PrimInt};
use crate::{ArrayCast, l, normalize_mat_column, normalize_mat_columns, VectorFieldAddAssign, VectorFieldOne};
use crate::conv_shape::ConvShape;
use crate::dot_sparse_slice::dot1;
use crate::from_usize::FromUsize;
use crate::init::InitFilledCapacity;
use crate::init_rand::InitRandWithCapacity;
use crate::soft_wta::NULL;

pub trait Layer<Idx: Debug + PrimInt + FromUsize > {
    fn shape(&self)->&ConvShape<Idx>;
    fn run(&self, x: &[Idx], winner_callback: impl FnMut(usize));
    fn run_into_vec(&self, x: &[Idx])->Vec<Idx> {
        let mut v = Vec::new();
        self.run(x,|k|v.push(Idx::from_usize(k)));
        v
    }
    fn learn(&mut self, x: &[Idx], y:&[Idx]);
    fn run_conv(&self, x: &[Idx], winner_callback: impl FnMut(usize));
    fn run_conv_into_vec(&self, x: &[Idx])->Vec<Idx> {
        let mut v = Vec::new();
        self.run_conv(x,|k|v.push(Idx::from_usize(k)));
        v
    }
    fn learn_conv(&mut self, x: &[Idx], y:&[Idx]);
}

#[derive(Clone, PartialEq)]
pub struct HwtaLayer<Idx: Debug + PrimInt> {
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
impl <Idx: Debug + PrimInt> Debug for HwtaLayer<Idx>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hwta({:?})", self.shape)
    }
}
impl<Idx: Debug + PrimInt + AsPrimitive<usize>> HwtaLayer<Idx> {
    pub fn new(shape:ConvShape<Idx>, norm:usize)->Self{
        let mut W = Vec::rand(shape.minicolumn_w_shape().product().as_());
        let r = Vec::full(shape.out_channels().as_(),0f32);
        let width = shape.kernel_column_volume().as_();
        normalize_mat_columns(width,&mut W, l(norm));
        Self{
            shape,
            r_step: 0.002,
            W,
            r,
            norm,
            w_step: 0.01,
            min_input_cardinality: 1
        }
    }
}
impl<Idx: Debug + PrimInt + AsPrimitive<usize> + Step + FromUsize> Layer<Idx> for HwtaLayer<Idx> {
    fn shape(&self) -> &ConvShape<Idx> {
        &self.shape
    }

    fn run(&self, x: &[Idx], mut winner_callback: impl FnMut(usize)) {
        if x.len() >= self.min_input_cardinality as usize {
            let mut s = dot1(x, &self.W, self.shape.kernel_column_volume().as_());

            s.add_(&self.r);
            let k = s.iter().cloned().position_max_by(f32::total_cmp).unwrap();
            winner_callback(k);
        }
    }

    fn learn(&mut self, x: &[Idx], y: &[Idx]) {
        if let Some(k) = y.first() {
            self.r[k.as_()] -= self.r_step;
            let c = self.shape.out_channels().as_();
            let w_step = self.w_step;
            for i in x{
                self.W[i.as_()*c + k.as_()] += w_step / x.len() as f32;
            }
            let width = self.shape.kernel_column_volume().as_();
            let n = self.norm;
            normalize_mat_column(width,k.as_(),&mut self.W,l(n));
        }
    }

    fn run_conv(&self, x: &[Idx], mut winner_callback: impl FnMut(usize)) {
        if x.len() >= self.min_input_cardinality as usize {
            let mut y = self.shape.sparse_dot_repeated_slice(x, self.W.as_slice());
            ConvShape::<Idx>::add_vector_repeated(&self.r, &mut y);
            let mut offset = 0;
            let c = self.shape.out_channels().as_();
            while offset < y.len() {
                let end = offset + c;
                winner_callback(y[offset..end].iter().cloned().position_max_by(f32::total_cmp).unwrap());
                offset = end;
            }
        }
    }

    fn learn_conv(&mut self, x: &[Idx], y: &[Idx]) {
        let Self{ shape, r_step, W, r, norm, w_step, .. } = self;
        let n_columns = shape.kernel_column_volume().as_();
        let norm = norm.as_();
        shape.sparse_unbiased_increment_repeated(W,*w_step,x,y);
        shape.unique_fired_output_neurons(y,|k|{
            normalize_mat_column(n_columns,k.as_(),W, l(norm));
            r[k.as_()] -= *r_step;
        });
    }
}
#[derive(Clone, PartialEq)]
pub struct SwtaLayer<Idx: Debug + PrimInt> {
    shape: ConvShape<Idx>,
    pub W_step: f32,
    pub U_step: f32,
    /// of shape  `[kernel_height, kernel_width, in_channels, out_channels]`
    W: Vec<f32>,
    /// of shape  `[out_channels, out_channels]`
    U: Vec<f32>,
    pub threshold: f32,
    norm:usize
}
impl <Idx: Debug + PrimInt> Debug for SwtaLayer<Idx>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Swta({:?})", self.shape)
    }
}
impl<Idx: Debug + PrimInt + AsPrimitive<usize>>  SwtaLayer<Idx> {
    pub fn new(shape:ConvShape<Idx>, norm:usize)->Self{
        let mut W = Vec::rand(shape.minicolumn_w_shape().product().as_());
        let U = Vec::rand(shape.minicolumn_u_shape().product().as_());
        let width = shape.kernel_column_volume().as_();
        normalize_mat_columns(width,&mut W, l(norm));
        Self{
            shape,
            W_step: 0.01,
            U_step: 0.01,
            W,
            U,
            threshold: 0.0001,
            norm
        }
    }
}
impl<Idx: Debug + PrimInt + AsPrimitive<usize> + Step + FromUsize> Layer<Idx> for SwtaLayer<Idx> {
    fn shape(&self) -> &ConvShape<Idx> {
        &self.shape
    }

    fn run(&self, x: &[Idx], winner_callback: impl FnMut(usize)) {
        let mut s = dot1(x, &self.W, self.shape.kernel_column_volume().as_());
        let mut y: Vec<u8> = s.iter().map(|&s_j| if s_j > self.threshold { NULL } else { 0 }).collect();
        crate::soft_wta::top_slice_( &self.U, &s, &mut y, winner_callback);
    }

    fn learn(&mut self, x: &[Idx], y: &[Idx]) {
        todo!()
    }

    fn run_conv(&self, x: &[Idx], mut winner_callback: impl FnMut(usize)) {
        let s = self.shape.sparse_dot_repeated_slice(x, self.W.as_slice());
        let mut y: Vec<u8> = s.iter().map(|&s_j| if s_j > self.threshold { NULL } else { 0 }).collect();
        crate::soft_wta::top_repeated_conv_(&self.shape.out_shape().as_scalar(), &self.U, &s, &mut y, winner_callback);
    }

    fn learn_conv(&mut self, x: &[Idx], y: &[Idx]) {
        let Self{ shape, U_step, W, norm, W_step, .. } = self;
        let n_columns = shape.kernel_column_volume().as_();
        let norm = norm.as_();
        shape.sparse_unbiased_increment_repeated(W,*W_step,x,y);
        shape.unique_fired_output_neurons(y,|k|{
            normalize_mat_column(n_columns,k.as_(),W, l(norm));
            // r[k.as_()] -= *r_step;
        });
    }
}



#[cfg(test)]
mod tests {
    use crate::rand_set;
    use super::*;

    #[test]
    fn test1() {
        let mut s1 = HwtaLayer::new(ConvShape::new_linear(10,10),2);
        let mut s2 = s1.clone();
        let x = rand_set(3,0u32..10);
        let y1 = s1.run_into_vec(&x);
        let y2 = s2.run_conv_into_vec(&x);
        assert_eq!(y1,y2);
        s1.learn(&x,&y1);
        s2.learn(&x,&y2);
        assert_eq!(s1,s2);
    }
    #[test]
    fn test2() {
        let mut s1 = SwtaLayer::new(ConvShape::new_linear(10,10),2);
        let mut s2 = s1.clone();
        let x = rand_set(3,0u32..10);
        let y1 = s1.run_into_vec(&x);
        let y2 = s2.run_conv_into_vec(&x);
        assert_eq!(y1,y2);
        // s1.learn(&x,&y1);
        // s2.learn(&x,&y2);
        // assert_eq!(s1,s2);
    }
}