use std::fmt::Debug;
use num_traits::{AsPrimitive, PrimInt};
use crate::ecc_layer::Layer;
use crate::from_usize::FromUsize;

// pub struct Machine<Idx:Debug + PrimInt + FromUsize + AsPrimitive<usize>, L:Layer<Idx>>{
//     layer:Vec<L>
// }
//
// impl <Idx:Debug + PrimInt + FromUsize + AsPrimitive<usize>, L:Layer<Idx>> Machine<Idx,L>{
//     pub fn new(input_shape:[Idx;3], kernels:&[[Idx;2]], strides:&[[Idx;2]], channels:&[Idx])->Self{
//
//     }
// }