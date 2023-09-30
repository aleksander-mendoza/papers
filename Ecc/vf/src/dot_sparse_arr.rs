use std::ops::{Add, Mul};
use crate::init::empty;
use num_traits::{MulAdd, Zero};

pub fn dot1<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const X: usize, const Z: usize>(lhs: &[I], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o: [T; X] = empty();
    for x in 0..X {
        o[x] = lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()][x])
    }
    o
}

pub fn dot0<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero, const Z: usize>(lhs: &[I], rhs: &[T; Z]) -> T {
    lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()])
}

pub fn inner_product<I: num_traits::AsPrimitive<usize>, T: Mul + Add + Copy + Zero, const Z: usize>(lhs: &[I], rhs: &[T; Z]) -> T {
    dot0(lhs, rhs)
}