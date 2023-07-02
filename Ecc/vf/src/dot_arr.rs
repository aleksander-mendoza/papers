use std::mem::MaybeUninit;
use std::ops::{Add, Mul};
use crate::init::{array_assume_init2, array_assume_init3, empty, uninit_array2, uninit_array3};
use num_traits::{MulAdd, Zero};



pub fn dot3<T: Add<Output = T> + Zero , const X: usize, const Y: usize, const Z: usize, const W: usize>(lhs: &[[[T; Z]; Y]; W], rhs: &[[[T; X]; Z]; W]) -> [[[T; X]; Y]; W]  where for<'a> &'a T: Mul<Output = T>{
    let mut o: [[[MaybeUninit<T>; X]; Y]; W]= uninit_array3();
    for w in 0..W {
        for x in 0..X {
            for y in 0..Y {
                o[w][y][x].write((0..Z).fold(T::zero(), |sum, z| sum + &lhs[w][y][z] * &rhs[w][z][x]));
            }
        }
    }
    unsafe{array_assume_init3(o)}
}

pub fn dot2<T: Add<Output = T> + Zero, const X: usize, const Y: usize, const Z: usize>(lhs: &[[T; Z]; Y], rhs: &[[T; X]; Z]) -> [[T; X]; Y]  where for<'a> &'a T: Mul<Output = T>{
    let mut o: [[MaybeUninit<T>; X]; Y] = uninit_array2();
    for x in 0..X {
        for y in 0..Y {
            o[y][x].write((0..Z).fold(T::zero(), |sum, z| sum + &lhs[y][z] * &rhs[z][x]));
        }
    }
    unsafe{array_assume_init2(o)}
}

pub fn dot1<T: Add<Output = T> + Zero, const X: usize, const Z: usize>(lhs: &[T; Z], rhs: &[[T; X]; Z]) -> [T; X]  where for<'a> &'a T: Mul<Output = T>{
    let mut o: [MaybeUninit<T>; X]  = MaybeUninit::uninit_array();
    for x in 0..X {
        o[x].write((0..Z).fold(T::zero(), |sum, z| sum + &lhs[z] * &rhs[z][x]));
    }
    unsafe{MaybeUninit::array_assume_init(o)}
}

pub fn dot0<T: Add<Output = T> + Zero, const Z: usize>(lhs: &[T; Z], rhs: &[T; Z]) -> T  where for<'a> &'a T: Mul<Output = T>{
    (0..Z).fold(T::zero(), |sum, z| sum + &lhs[z] * &rhs[z])
}

pub fn inner_product<T: Add<Output = T> + Zero, const Z: usize>(lhs: &[T; Z], rhs: &[T; Z]) -> T  where for<'a> &'a T: Mul<Output = T>{
    dot0(lhs, rhs)
}