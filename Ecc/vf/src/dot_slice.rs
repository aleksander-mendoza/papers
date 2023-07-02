use std::ops::{Add, Mul};
use crate::init::{empty, InitEmptyWithCapacity};
use num_traits::{MulAdd, Zero};
use crate::shape::Shape;

pub fn dot3<T: Add<Output=T> + Zero>(lhs: &[T], shape_lhs: &[usize; 3], rhs: &[T], shape_rhs: &[usize; 3]) -> (Vec<T>, [usize; 3]) where for<'a> &'a T: Mul<Output = T>{
    // [Z, Y, W] == shape_lhs;
    // [X, Z, W] == shape_rhs;
    assert_eq!(shape_lhs[0], shape_rhs[1]);
    assert_eq!(shape_lhs[2], shape_rhs[2]);

    let X = shape_rhs[0];
    let Y = shape_lhs[1];
    let Z = shape_lhs[0];
    let W = shape_rhs[2];
    let mut o = Vec::with_capacity(X * Y * W);
    let shape_o = [X, Y, W];
    for x in 0..X {
        for y in 0..Y {
            for w in 0..W {
                debug_assert_eq!(o.len(),shape_o.idx(&[x, y, w]));
                o.push((0..Z).fold(T::zero(), |sum, z| sum + &lhs[shape_lhs.idx(&[z, y, w])] * &rhs[shape_rhs.idx(&[w, z, x])]));
            }
        }
    }
    (o, shape_o)
}

pub fn dot2<T: Add<Output=T> + Zero>(lhs: &[T], shape_lhs: &[usize; 2], rhs: &[T], shape_rhs: &[usize; 2]) -> (Vec<T>, [usize; 2]) where for<'a> &'a T: Mul<Output = T>{
    // shape_lhs == [Z, Y]
    // shape_rhs == [X, Z]
    assert_eq!(shape_lhs[0], shape_rhs[1]);
    let X = shape_rhs[0];
    let Y = shape_lhs[1];
    let Z = shape_lhs[0];
    let mut o = Vec::with_capacity(X * Y);
    let shape_o = [X, Y];
    for x in 0..X {
        for y in 0..Y {
            debug_assert_eq!(o.len(),shape_o.idx(&[x, y]));
            o.push((0..Z).fold(T::zero(), |sum, z| sum + &lhs[shape_lhs.idx(&[z, y])] * &rhs[shape_rhs.idx(&[x, z])]));
        }
    }
    (o, shape_o)
}

pub fn dot1<T: Add<Output=T> + Zero>(lhs: &[T], rhs: &[T], shape_rhs: &[usize; 2]) -> Vec<T> where for<'a> &'a T: Mul<Output=T> {
    assert_eq!(lhs.len(), shape_rhs[1]);
    let mut o = Vec::with_capacity(shape_rhs[0]);
    for x in 0..o.len() {
        o.push(lhs.iter().enumerate().fold(T::zero(), |sum, (z, l)| sum + l * &rhs[shape_rhs.idx(&[x, z])]));
    }
    o
}

pub fn dot0<T: Add<Output=T> + Zero>(lhs: &[T], rhs: &[T]) -> T where for<'a> &'a T: Mul<Output=T> {
    lhs.iter().zip(rhs.iter()).fold(T::zero(), |sum, (l, r)| sum + l * r)
}

pub fn inner_product<T: Add<Output=T> + Zero>(lhs: &[T], rhs: &[T]) -> T where for<'a> &'a T: Mul<Output=T> {
    dot0(lhs, rhs)
}