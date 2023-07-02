use std::ops::{Add, Mul};
use crate::init::empty;
use num_traits::{MulAdd, Zero};
use crate::lin_alg::Dot;

pub fn dot3<T: MulAdd<Output=T> + Copy + Zero, const X: usize, const Y: usize, const Z: usize, const W: usize>(lhs: &[[[T; Z]; Y]; W], rhs: &[[[T; X]; Z]; W]) -> [[[T; X]; Y]; W] {
    let mut o:[[[T; X]; Y]; W] = empty();
    for w in 0..W {
        for x in 0..X {
            for y in 0..Y {
                o[w][y][x] = (0..Z).fold(T::zero(), |sum, z| lhs[w][y][z].mul_add(rhs[w][z][x], sum));
            }
        }
    }
    o
}

pub fn dot2<T: MulAdd<Output=T> + Copy + Zero, const X: usize, const Y: usize, const Z: usize>(lhs: &[[T; Z]; Y], rhs: &[[T; X]; Z]) -> [[T; X]; Y] {
    let mut o:[[T; X]; Y] = empty();
    for x in 0..X {
        for y in 0..Y {
            o[y][x] = (0..Z).fold(T::zero(), |sum, z| lhs[y][z].mul_add(rhs[z][x], sum));
        }
    }
    o
}

pub fn dot1<T: MulAdd<Output=T> + Copy + Zero, const X: usize, const Z: usize>(lhs: &[T; Z], rhs: &[[T; X]; Z]) -> [T; X] {
    let mut o = empty();
    for x in 0..X {
        o[x] = (0..Z).fold(T::zero(), |sum, z| lhs[z].mul_add(rhs[z][x], sum));
    }
    o
}

pub fn dot0<T: MulAdd<Output=T> + Copy + Zero, const Z: usize>(lhs: &[T; Z], rhs: &[T; Z]) -> T {
    (0..Z).fold(T::zero(), |sum, z| lhs[z].mul_add(rhs[z], sum))
}
macro_rules! impl_dot0 {
    ($t:ident) => {
        impl <const X:usize> Dot for &[$t;X]{
            type Output = $t;

            fn dot(self, other: Self) -> Self::Output {
                dot0(self,other)
            }
        }
    };
}
impl_dot0!(f32);
impl_dot0!(f64);
impl_dot0!(usize);
impl_dot0!(u8);
impl_dot0!(u16);
impl_dot0!(u32);
impl_dot0!(u64);
impl_dot0!(isize);
impl_dot0!(i8);
impl_dot0!(i16);
impl_dot0!(i32);
impl_dot0!(i64);

macro_rules! impl_dot1 {
    ($t:ident) => {
        impl <const X:usize, const Z:usize> Dot<&[[$t; X]; Z]> for &[$t;Z] {
            type Output = [$t; X];

            fn dot(self, other: &[[$t; X]; Z]) -> Self::Output {
                dot1(self,other)
            }
        }
    };
}
impl_dot1!(f32);
impl_dot1!(f64);
impl_dot1!(usize);
impl_dot1!(u8);
impl_dot1!(u16);
impl_dot1!(u32);
impl_dot1!(u64);
impl_dot1!(isize);
impl_dot1!(i8);
impl_dot1!(i16);
impl_dot1!(i32);
impl_dot1!(i64);

macro_rules! impl_dot2 {
    ($t:ident) => {
        impl <const X:usize, const Y:usize, const Z:usize> Dot<&[[$t; X]; Z]> for &[[$t; Z]; Y] {
            type Output = [[$t; X]; Y];

            fn dot(self, other: &[[$t; X]; Z]) -> Self::Output {
                dot2(self,other)
            }
        }
    };
}
impl_dot2!(f32);
impl_dot2!(f64);
impl_dot2!(usize);
impl_dot2!(u8);
impl_dot2!(u16);
impl_dot2!(u32);
impl_dot2!(u64);
impl_dot2!(isize);
impl_dot2!(i8);
impl_dot2!(i16);
impl_dot2!(i32);
impl_dot2!(i64);


macro_rules! impl_dot3 {
    ($t:ident) => {
        impl <const X:usize, const Y:usize, const Z:usize, const W:usize> Dot<&[[[$t; X]; Z]; W]> for &[[[$t; Z]; Y]; W]  {
            type Output = [[[$t; X]; Y]; W];

            fn dot(self, other: &[[[$t; X]; Z]; W]) -> Self::Output {
                dot3(self,other)
            }
        }
    };
}
impl_dot3!(f32);
impl_dot3!(f64);
impl_dot3!(usize);
impl_dot3!(u8);
impl_dot3!(u16);
impl_dot3!(u32);
impl_dot3!(u64);
impl_dot3!(isize);
impl_dot3!(i8);
impl_dot3!(i16);
impl_dot3!(i32);
impl_dot3!(i64);
