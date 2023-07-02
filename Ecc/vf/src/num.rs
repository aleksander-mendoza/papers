use std::ops::{Div, DivAssign, Mul, Neg, Rem};
use std::ptr::slice_from_raw_parts;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use crate::*;

pub trait NegAssign {
    fn neg_assign(&mut self);
}


impl <T:Copy+Neg<Output=T>> NegAssign for T{
    fn neg_assign(&mut self) {
        *self = -*self
    }
}

pub trait Abs {
    type Output;
    fn abs(self) -> Self::Output;
}
macro_rules! norm_abs {
    ($t:ident) => {
        impl Abs for $t{
            type Output = $t;

            fn abs(self) -> Self::Output {
                self.abs()
            }
        }
    };
}
/**identity function*/
macro_rules! norm_id {
    ($t:ident) => {
        impl Abs for $t{
            type Output = $t;

            fn abs(self) -> Self::Output {
                self
            }
        }
    };
}

norm_abs!(f32);
norm_abs!(f64);
norm_id!(bool);
norm_id!(usize);
norm_id!(u8);
norm_id!(u16);
norm_id!(u32);
norm_id!(u64);
norm_abs!(isize);
norm_abs!(i8);
norm_abs!(i16);
norm_abs!(i32);
norm_abs!(i64);



pub trait Dist<Rhs = Self> {
    type Output;
    fn dist(self, other: Rhs) -> Self::Output;
}
/**Use this macro only for primitive types*/
macro_rules! dist_metric_induced_by_norm {
    ($t:ident) => {
        impl Dist for $t{
            type Output = $t;

            fn dist(self,other:Self) -> Self::Output {
                if self > other{ // works even with unsigned primitives
                    self - other
                }else{
                    other - self
                }
            }
        }
    };
}
impl Dist for bool {
    type Output = bool;

    fn dist(self, other: Self) -> Self::Output {
        self == other // this is discrete metric
    }
}

dist_metric_induced_by_norm!(f32);
dist_metric_induced_by_norm!(f64);
dist_metric_induced_by_norm!(usize);
dist_metric_induced_by_norm!(u8);
dist_metric_induced_by_norm!(u16);
dist_metric_induced_by_norm!(u32);
dist_metric_induced_by_norm!(u64);
dist_metric_induced_by_norm!(isize);
dist_metric_induced_by_norm!(i8);
dist_metric_induced_by_norm!(i16);
dist_metric_induced_by_norm!(i32);
dist_metric_induced_by_norm!(i64);


fn square<X: Copy+Mul<Output=X>>(x: X) -> X {
    x * x
}

impl<const DIM: usize> Dist for &[f32; DIM] {
    type Output = f32;

    fn dist(self, other: Self) -> Self::Output {
        (0..DIM).map(|i| square(self[i] - other[i])).sum::<Self::Output>().sqrt()
    }
}

impl<const DIM: usize> Dist for &[f64; DIM] {
    type Output = f64;

    fn dist(self, other: Self) -> Self::Output {
        (0..DIM).map(|i| square(self[i] - other[i])).sum::<Self::Output>().sqrt()
    }
}

impl Dist for &[f32] {
    type Output = f32;

    fn dist(self, other: Self) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(&a, &b)| square(a - b)).sum::<Self::Output>().sqrt()
    }
}

impl Dist for &[f64] {
    type Output = f64;

    fn dist(self, other: Self) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(&a, &b)| square(a - b)).sum::<Self::Output>().sqrt()
    }
}

impl Dist for &str {
    type Output = usize;

    fn dist(self, other: Self) -> Self::Output {
        levenshtein::levenshtein(self, other)
    }
}


/**This trait signals that two types have the same size (in memory)*/
pub unsafe trait SameSize<Target: Sized>: Sized where Target:SameSize<Self> {
}

macro_rules! same_size {
    ($t:ident, $d:ident) => {
        unsafe impl SameSize<$t> for $d{
        }
    };
}


same_size!(i8,i8);
same_size!(i8,u8);
same_size!(u8,i8);
same_size!(u8,u8);

same_size!(i16,i16);
same_size!(i16,u16);
same_size!(u16,i16);
same_size!(u16,u16);

same_size!(f32,i32);
same_size!(f32,u32);
same_size!(f32,f32);
same_size!(i32,i32);
same_size!(i32,u32);
same_size!(i32,f32);
same_size!(u32,i32);
same_size!(u32,u32);
same_size!(u32,f32);


same_size!(f64,i64);
same_size!(f64,u64);
same_size!(f64,f64);
same_size!(i64,i64);
same_size!(i64,u64);
same_size!(i64,f64);
same_size!(u64,i64);
same_size!(u64,u64);
same_size!(u64,f64);

unsafe impl<T: SameSize<D>, D: SameSize<T>, const DIM: usize> SameSize<[T; DIM]> for [D; DIM] {
}



pub trait RemDiv<Rhs = Self>: Rem<Rhs> + Div<Rhs> {
    fn rem_div(self, rhs: Rhs) -> (<Self as Rem<Rhs>>::Output, <Self as Div<Rhs>>::Output);
}

impl<T: Rem<Rhs> + Div<Rhs> + Clone, Rhs: Clone> RemDiv<Rhs> for T {
    fn rem_div(self, rhs: Rhs) -> (<Self as Rem<Rhs>>::Output, <Self as Div<Rhs>>::Output) {
        (self.clone() % rhs.clone(), self / rhs)
    }
}



pub trait RemDivAssign<Rhs = Self>: Rem<Rhs> + DivAssign<Rhs> {
    fn rem_div_assign(&mut self, rhs: Rhs) -> <Self as Rem<Rhs>>::Output;
}

impl<T: Rem<Rhs> + DivAssign<Rhs> + Clone, Rhs: Clone> RemDivAssign<Rhs> for T {
    fn rem_div_assign(&mut self, rhs: Rhs) -> <Self as Rem<Rhs>>::Output {
        let r = self.clone().rem(rhs.clone());
        self.div_assign(rhs);
        r
    }
}


pub trait Norm {
    type Output;
    fn norm(self) -> Self::Output;
}
macro_rules! norm_abs {
    ($t:ident) => {
        impl Norm for $t{
            type Output = $t;

            fn norm(self) -> Self::Output {
                self.abs()
            }
        }
    };
}
/**identity function*/
macro_rules! norm_id {
    ($t:ident) => {
        impl Norm for $t{
            type Output = $t;

            fn norm(self) -> Self::Output {
                self
            }
        }
    };
}

norm_abs!(f32);
norm_abs!(f64);
norm_id!(bool);
norm_id!(usize);
norm_id!(u8);
norm_id!(u16);
norm_id!(u32);
norm_id!(u64);
norm_abs!(isize);
norm_abs!(i8);
norm_abs!(i16);
norm_abs!(i32);
norm_abs!(i64);

impl <T:Norm+Copy> Norm for &T {
    type Output = T::Output;

    fn norm(self) -> Self::Output {
        self.norm()
    }
}
impl Norm for (f32, f32) {
    type Output = f32;

    fn norm(self) -> Self::Output {
        let (a, b) = self;
        f32::sqrt(a * a + b * b)
    }
}

impl Norm for (f64, f64) {
    type Output = f64;

    fn norm(self) -> Self::Output {
        let (a, b) = self;
        f64::sqrt(a * a + b * b)
    }
}

impl Norm for &str {
    type Output = usize;

    fn norm(self) -> Self::Output {
        self.len()
    }
}

impl Norm for &[f32] {
    type Output = f32;

    fn norm(self) -> Self::Output {
        l2(self, 1)
    }
}

impl Norm for &[f64] {
    type Output = f64;

    fn norm(self) -> Self::Output {
        l2(self, 1)
    }
}