use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign, MulAdd, MulAddAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use crate::*;
use crate::init::{UninitEmptyWithCapacity, InitEmptyWithCapacity};

impl<T> VectorField<T> for [T] {
    #[inline]
    fn fold<I>(&self, zero: I, mut f: impl FnMut(I, &T) -> I) -> I {
        self.iter().fold(zero, |a, b| f(a, b))
    }

    fn rfold<I>(&self, zero: I, mut f: impl FnMut(I, &T) -> I) -> I {
        self.iter().rfold(zero, |a, b| f(a, b))
    }

    fn fold_<I>(&mut self, zero: I, mut f: impl FnMut(I, &mut T) -> I) -> I {
        self.iter_mut().fold(zero, |a, b| f(a, b))
    }

    fn enumerate_fold_<I>(&mut self, zero: I, mut f: impl FnMut(usize, I, &mut T) -> I) -> I {
        self.iter_mut().enumerate().fold(zero, |a, (i,b)| f(i, a, b))
    }

    fn enumerate_rfold_<I>(&mut self, zero: I, mut f: impl FnMut(usize, I, &mut T) -> I) -> I {
        self.iter_mut().enumerate().rfold(zero, |a, (i,b)| f(i, a, b))
    }

    fn rfold_<I>(&mut self, zero: I, mut f: impl FnMut(I, &mut T) -> I) -> I {
        self.iter_mut().rfold(zero, |a, b| f(a, b))
    }

    fn map_(&mut self, mut f: impl FnMut(&mut T)) -> &mut Self {
        self.iter_mut().for_each(f);
        self
    }
    #[inline]
    fn all(&self, mut f: impl FnMut(&T) -> bool) -> bool {
        self.iter().all(f)
    }

    #[inline]
    fn any(&self, mut f: impl FnMut(&T) -> bool) -> bool {
        self.iter().any(f)
    }

    #[inline]
    fn all_zip(&self, other: &Self, mut f: impl FnMut(&T, &T) -> bool) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| f(a, b))
    }
    #[inline]
    fn any_zip(&self, other: &Self, mut f: impl FnMut(&T, &T) -> bool) -> bool {
        self.iter().zip(other.iter()).any(|(a, b)| f(a, b))
    }
    fn zip_(&mut self, other: &Self, mut f: impl FnMut(&mut T, &T)) -> &mut Self {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| f(a, b));
        self
    }
    type O = Vec<T>;
    #[inline]
    fn map(&self, mut f: impl FnMut(&T) -> T) -> Self::O {
        self.iter().map(f).collect()
    }

    #[inline]
    fn zip(&self, other: &Self, mut f: impl FnMut(&T, &T) -> T) -> Self::O {
        self.iter().zip(other.iter()).map(|(a, b)| f(a, b)).collect()
    }

    #[inline]
    fn zip3(&self, other: &Self, other2: &Self, mut f: impl FnMut(&T, &T, &T) -> T) -> Self::O {
        self.iter().zip(other.iter()).zip(other2.iter()).map(|((a, b),c)| f(a, b, c)).collect()
    }
    fn zip3_(&mut self, other: &Self, other2: &Self, mut f: impl FnMut(&mut T, &T, &T)) -> &mut Self{
        self.iter_mut().zip(other.iter()).zip(other2.iter()).for_each(|((a, b), c)| f(a, b, c));
        self
    }
    fn fold_map<D>(&self, mut zero: D, mut f: impl FnMut(D, &T) -> (D, T)) -> (D, Self::O) {
        let mut arr = Vec::with_capacity(self.len());
        for i in self {
            let (z, a) = f(zero, i);
            zero = z;
            arr.push(a);
        }
        (zero, arr)
    }

    fn rfold_map<D>(&self, mut zero: D, mut f: impl FnMut(D, &T) -> (D, T)) -> (D, Self::O) {
        unsafe {
            let mut arr = Vec::empty_uninit(self.len());
            for (i, j) in self.iter().rev().zip(arr.iter_mut().rev()) {
                let (z, a) = f(zero, i);
                zero = z;
                std::ptr::write(j as *mut T, a);
            }
            (zero, arr)
        }
    }
}

impl<T: Copy + Add<Output=T>> VectorFieldAdd<T> for [T] {}

impl <T: MulAdd<Output=T> + Mul<Output=T> + Add<Output=T> + Copy> VectorFieldMulAdd<T> for [T]{}

impl<T: Copy + Add<Output=T> + Zero> VectorFieldZero<T> for [T] {}

impl<T: Copy + Sub<Output=T>> VectorFieldSub<T> for [T] {}

impl<T: Copy + Div<Output=T>> VectorFieldDiv<T> for [T] {}

impl<T: Copy + Div<Output=T> + Zero> VectorFieldDivDefaultZero<T> for [T] {}

impl<T: Copy + Rem<Output=T> + Zero> VectorFieldRemDefaultZero<T> for [T] {}

impl<T: Copy + Mul<Output=T>> VectorFieldMul<T> for [T] {}

impl<T: Copy + Mul<Output=T> + One> VectorFieldOne<T> for [T] {}

impl<T: Copy + Rem<Output=T>> VectorFieldRem<T> for [T] {}

impl<T: Copy + PartialOrd> VectorFieldPartialOrd<T> for [T] {}

impl<T: Neg<Output=T> + Zero + PartialOrd + Copy> VectorFieldAbs<T> for [T] {}

impl<T: Neg<Output=T> + Copy> VectorFieldNeg<T> for [T] {}


impl<T: Copy + AddAssign> VectorFieldAddAssign<T> for [T] {}

impl <T: MulAddAssign + MulAdd<Output=T> + Mul<Output=T> + Add<Output=T> + Copy>  VectorFieldMulAddAssign<T> for [T]{}

impl<T: Copy + SubAssign> VectorFieldSubAssign<T> for [T] {}

impl<T: Copy + DivAssign> VectorFieldDivAssign<T> for [T] {}

impl<T: Copy + DivAssign + Zero> VectorFieldDivAssignDefaultZero<T> for [T] {}

impl<T: Copy + RemAssign + Zero> VectorFieldRemAssignDefaultZero<T> for [T] {}

impl<T: Copy + MulAssign> VectorFieldMulAssign<T> for [T] {}

impl<T: Copy + RemAssign> VectorFieldRemAssign<T> for [T] {}


impl<T: Num + Copy + PartialOrd> VectorFieldNum<T> for [T] {}

impl<T: NumAssign + Copy + PartialOrd> VectorFieldNumAssign<T> for [T] {}

pub trait VecCast<Scalar: Copy> {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> Vec<IntoScalar> where Scalar: AsPrimitive<IntoScalar>;
}

impl<T: Copy> VecCast<T> for [T] {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> Vec<IntoScalar> where T: AsPrimitive<IntoScalar> {
        self.iter().map(|i| i.as_()).collect()
    }
}

impl<T: Copy> VectorFieldRngAssign<T> for [T] where Standard: Distribution<T> {
    fn rand_(&mut self, rng: &mut impl rand::Rng) -> &mut Self {
        self.iter_mut().for_each(|i| *i = rng.gen::<T>());
        self
    }
}