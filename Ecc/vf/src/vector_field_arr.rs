use crate::*;
use std::ops::{Add, Sub, Div, Mul, Rem, Index, IndexMut, Neg, AddAssign, SubAssign, DivAssign, MulAssign, RemAssign};
use std::mem::MaybeUninit;
use num_traits::{Zero, One, Num, AsPrimitive, NumAssign, MulAdd, MulAddAssign};
use rand::Rng;
use rand::distributions::{Standard, Distribution};
use crate::init::{empty, empty_uninit};


impl<T, const DIM: usize> VectorField<T> for [T; DIM] {
    fn fold<D>(&self, zero: D, mut f: impl FnMut(D, &T) -> D) -> D {
        self.iter().fold(zero, |a, b| f(a, b))
    }

    fn rfold<D>(&self, zero: D, mut f: impl FnMut(D, &T) -> D) -> D {
        self.iter().rfold(zero, |a, b| f(a, b))
    }

    fn fold_<D>(&mut self, zero: D, mut f: impl FnMut(D, &mut T) -> D) -> D {
        self.iter_mut().fold(zero, |a, b| f(a, b))
    }

    fn enumerate_fold_<D>(&mut self, zero: D, mut f: impl FnMut(usize, D, &mut T) -> D) -> D {
        self.iter_mut().enumerate().fold(zero, |a, (i,b)| f(i, a, b))
    }

    fn enumerate_rfold_<D>(&mut self, zero: D, mut f: impl FnMut(usize, D, &mut T) -> D) -> D {
        self.iter_mut().enumerate().rfold(zero, |a,(i, b)| f(i, a, b))
    }

    fn rfold_<D>(&mut self, zero: D, mut f: impl FnMut(D, &mut T) -> D) -> D {
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
    type O = Self;
    #[inline]
    fn map(&self, mut f: impl FnMut(&T) -> T) -> Self {
        map_arr(self,f)
    }

    #[inline]
    fn zip(&self, other: &Self, mut f: impl FnMut(&T, &T) -> T) -> Self {
        zip_arr(self,other,f)
    }

    fn zip3(&self, other: &Self, other2: &Self, mut f: impl FnMut(&T, &T, &T) -> T) -> Self::O {
        zip3_arr(self,other,other2,f)
    }
    fn zip3_(&mut self, other: &Self, other2: &Self, mut f: impl FnMut(&mut T, &T, &T)) -> &mut Self{
        self.iter_mut().zip(other.iter()).zip(other2.iter()).for_each(|((a, b),c)| f(a, b, c));
        self
    }
    fn fold_map<D>(&self, mut zero: D, mut f: impl FnMut(D, &T) -> (D, T)) -> (D, Self::O) {
        let mut arr: [MaybeUninit<T>; DIM] = MaybeUninit::uninit_array();
        for i in 0..DIM {
            let (z, a) = f(zero, &self[i]);
            zero = z;
            arr[i].write(a);
        }
        (zero, unsafe{MaybeUninit::array_assume_init(arr)})
    }

    fn rfold_map<D>(&self, mut zero: D, mut f: impl FnMut(D, &T) -> (D, T)) -> (D, Self::O) {
        let mut arr: [MaybeUninit<T>; DIM] = MaybeUninit::uninit_array();
        for i in (0..DIM).rev() {
            let (z, a) = f(zero, &self[i]);
            zero = z;
            arr[i].write(a);
        }
        (zero, unsafe{MaybeUninit::array_assume_init(arr)})
    }
}

impl<T: Copy, const DIM: usize> VectorFieldOwned<T> for [T; DIM] {
    fn _map(mut self, mut f: impl FnMut(T) -> T) -> Self {
        self.iter_mut().for_each(|e| *e = f(*e));
        self
    }
    fn _zip(mut self, other: &Self, mut f: impl FnMut(T, T) -> T) -> Self {
        self.iter_mut().zip(other.iter().cloned()).for_each(|(a, b)| *a = f(*a, b));
        self
    }
    fn _zip3(mut self, other: &Self,other2: &Self, mut f: impl FnMut(T, T, T) -> T) -> Self {
        self.iter_mut().zip(other.iter()).zip(other2.iter()).for_each(|((a,b), c)| *a = f(*a, *b, *c));
        self
    }
}

impl<T: Copy + Add<Output=T>, const DIM: usize> VectorFieldAdd<T> for [T; DIM] {}

impl <T: MulAdd<Output=T> + Mul<Output=T> + Add<Output=T> + Copy, const DIM: usize> VectorFieldMulAdd<T> for [T;DIM]{}

impl<T: Copy + Add<Output=T> + Zero, const DIM: usize> VectorFieldZero<T> for [T; DIM] {}

impl<T: Copy + Sub<Output=T>, const DIM: usize> VectorFieldSub<T> for [T; DIM] {}

impl<T: Copy + Div<Output=T>, const DIM: usize> VectorFieldDiv<T> for [T; DIM] {}

impl<T: Copy + Div<Output=T> + Zero, const DIM: usize> VectorFieldDivDefaultZero<T> for [T; DIM] {}

impl<T: Copy + Rem<Output=T> + Zero, const DIM: usize> VectorFieldRemDefaultZero<T> for [T; DIM] {}

impl<T: Copy + Mul<Output=T>, const DIM: usize> VectorFieldMul<T> for [T; DIM] {}

impl<T: Copy + Mul<Output=T> + One, const DIM: usize> VectorFieldOne<T> for [T; DIM] {}

impl<T: Copy + Rem<Output=T>, const DIM: usize> VectorFieldRem<T> for [T; DIM] {}

impl<T: Copy + PartialOrd, const DIM: usize> VectorFieldPartialOrd<T> for [T; DIM] {}

impl<T: Neg<Output=T> + Zero + PartialOrd + Copy, const DIM: usize> VectorFieldAbs<T> for [T; DIM] {}

impl<T: Neg<Output=T> + Copy, const DIM: usize> VectorFieldNeg<T> for [T; DIM] {}


impl<T: Copy + Add<Output=T>, const DIM: usize> VectorFieldAddOwned<T> for [T; DIM] {}

impl<T: Copy + Sub<Output=T>, const DIM: usize> VectorFieldSubOwned<T> for [T; DIM] {}

impl<T: Copy + Div<Output=T>, const DIM: usize> VectorFieldDivOwned<T> for [T; DIM] {}

impl<T: Copy + Div<Output=T> + Zero, const DIM: usize> VectorFieldDivDefaultZeroOwned<T> for [T; DIM] {}

impl<T: Copy + Rem<Output=T> + Zero, const DIM: usize> VectorFieldRemDefaultZeroOwned<T> for [T; DIM] {}

impl<T: Copy + Mul<Output=T>, const DIM: usize> VectorFieldMulOwned<T> for [T; DIM] {}

impl<T: Copy + Rem<Output=T>, const DIM: usize> VectorFieldRemOwned<T> for [T; DIM] {}

impl<T: Copy + PartialOrd, const DIM: usize> VectorFieldPartialOrdOwned<T> for [T; DIM] {}

impl<T: Neg<Output=T> + Zero + PartialOrd + Copy, const DIM: usize> VectorFieldAbsOwned<T> for [T; DIM] {}

impl<T: Neg<Output=T> + Copy, const DIM: usize> VectorFieldNegOwned<T> for [T; DIM] {}


impl<T: Copy + AddAssign, const DIM: usize> VectorFieldAddAssign<T> for [T; DIM] {}

impl <T: MulAddAssign + MulAdd<Output=T> + Mul<Output=T> + Add<Output=T> + Copy, const DIM: usize>  VectorFieldMulAddAssign<T> for [T;DIM]{}

impl<T: Copy + SubAssign, const DIM: usize> VectorFieldSubAssign<T> for [T; DIM] {}

impl<T: Copy + DivAssign, const DIM: usize> VectorFieldDivAssign<T> for [T; DIM] {}

impl<T: Copy + DivAssign + Zero, const DIM: usize> VectorFieldDivAssignDefaultZero<T> for [T; DIM] {}

impl<T: Copy + RemAssign + Zero, const DIM: usize> VectorFieldRemAssignDefaultZero<T> for [T; DIM] {}

impl<T: Copy + MulAssign, const DIM: usize> VectorFieldMulAssign<T> for [T; DIM] {}

impl<T: Copy + RemAssign, const DIM: usize> VectorFieldRemAssign<T> for [T; DIM] {}

impl<T: Num + Copy + PartialOrd, const DIM: usize> VectorFieldNum<T> for [T; DIM] {}

impl<T: Num + Copy + PartialOrd, const DIM: usize> VectorFieldNumOwned<T> for [T; DIM] {}

impl<T: NumAssign + Copy + PartialOrd, const DIM: usize> VectorFieldNumAssign<T> for [T; DIM] {}

pub trait ArrayCast<Scalar: Copy, const DIM: usize> {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> [IntoScalar; DIM] where Scalar: AsPrimitive<IntoScalar>;
}

impl<T: Copy, const DIM: usize> ArrayCast<T, DIM> for [T; DIM] {
    fn as_scalar<IntoScalar: 'static + Copy>(&self) -> [IntoScalar; DIM] where T: AsPrimitive<IntoScalar> {
        let mut arr: [IntoScalar; DIM] = empty();
        for i in 0..DIM {
            arr[i] = self[i].as_();
        }
        arr
    }
}

impl<T: Copy, const DIM: usize> VectorFieldRngOwned<T> for [T; DIM] where Standard: Distribution<T> {
    fn _rand(mut self, rng: &mut impl Rng) -> Self {
        self.iter_mut().map(|i| *i = rng.gen::<T>());
        self
    }
}

impl<T: Copy, const DIM: usize> VectorFieldRngAssign<T> for [T; DIM] where Standard: Distribution<T> {
    fn rand_(&mut self, rng: &mut impl rand::Rng) -> &mut Self {
        self.iter_mut().map(|i| *i = rng.gen::<T>());
        self
    }
}

impl<T: Copy, const DIM: usize> VectorFieldFull<T> for [T; DIM] {}

impl<T: Copy + Zero, const DIM: usize> VectorFieldInitZero<T> for [T; DIM] {}

impl<T: Copy + One, const DIM: usize> VectorFieldInitOne<T> for [T; DIM] {}

// pub fn rand_vec(&self, rng: &mut impl rand::Rng)->Self{
//     let mut s:[T;DIM] = empty();
//     for i in 0..DIM{
//         s[i] = rng.gen::<T>()%self[i];
//     }
//     s
// }
