use std::fmt::{Debug, Formatter, Pointer};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use num_traits::{MulAdd, MulAddAssign, One};
use crate::init::{InitEmptyWithCapacity, InitFilled, InitFilledCapacity, InitRFoldWithCapacity, InitFoldWithCapacity};
use crate::{blas_safe, VectorFieldAdd, VectorFieldAddAssign, VectorFieldDiv, VectorFieldDivAssign, VectorFieldMul, VectorFieldMulAssign, VectorFieldOne, VectorFieldRem, VectorFieldRemAssign, VectorFieldSub, VectorFieldSubAssign};

/**C contiguous 2D matrix*/
pub struct Mat<S, const DIM: usize> {
    shape: [u32; DIM],
    data: Box<[S]>,
}
impl<S:Clone, const DIM: usize> Clone for Mat<S, DIM> {
    fn clone(&self) -> Self {
        Self{ shape: self.shape, data: self.data.clone() }
    }
}
impl<S:PartialEq, const DIM: usize> PartialEq for Mat<S, DIM> {
    fn eq(&self, other: &Self) -> bool {
        self.shape==other.shape && self.data==other.data
    }
}
impl<S:Debug, const DIM: usize> Debug for Mat<S, DIM> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.data,f)
    }
}
impl<S:Eq, const DIM: usize> Eq for Mat<S, DIM> {
}
impl<S, const DIM: usize> blas_safe::Vector<S> for Mat<S, DIM> {
    fn len(&self) -> i32 {
        self.data.len() as i32
    }

    fn as_slice(&self) -> &[S] {
        &self.data
    }

    fn as_mut_slice(&mut self) -> &mut [S] {
        &mut self.data
    }
}

impl<S> blas_safe::Matrix<S> for Mat<S, 2> {
    fn rows(&self) -> u32 {
        // shape is [height, width]
        self.shape[0]
    }

    fn cols(&self) -> u32 {
        self.shape[1]
    }

}

impl<S, const DIM: usize> Mat<S, DIM> {
    pub fn new(shape: [u32; DIM], data: Box<[S]>) -> Self {
        assert_eq!(shape.product() as usize, data.len());
        Self {
            shape,
            data,
        }
    }
}

impl<S> From<Vec<S>> for Mat<S, 1> {
    fn from(value: Vec<S>) -> Self {
        Self { shape: [value.len() as u32], data: value.into_boxed_slice() }
    }
}

impl<S:Clone, const DIM: usize> InitFilledCapacity<S> for Mat<S, DIM> {
    type C = [u32; DIM];

    fn init_filled(capacity: Self::C, f: S) -> Self {
        Self { shape: capacity, data: vec![f; capacity.product() as usize].into_boxed_slice() }
    }
}

impl<S:Copy, const DIM: usize> InitEmptyWithCapacity for Mat<S, DIM> {
    type C = [u32; DIM];

    fn empty(capacity: Self::C) -> Self {
        Self { shape: capacity, data: Vec::empty(capacity.product() as usize).into_boxed_slice() }
    }
}

impl<S:Copy, const DIM: usize> InitFoldWithCapacity<S> for Mat<S, DIM> {
    type C = [u32; DIM];

    fn init_fold(capacity: Self::C, start: S, f: impl FnMut(S, usize) -> S) -> Self {
        Self { shape: capacity, data: Vec::init_fold(capacity.product() as usize, start, f).into_boxed_slice() }
    }
}

impl<S:Copy, const DIM: usize> InitRFoldWithCapacity<S> for Mat<S, DIM> {
    type C = [u32; DIM];

    fn init_rfold(capacity: Self::C, end: S, f: impl FnMut(S, usize) -> S) -> Self {
        Self { shape: capacity, data: Vec::init_rfold(capacity.product() as usize, end, f).into_boxed_slice() }
    }
}


impl<'a, S: Copy + Add<Output=S>, const DIM: usize> Add for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn add(self, rhs: Self) -> Self::Output {
        Mat { shape: self.shape, data: self.data.add(&rhs.data).into_boxed_slice() }
    }
}

impl<'a, S: Copy + Sub<Output=S>, const DIM: usize> Sub for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn sub(self, rhs: Self) -> Self::Output {
        Mat { shape: self.shape, data: self.data.sub(&rhs.data).into_boxed_slice() }
    }
}

impl<'a, S: Copy + Mul<Output=S>, const DIM: usize> Mul for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn mul(self, rhs: Self) -> Self::Output {
        Mat { shape: self.shape, data: self.data.mul(&rhs.data).into_boxed_slice() }
    }
}


impl<'a, S: Copy + Div<Output=S>, const DIM: usize> Div for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn div(self, rhs: Self) -> Self::Output {
        Mat { shape: self.shape, data: self.data.div(&rhs.data).into_boxed_slice() }
    }
}


impl<'a, S: Copy + Rem<Output=S>, const DIM: usize> Rem for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn rem(self, rhs: Self) -> Self::Output {
        Mat { shape: self.shape, data: self.data.rem(&rhs.data).into_boxed_slice() }
    }
}


impl<'a, S: Copy + AddAssign, const DIM: usize> AddAssign<&Mat<S, DIM>> for Mat<S, DIM> {
    fn add_assign(&mut self, rhs: &Mat<S, DIM>) {
        self.data.add_(&rhs.data);
    }
}

impl<'a, S: Copy + SubAssign, const DIM: usize> SubAssign<&Mat<S, DIM>> for Mat<S, DIM> {
    fn sub_assign(&mut self, rhs: &Mat<S, DIM>) {
        self.data.sub_(&rhs.data);
    }
}


impl<'a, S: Copy + MulAssign, const DIM: usize> MulAssign<&Mat<S, DIM>> for Mat<S, DIM> {
    fn mul_assign(&mut self, rhs: &Mat<S, DIM>) {
        self.data.mul_(&rhs.data);
    }
}

impl<'a, S: Copy + DivAssign, const DIM: usize> DivAssign<&Mat<S, DIM>> for Mat<S, DIM> {
    fn div_assign(&mut self, rhs: &Mat<S, DIM>) {
        self.data.div_(&rhs.data);
    }
}

impl<'a, S: Copy + RemAssign, const DIM: usize> RemAssign<&Mat<S, DIM>> for Mat<S, DIM> {
    fn rem_assign(&mut self, rhs: &Mat<S, DIM>) {
        self.data.rem_(&rhs.data);
    }
}


impl<'a, S: blas_safe::Axpy, const DIM: usize> MulAddAssign<S,&'a Mat<S, DIM>> for Mat<S, DIM> {
    fn mul_add_assign(&mut self, a: S, b: &'a Mat<S, DIM>) {
        S::axpy(&a,b,self)
    }
}
impl<'a, S: Clone+blas_safe::Axpy, const DIM: usize> MulAdd<S> for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn mul_add(self, a: S, b: Self) -> Self::Output {
        let mut cpy = Mat::clone(self);
        cpy.mul_add_assign(a,b);
        cpy
    }
}
impl<'a, S: Copy + Add<Output=S> + Copy, const DIM: usize> Add<S> for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn add(self, rhs: S) -> Self::Output {
        Mat { shape: self.shape, data: self.data.add_scalar(rhs).into_boxed_slice() }
    }
}

impl<'a, S: Copy + Sub<Output=S>, const DIM: usize> Sub<S> for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn sub(self, rhs: S) -> Self::Output {
        Mat { shape: self.shape, data: self.data.sub_scalar(rhs).into_boxed_slice() }
    }
}

impl<'a, S: Copy + Mul<Output=S>, const DIM: usize> Mul<S> for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn mul(self, rhs: S) -> Self::Output {
        Mat { shape: self.shape, data: self.data.mul_scalar(rhs).into_boxed_slice() }
    }
}


impl<'a, S: Copy + Div<Output=S>, const DIM: usize> Div<S> for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn div(self, rhs: S) -> Self::Output {
        Mat { shape: self.shape, data: self.data.div_scalar(rhs).into_boxed_slice() }
    }
}


impl<'a, S: Copy + Rem<Output=S>, const DIM: usize> Rem<S> for &'a Mat<S, DIM> {
    type Output = Mat<S, DIM>;

    fn rem(self, rhs: S) -> Self::Output {
        Mat { shape: self.shape, data: self.data.rem_scalar(rhs).into_boxed_slice() }
    }
}


impl<'a, S: Copy + AddAssign, const DIM: usize> AddAssign<S> for Mat<S, DIM> {
    fn add_assign(&mut self, rhs: S) {
        self.data.add_scalar_(rhs);
    }
}

impl<'a, S: Copy + SubAssign, const DIM: usize> SubAssign<S> for Mat<S, DIM> {
    fn sub_assign(&mut self, rhs: S) {
        self.data.sub_scalar_(rhs);
    }
}


impl<'a, S: blas_safe::Scal, const DIM: usize> MulAssign<S> for Mat<S, DIM> {
    fn mul_assign(&mut self, rhs: S) {
        S::scal(&rhs, self)
    }
}

impl<'a, S: Copy + One + Div<Output=S> + blas_safe::Scal, const DIM: usize> DivAssign<S> for Mat<S, DIM> {
    fn div_assign(&mut self, rhs: S) {
        let inv = S::one() / rhs;
        S::scal(&inv,self)
    }
}

impl<'a, S: Copy + RemAssign, const DIM: usize> RemAssign<S> for Mat<S, DIM> {
    fn rem_assign(&mut self, rhs: S) {
        self.data.rem_scalar_(rhs);
    }
}

pub trait Dot<Rhs=Self>{
    type Output;
    fn dot(self,other:Rhs)->Self::Output;
}
impl<S: blas_safe::Dot, const DIM: usize> Dot for &Mat<S, DIM> {
    type Output = S;
    fn dot(self, other: Self) -> Self::Output {
        S::dot(self,other)
    }
}

pub trait AbsMax{
    fn abs_max(self)->usize;
}
impl<S: blas_safe::Iamax, const DIM: usize> AbsMax for &Mat<S, DIM> {
    fn abs_max(self) -> usize{
        S::iamax(self)
    }
}
pub trait CopyFrom{
    fn copy_from(&mut self, other:&Self);
}
impl<S: blas_safe::Copy, const DIM: usize> CopyFrom for Mat<S, DIM> {
    fn copy_from(&mut self, other:&Self){
        S::copy(other,self)
    }
}
pub trait Swap{
    fn swap(&mut self, other:&mut Self);
}
impl<S: blas_safe::Swap, const DIM: usize> Swap for Mat<S, DIM> {
    fn swap(&mut self, other:&mut Self){
        S::swap(other,self)
    }
}
pub trait Norm2{
    type Output;
    fn norm2(self)->Self::Output;
}
impl<S: blas_safe::Nrm2, const DIM: usize> Norm2 for &Mat<S, DIM> {
    type Output = S::Output;

    fn norm2(self) -> S::Output{
        S::nrm2(self)
    }
}
/**Inverse*/
pub trait Inv{
    fn inv(self)->Self;
}
// impl<S: blas_safe::Nrm2, const DIM: usize> Inv for &Mat<S, DIM> {
//
// }