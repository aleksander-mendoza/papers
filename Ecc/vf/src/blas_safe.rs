use num_complex::{Complex32, Complex64};

#[repr(C)]
#[derive(Copy, Clone)]
pub enum Order {
    RowMajor = 101,
    ColMajor = 102,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum Transpose {
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum Symmetry {
    Upper = 121,
    Lower = 122,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum Diagonal {
    NonUnit = 131,
    Unit = 132,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum Side {
    Left = 141,
    Right = 142,
}

pub trait Vector<T> {
    /// The stride within the vector. For example, if `inc` returns 7, every
    /// 7th element is used. Defaults to 1.
    fn inc(&self) -> i32 {
        1
    }
    /// The number of elements in the vector.
    fn len(&self) -> i32;
    /// An unsafe pointer to a contiguous block of memory.
    fn as_slice(&self) -> &[T];
    /// An unsafe mutable pointer to a contiguous block of memory.
    fn as_mut_slice(&mut self) -> &mut [T];
    /// Check if Vector is empty
    fn is_empty(&self) -> bool {
        self.len() == 0i32
    }
}

/// Methods that allow a type to be used in BLAS functions as a matrix.
pub trait Matrix<T> : Vector<T> {
    /// The leading dimension of the matrix. Defaults to `cols` for `RowMajor`
    /// order and 'rows' for `ColMajor` order.
    fn lead_dim(&self) -> u32 {
        match self.order() {
            Order::RowMajor => self.cols(),
            Order::ColMajor => self.rows(),
        }
    }
    /// The order of the matrix. Defaults to `RowMajor`.
    fn order(&self) -> Order {
        Order::RowMajor
    }
    /// Returns the number of rows. It is assumed that rows*cols==len
    fn rows(&self) -> u32;
    /// Returns the number of columns. It is assumed that rows*cols==len
    fn cols(&self) -> u32;
}


pub trait BandMatrix<T>: Matrix<T> {
    fn sub_diagonals(&self) -> u32;
    fn sup_diagonals(&self) -> u32;

    fn as_matrix(&self) -> &dyn Matrix<T>;
}

pub trait Copy: Sized {
    /// Copies `src.len()` elements of `src` into `dst`.
    fn copy(src: &(impl ?Sized + Vector<Self>), dst: &mut (impl ?Sized + Vector<Self>));
}

impl Copy for f32 {
    fn copy(src: &(impl ?Sized + Vector<Self>), dst: &mut (impl ?Sized + Vector<Self>)) {
        unsafe{
            let i = dst.inc();
            blas::scopy(dst.len(), src.as_slice(), src.inc(), dst.as_mut_slice(), i);
        }
    }
}
impl Copy for f64 {
    fn copy(src: &(impl ?Sized + Vector<Self>), dst: &mut (impl ?Sized + Vector<Self>)) {
        unsafe{
            let i = dst.inc();
            blas::dcopy(dst.len(), src.as_slice(), src.inc(), dst.as_mut_slice(), i)
        }
    }
}
impl Copy for Complex32 {
    fn copy(src: &(impl ?Sized + Vector<Self>), dst: &mut (impl ?Sized + Vector<Self>)) {
        unsafe{
            let i = dst.inc();
            blas::ccopy(dst.len(), src.as_slice(), src.inc(), dst.as_mut_slice(), i)
        }
    }
}
impl Copy for Complex64 {
    fn copy(src: &(impl ?Sized + Vector<Self>), dst: &mut (impl ?Sized + Vector<Self>)) {
        unsafe{
            let i = dst.inc();
            blas::zcopy(dst.len(), src.as_slice(), src.inc(), dst.as_mut_slice(), i)
        }
    }
}

/// Computes `a * x + y` and stores the result in `y`.
pub trait Axpy: Sized {
    fn axpy(alpha: &Self, x: &(impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>));
}
impl Axpy for f32 {
    fn axpy(alpha: &Self, x: &(impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = y.inc();
            blas::saxpy(x.len(),*alpha,
                              x.as_slice(), x.inc(),
                              y.as_mut_slice(), i);
        }
    }
}
impl Axpy for f64 {
    fn axpy(alpha: &Self, x: &(impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = y.inc();
            blas::daxpy(x.len(),*alpha,
                        x.as_slice(), x.inc(),
                        y.as_mut_slice(), i);
        }
    }
}
impl Axpy for Complex32 {
    fn axpy(alpha: &Self, x: &(impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = y.inc();
            blas::caxpy(x.len(),*alpha,
                        x.as_slice(), x.inc(),
                        y.as_mut_slice(), i);
        }
    }
}
impl Axpy for Complex64 {
    fn axpy(alpha: &Self, x: &(impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = y.inc();
            blas::zaxpy(x.len(),*alpha,
                        x.as_slice(), x.inc(),
                        y.as_mut_slice(), i);
        }
    }
}
/// Computes `a * x` and stores the result in `x`.
pub trait Scal: Sized {
    fn scal(alpha: &Self, x: &mut (impl ?Sized + Vector<Self>));
}
impl Scal for f32{
    fn scal(alpha: &Self, x: &mut (impl ?Sized + Vector<Self>)) {
        unsafe {
            let i = x.inc();
            blas::sscal(x.len(), *alpha, x.as_mut_slice(),i);
        }
    }
}
impl Scal for f64{
    fn scal(alpha: &Self, x: &mut (impl ?Sized + Vector<Self>)) {
        unsafe {
            let i = x.inc();
            blas::dscal(x.len(), *alpha, x.as_mut_slice(),i);
        }
    }
}
impl Scal for Complex32{
    fn scal(alpha: &Self, x: &mut (impl ?Sized + Vector<Self>)) {
        unsafe {
            let i = x.inc();
            blas::cscal(x.len(), *alpha, x.as_mut_slice(),i);
        }
    }
}
impl Scal for Complex64{
    fn scal(alpha: &Self, x: &mut (impl ?Sized + Vector<Self>)) {
        unsafe {
            let i = x.inc();
            blas::zscal(x.len(), *alpha, x.as_mut_slice(),i);
        }
    }
}

/// Swaps the content of `x` and `y`.
pub trait Swap: Sized {
    /// If they are different lengths, the shorter length is used.
    fn swap(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>));
}
impl Swap for f32{
    fn swap(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = x.inc();
            let j = y.inc();
            blas::sswap(x.len(), x.as_mut_slice(), i, y.as_mut_slice(), j);
        }
    }
}
impl Swap for f64{
    fn swap(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = x.inc();
            let j = y.inc();
            blas::dswap(x.len(), x.as_mut_slice(), i, y.as_mut_slice(), j);
        }
    }
}
impl Swap for Complex32{
    fn swap(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = x.inc();
            let j = y.inc();
            blas::cswap(x.len(), x.as_mut_slice(), i, y.as_mut_slice(), j);
        }
    }
}
impl Swap for Complex64{
    fn swap(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>)) {
        assert_eq!(x.len(),y.len());
        unsafe {
            let i = x.inc();
            let j = y.inc();
            blas::zswap(x.len(), x.as_mut_slice(), i, y.as_mut_slice(), j);
        }
    }
}

/// Computes `x^T * y`.
pub trait Dot: Sized {
    fn dot(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self;
}
impl Dot for f32{
    fn dot(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self {
        assert_eq!(x.len(),y.len());
        unsafe {
            blas::sdot(x.len(),x.as_slice(), x.inc(),y.as_slice(), y.inc())
        }
    }
}
impl Dot for f64{
    fn dot(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self {
        assert_eq!(x.len(),y.len());
        unsafe {
            blas::ddot(x.len(),x.as_slice(), x.inc(),y.as_slice(), y.inc())
        }
    }
}
impl Dot for Complex32{
    fn dot(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self {
        assert_eq!(x.len(),y.len());
        let mut pres = Self::default();
        unsafe {
            blas::cdotu(std::slice::from_mut(&mut pres), x.len(),x.as_slice(), x.inc(),y.as_slice(), y.inc())
        }
        pres
    }
}
impl Dot for Complex64{
    fn dot(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self {
        assert_eq!(x.len(),y.len());
        let mut pres = Self::default();
        unsafe {
            blas::zdotu(std::slice::from_mut(&mut pres), x.len(),x.as_slice(), x.inc(),y.as_slice(), y.inc())
        }
        pres
    }
}
/// Computes `c^T * y` where `c` is a complex conjugate of `x`.
pub trait Dotc: Sized {
    fn dotc(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self;
}
impl Dotc for Complex32{
    fn dotc(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self {
        assert_eq!(x.len(),y.len());
        let mut pres = Self::default();
        unsafe {
            blas::cdotc(std::slice::from_mut(&mut pres), x.len(),x.as_slice(), x.inc(),y.as_slice(), y.inc())
        }
        pres
    }
}
impl Dotc for Complex64{
    fn dotc(x: &(impl ?Sized + Vector<Self>), y: &(impl ?Sized + Vector<Self>)) -> Self {
        assert_eq!(x.len(),y.len());
        let mut pres = Self::default();
        unsafe {
            blas::zdotc(std::slice::from_mut(&mut pres), x.len(),x.as_slice(), x.inc(),y.as_slice(), y.inc())
        }
        pres
    }
}

/// Computes the sum of the absolute values of elements in a vector.
///
/// Complex vectors use `||Re(x)||_1 + ||Im(x)||_1`
pub trait Asum: Sized {
    type Output;
    fn asum(x: &(impl ?Sized + Vector<Self>)) -> Self::Output;
}
impl Asum for f32{
    type Output = Self;
    fn asum(x: &(impl ?Sized + Vector<Self>)) -> Self {
        unsafe{
            blas::sasum(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Asum for f64{
    type Output = Self;
    fn asum(x: &(impl ?Sized + Vector<Self>)) -> Self {
        unsafe{
            blas::dasum(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Asum for Complex32{
    type Output = f32;
    fn asum(x: &(impl ?Sized + Vector<Self>)) -> Self::Output {
        unsafe{
            blas::scasum(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Asum for Complex64{
    type Output = f64;
    fn asum(x: &(impl ?Sized + Vector<Self>)) -> Self::Output {
        unsafe{
            blas::dzasum(x.len(),x.as_slice(),x.inc())
        }
    }
}

/// Computes the L2 norm (Euclidian length) of a vector.
pub trait Nrm2: Sized {
    type Output;
    fn nrm2(x: &(impl ?Sized + Vector<Self>)) -> Self::Output;
}
impl Nrm2 for f32{
    type Output = Self;

    fn nrm2(x: &(impl ?Sized + Vector<Self>)) -> Self {
        unsafe{
            blas::snrm2(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Nrm2 for f64{
    type Output = Self;

    fn nrm2(x: &(impl ?Sized + Vector<Self>)) -> Self {
        unsafe{
            blas::dnrm2(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Nrm2 for Complex32{
    type Output = f32;

    fn nrm2(x: &(impl ?Sized + Vector<Self>)) -> Self::Output {
        unsafe{
            blas::scnrm2(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Nrm2 for Complex64{
    type Output = f64;

    fn nrm2(x: &(impl ?Sized + Vector<Self>)) -> Self::Output {
        unsafe{
            blas::dznrm2(x.len(),x.as_slice(),x.inc())
        }
    }
}

/// Finds the index of the element with maximum absolute value in a vector.
///
/// Complex vectors maximize the value `|Re(x_k)| + |Im(x_k)|`.
///
/// The first index with a maximum is returned.
pub trait Iamax: Sized {
    fn iamax(x: &(impl ?Sized + Vector<Self>)) -> usize;
}
impl Iamax for f32{
    fn iamax(x: &(impl ?Sized + Vector<Self>)) -> usize {
        unsafe{
            blas::isamax(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Iamax for f64{
    fn iamax(x: &(impl ?Sized + Vector<Self>)) -> usize {
        unsafe{
            blas::idamax(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Iamax for Complex32{
    fn iamax(x: &(impl ?Sized + Vector<Self>)) -> usize {
        unsafe{
            blas::icamax(x.len(),x.as_slice(),x.inc())
        }
    }
}
impl Iamax for Complex64{
    fn iamax(x: &(impl ?Sized + Vector<Self>)) -> usize {
        unsafe{
            blas::izamax(x.len(),x.as_slice(),x.inc())
        }
    }
}

/// Applies a Givens rotation matrix to a pair of vectors, where `cos` is
/// the value of the cosine of the angle in the Givens matrix, and `sin` is
/// the sine.
pub trait Rot: Sized {
    type R;
    fn rot(x: &mut (impl ?Sized + Vector<Self>),y: &mut (impl ?Sized + Vector<Self>), cos: &Self::R, sin: &Self::R);
}
impl Rot for f32{
    type R = Self;

    fn rot(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>), cos: &Self, sin: &Self) {
        assert_eq!(x.len(),y.len());
        let incx = x.inc();
        let incy = y.inc();
        unsafe{
            blas::srot(x.len(),x.as_mut_slice(),incx,y.as_mut_slice(),incy,*cos,*sin)
        }
    }
}
impl Rot for f64{
    type R = Self;

    fn rot(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>), cos: &Self, sin: &Self) {
        assert_eq!(x.len(),y.len());
        let incx = x.inc();
        let incy = y.inc();
        unsafe{
            blas::drot(x.len(),x.as_mut_slice(),incx,y.as_mut_slice(),incy,*cos,*sin)
        }
    }
}
impl Rot for Complex32{
    type R = f32;

    fn rot(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>), cos: &Self::R, sin: &Self::R) {
        assert_eq!(x.len(),y.len());
        let incx = x.inc();
        let incy = y.inc();
        unsafe{
            blas::csrot(x.len(),x.as_mut_slice(),incx,y.as_mut_slice(),incy,*cos,*sin)
        }
    }
}
impl Rot for Complex64{
    type R = f64;

    fn rot(x: &mut (impl ?Sized + Vector<Self>), y: &mut (impl ?Sized + Vector<Self>), cos: &Self::R, sin: &Self::R) {
        assert_eq!(x.len(),y.len());
        let incx = x.inc();
        let incy = y.inc();
        unsafe{
            blas::zdrot(x.len(),x.as_mut_slice(),incx,y.as_mut_slice(),incy,*cos,*sin)
        }
    }
}


pub trait Gemm: Sized {
    fn gemm(
        alpha: &Self,
        at: Transpose,
        a: &dyn Matrix<Self>,
        bt: Transpose,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

/*


macro_rules! gemm_impl(($($t: ident), +) => (
    $(
        impl Gemm for $t {
            fn gemm(alpha: &$t, at: Transpose, a: &dyn Matrix<$t>, bt: Transpose, b: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    let (m, k)  = match at {
                        Transpose::NoTrans => (a.rows(), a.cols()),
                        _ => (a.cols(), a.rows()),
                    };

                    let n = match bt {
                        Transpose::NoTrans => b.cols(),
                        _ => b.rows(),
                    };

                    prefix!($t, gemm)(a.order(),
                        at, bt,
                        m, n, k,
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

gemm_impl!(f32, f64, Complex32, Complex64);

#[cfg(test)]
mod gemm_tests {
    use crate::attribute::Transpose;
    use crate::matrix::ops::Gemm;
    use crate::matrix::tests::M;
    use std::iter::repeat;

    #[test]
    fn real() {
        let a = M(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = M(2, 2, vec![-1.0, 3.0, 1.0, 1.0]);
        let t = Transpose::NoTrans;

        let mut c = M(2, 2, repeat(0.0).take(4).collect());
        Gemm::gemm(&1f32, t, &a, t, &b, &0f32, &mut c);

        assert_eq!(c.2, vec![1.0, 5.0, 1.0, 13.0]);
    }

    #[test]
    fn transpose() {
        let a = M(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = M(2, 3, vec![-1.0, 3.0, 1.0, 1.0, 1.0, 1.0]);
        let t = Transpose::Trans;

        let mut c = M(2, 2, repeat(0.0).take(4).collect());
        Gemm::gemm(&1f32, t, &a, t, &b, &0f32, &mut c);

        assert_eq!(c.2, vec![13.0, 9.0, 16.0, 12.0]);
    }
}

pub trait Symm: Sized {
    fn symm(
        side: Side,
        symmetry: Symmetry,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

pub trait Hemm: Sized {
    fn hemm(
        side: Side,
        symmetry: Symmetry,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

macro_rules! symm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, alpha: &$t, a: &dyn Matrix<$t>, b: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(),
                        side, symmetry,
                        a.rows(), b.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

symm_impl!(Symm, symm, f32, f64, Complex32, Complex64);
symm_impl!(Hemm, hemm, Complex32, Complex64);

pub trait Trmm: Sized {
    fn trmm(
        side: Side,
        symmetry: Symmetry,
        trans: Transpose,
        diag: Diagonal,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &mut dyn Matrix<Self>,
    );
}

pub trait Trsm: Sized {
    fn trsm(
        side: Side,
        symmetry: Symmetry,
        trans: Transpose,
        diag: Diagonal,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &mut dyn Matrix<Self>,
    );
}

macro_rules! trmm_impl(($trait_name: ident, $fn_name: ident, $($t: ident), +) => (
    $(
        impl $trait_name for $t {
            fn $fn_name(side: Side, symmetry: Symmetry, trans: Transpose, diag: Diagonal, alpha: &$t, a: &dyn Matrix<$t>, b: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, $fn_name)(a.order(),
                        side, symmetry, trans, diag,
                        b.rows(), b.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_mut_ptr().as_c_ptr(), b.lead_dim());
                }
            }
        }
    )+
));

trmm_impl!(Trmm, trmm, f32, f64, Complex32, Complex64);
trmm_impl!(Trsm, trsm, Complex32, Complex64);

pub trait Herk: Sized {
    fn herk(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: &Self,
        a: &dyn Matrix<Complex<Self>>,
        beta: &Self,
        c: &mut dyn Matrix<Complex<Self>>,
    );
}

pub trait Her2k: Sized {
    fn her2k(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: Complex<Self>,
        a: &dyn Matrix<Complex<Self>>,
        b: &dyn Matrix<Complex<Self>>,
        beta: &Self,
        c: &mut dyn Matrix<Complex<Self>>,
    );
}

macro_rules! herk_impl(($($t: ident), +) => (
    $(
        impl Herk for $t {
            fn herk(symmetry: Symmetry, trans: Transpose, alpha: &$t, a: &dyn Matrix<Complex<$t>>, beta: &$t, c: &mut dyn Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, herk)(a.order(),
                        symmetry, trans,
                        a.rows(), a.cols(),
                        *alpha,
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        *beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }

        impl Her2k for $t {
            fn her2k(symmetry: Symmetry, trans: Transpose, alpha: Complex<$t>, a: &dyn Matrix<Complex<$t>>, b: &dyn Matrix<Complex<$t>>, beta: &$t, c: &mut dyn Matrix<Complex<$t>>) {
                unsafe {
                    prefix!(Complex<$t>, her2k)(a.order(),
                        symmetry, trans,
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        *beta,
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

herk_impl!(f32, f64);

pub trait Syrk: Sized {
    fn syrk(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

pub trait Syr2k: Sized {
    fn syr2k(
        symmetry: Symmetry,
        trans: Transpose,
        alpha: &Self,
        a: &dyn Matrix<Self>,
        b: &dyn Matrix<Self>,
        beta: &Self,
        c: &mut dyn Matrix<Self>,
    );
}

macro_rules! syrk_impl(($($t: ident), +) => (
    $(
        impl Syrk for $t {
            fn syrk(symmetry: Symmetry, trans: Transpose, alpha: &$t, a: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, syrk)(a.order(),
                        symmetry, trans,
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }

        impl Syr2k for $t {
            fn syr2k(symmetry: Symmetry, trans: Transpose, alpha: &$t, a: &dyn Matrix<$t>, b: &dyn Matrix<$t>, beta: &$t, c: &mut dyn Matrix<$t>) {
                unsafe {
                    prefix!($t, syr2k)(a.order(),
                        symmetry, trans,
                        a.rows(), a.cols(),
                        alpha.as_const(),
                        a.as_ptr().as_c_ptr(), a.lead_dim(),
                        b.as_ptr().as_c_ptr(), b.lead_dim(),
                        beta.as_const(),
                        c.as_mut_ptr().as_c_ptr(), c.lead_dim());
                }
            }
        }
    )+
));

syrk_impl!(f32, f64, Complex32, Complex64);
*/