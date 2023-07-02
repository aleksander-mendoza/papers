use std::mem::MaybeUninit;
use std::ops::{Add, Mul, MulAssign};
use std::ptr;
use num_traits::{One, Zero};
use crate::{_append, _concat, _zip_arr, append, map_arr, mat_slice, shape2, shape3, VectorField, VectorFieldMul, VectorFieldMulAssign, xy_z3, xyz_w4, zip_arr};
use crate::arr_concat::concat;
use crate::init::{array_assume_init2, uninit_array2};
use crate::shape::Shape;

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a column*/
pub fn fold_columns<A: Clone, D, const W: usize, const H: usize>(matrix: &[[D; W]; H], init: &mut [A; W], fold: impl Fn(A, &D) -> A) {
    mat_slice::fold_columns(W, matrix.flatten(), init.as_mut_slice(), fold)
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a row*/
pub fn fold_rows<A: Clone, D, const W: usize, const H: usize>(matrix: &[[D; W]; H], init: &mut [A; H], fold: impl Fn(A, &D) -> A) {
    mat_slice::fold_rows(W, matrix.flatten(), init.as_mut_slice(), fold)
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a column*/
pub fn fold_columns_mut<A: Clone, D, const W: usize, const H: usize>(matrix: &mut [[D; W]; H], init: &mut [A; W], fold: impl Fn(A, &mut D) -> A) {
    mat_slice::fold_columns_mut(W, matrix.flatten_mut(), init.as_mut_slice(), fold)
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a row*/
pub fn fold_rows_mut<A: Clone, D, const W: usize, const H: usize>(matrix: &mut [[D; W]; H], init: &mut [A; H], fold: impl Fn(A, &mut D) -> A) {
    mat_slice::fold_rows_mut(W, matrix.flatten_mut(), init.as_mut_slice(), fold)
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function sums all elements in a column*/
pub fn sum_columns<A: Clone + Add<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], sums: &mut [A; W]) {
    fold_columns(matrix, sums, |a, b| a + b.clone())
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function sums all elements in a row*/
pub fn sum_rows<A: Clone + Add<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], sums: &mut [A; H]) {
    fold_rows(matrix, sums, |a, b| a + b.clone())
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies all elements in a column*/
pub fn product_columns<A: Clone + Mul<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], products: &mut [A; W]) {
    fold_columns(matrix, products, |a, b| a * b.clone())
}
pub fn mat3_to_mat4_fill<A:Clone>(mat:[[A;3];3], fill_value:A, diagonal_fill_value:A)->[[A;4];4]{
    let [r0,r1,r2] = mat;
    [
        xyz_w4(r0,fill_value.clone()),
        xyz_w4(r1,fill_value.clone()),
        xyz_w4(r2,fill_value.clone()),
        [fill_value.clone(),fill_value.clone(),fill_value.clone(),diagonal_fill_value.clone()]
    ]
}
/**Same as mat3_to_mat4 but fill_value is 0 and diagonal_fill_value is 1*/
pub fn mat3_to_mat4<A:Zero+One>(mat:[[A;3];3])->[[A;4];4]{
    let [r0,r1,r2] = mat;
    [
        xyz_w4(r0,A::zero()),
        xyz_w4(r1,A::zero()),
        xyz_w4(r2,A::zero()),
        [A::zero(),A::zero(),A::zero(),A::one()]
    ]
}

pub fn mat2_add_column<A>(mat:[[A;2];2], column:[A;2])->[[A;3];2]{
    let [r0,r1] = mat;
    let [c0,c1] = column;
    [
        xy_z3(r0,c0),
        xy_z3(r1,c1),
    ]
}
pub fn mat3_add_column<A>(mat:[[A;3];3], column:[A;3])->[[A;4];3]{
    let [r0,r1,r2] = mat;
    let [c0,c1,c2] = column;
    [
        xyz_w4(r0,c0),
        xyz_w4(r1,c1),
        xyz_w4(r2,c2),
    ]
}
pub fn mat3x2_add_row<A>(mat:[[A;3];2], row:[A;3])->[[A;3];3]{
    xy_z3(mat,row)
}
pub fn mat4x3_add_row<A>(mat:[[A;4];3], row:[A;4])->[[A;4];4]{
    xyz_w4(mat,row)
}
pub fn mat3_add_row<A>(mat:[[A;3];3], row:[A;3])->[[A;3];4]{
    xyz_w4(mat,row)
}
pub fn add_row<A,const W:usize,const H:usize>(mat:[[A;W];H], row:[A;W])->[[A;W];{H+1}]{
    _concat(mat,[row])
}

pub fn add_column<A,const W:usize,const H:usize>(mat:[[A;W];H], column:[A;H])->[[A;{W+1}];H]{
    _zip_arr(mat,column,|row, col|_append(row,col))
}
pub fn mat2_to_mat3<A:Clone>(mat:[[A;2];2], fill_value:A)->[[A;3];3]{
    let [r0,r1] = mat;
    [
        xy_z3(r0,fill_value.clone()),
        xy_z3(r1,fill_value.clone()),
        [fill_value.clone(),fill_value.clone(),fill_value.clone()]
    ]
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies all elements in a row*/
pub fn product_rows<A: Clone + Mul<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], products: &mut [A; H]) {
    fold_rows(matrix, products, |a, b| a * b.clone())
}
pub fn row_wise<A,B,C, const W: usize, const H: usize>(matrix: &[[A; W]; H], vec: &[B; W], mut zip: impl FnMut(&[A; W],&[B;W])->[C;W]) ->[[C;W];H]{
    map_arr(matrix,|row|zip(row,vec))
}
pub fn row_wise_<A,B, const W: usize, const H: usize>(matrix: &mut [[A; W]; H], vec: &[B; W], mut zip: impl FnMut(&mut [A; W],&[B;W])){
    for row in matrix{ zip(row,vec) }
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies matrix with vector element-wise (row-wise)*/
pub fn mul_row_wise_<A: Copy + MulAssign, const W: usize, const H: usize>(matrix: &mut [[A; W]; H], vec: &[A; W]) {
    row_wise_(matrix, vec, |a, b|{a.mul_(b);})
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies matrix with vector element-wise (row-wise)*/
pub fn mul_row_wise<A: Copy + Mul<Output=A>, const W: usize, const H: usize>(matrix: &[[A; W]; H], vec: &[A; W]) -> [[A; W]; H]{
    row_wise(matrix, vec, |a, b|a.mul(b))
}
pub fn diag_with<A:Copy,const W:usize>(zero:A,mut f:impl FnMut(usize)->A) -> [[A;W];W]{
    let mut w = [[zero;W];W];
    for i in 0..W{
        w[i][i] = f(i);
    }
    w
}
pub fn from_diag<A:Copy,const W:usize>(zero:A,diagonal:&[A;W]) -> [[A;W];W]{
    diag_with(zero,|i|diagonal[i])
}
pub fn identity<A:Copy,const W:usize>(zero:A,one:A) -> [[A;W];W]{
    diag_with(zero,|_|one)
}
pub fn id<A:Copy+Zero+One,const W:usize>() -> [[A;W];W]{
    identity(A::zero(),A::one())
}

pub fn transpose<A:Clone, const W:usize, const H:usize>(mat:&[[A;W];H])->[[A;H];W]{
    let mut arr = uninit_array2::<_,H,W>();
    for i in 0..H{
        let r = &mat[i];
        for j in 0..W{
            arr[j][i].write(r[j].clone());
        }
    }
    unsafe{array_assume_init2(arr)}
}

pub fn swap2<A, const W:usize, const H:usize>(mat:&mut [[A;W];H], pos1:&[usize;2], pos2:&[usize;2]){
    let shape = shape2(mat);
    assert!(pos1.lt(&shape),"position1 {:?} out of bounds {:?}", pos1, shape);
    assert!(pos2.lt(&shape),"position2 {:?} out of bounds {:?}", pos2, shape);
    unsafe{
        swap2_unchecked(mat,pos1,pos2)
    }
}
pub unsafe fn swap2_unchecked<A, const W:usize, const H:usize>(mat:&mut [[A;W];H], pos1:&[usize;2], pos2:&[usize;2]){
    let shape = shape2(mat);
    let i1 = shape.idx(&pos1);
    let i2 = shape.idx(&pos2);
    let p = mat.as_mut_ptr() as *mut A;
    unsafe{
        std::ptr::swap(p.add(i1), p.add(i2))
    }
}
pub fn swap3<A, const W:usize, const H:usize, const D:usize>(mat:&mut [[[A;W];H];D], pos1:&[usize;3], pos2:&[usize;3]){
    let shape = shape3(mat);
    assert!(pos1.lt(&shape),"position1 {:?} out of bounds {:?}", pos1, shape);
    assert!(pos2.lt(&shape),"position2 {:?} out of bounds {:?}", pos2, shape);
    unsafe{
        swap3_unchecked(mat,pos1,pos2);
    }
}
pub unsafe fn swap3_unchecked<A, const W:usize, const H:usize, const D:usize>(mat:&mut [[[A;W];H];D], pos1:&[usize;3], pos2:&[usize;3]){
    let shape = shape3(mat);
    let i1 = shape.idx(&pos1);
    let i2 = shape.idx(&pos2);
    let p = mat.as_mut_ptr() as *mut A;
    unsafe{
        std::ptr::swap(p.add(i1), p.add(i2))
    }
}
pub fn transpose_<A, const W:usize>(mat:&mut [[A;W];W]){
    for i in 0..W{
        for j in (i+1)..W{
            unsafe{swap2_unchecked(mat,&[i,j], &[j,i])};
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::init::arange2;
    use crate::init_rand::rand2;
    use super::*;

    #[test]
    fn test1() {
        for _ in 0..5 {
            let mut arr1 = arange2::<u32, 4, 4>();
            let arr2 = transpose(&arr1);
            transpose_(&mut arr1);
            assert_eq!(arr1, arr2);
        }
    }
}