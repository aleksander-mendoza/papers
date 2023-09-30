use std::mem::MaybeUninit;
use std::ops::{Add, Mul};
use num_traits::Zero;
use crate::init::UninitEmptyWithCapacity;
use crate::VectorField;

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a column*/
pub fn fold_columns<A:Clone,D>(width:usize, matrix: &[D], init:&mut[A], fold:impl Fn(A,&D)->A){
    for j in 0..width{
        let modulo_equivalence_class = &matrix[j..];
        init[j] = modulo_equivalence_class.iter().step_by(width).fold(init[j].clone(),&fold);
    }
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a row*/
pub fn fold_rows<A:Clone, D>(width:usize, matrix: &[D], init:&mut[A], fold:impl Fn(A,&D)->A){
    let mut from = 0;
    let mut j = 0;
    while from < matrix.len(){
        let to = from + width;
        let row = &matrix[from..to];
        init[j]= row.iter().fold(init[j].clone(),&fold);
        from = to;
        j += 1;
    }
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a column*/
pub fn fold_columns_mut<A:Clone,D>(width:usize, matrix: &mut [D], init:&mut[A], fold:impl Fn(A,&mut D)->A){
    for j in 0..width{
        let mut a = init[j].clone();
        let mut i = j;
        while i < matrix.len(){
            a = fold(a,&mut matrix[i]);
            i += width;
        }
        init[j] = a;
    }
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function folds all elements in a row*/
pub fn fold_rows_mut<A:Clone, D>(width:usize, matrix: &mut [D], init:&mut[A], fold:impl Fn(A,&mut D)->A){
    let mut from = 0;
    let mut j = 0;
    while from < matrix.len(){
        let to = from + width;
        let row = &mut matrix[from..to];
        let mut a = init[j].clone();
        for r in row{
            a = fold(a,r);
        }
        init[j]= a;
        from = to;
        j += 1;
    }
}

/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function sums all elements in a column*/
pub fn sum_mat_columns<A:Clone+Add<Output=A>>(width:usize, matrix: &[A], sums:&mut[A]){
    fold_columns(width, matrix, sums, |a, b|a+b.clone())
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function sums all elements in a row*/
pub fn sum_mat_rows<A:Clone+Add<Output=A>>(width:usize, matrix: &[A], sums:&mut[A]){
    fold_rows(width, matrix, sums, |a, b|a+b.clone())
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies all elements in a column*/
pub fn product_mat_columns<A:Clone+Mul<Output=A>>(width:usize, matrix: &[A], products:&mut[A]){
    fold_columns(width, matrix, products, |a, b|a*b.clone())
}
/**C-contiguous matrix of shape [height, width]. Stride is equal to width. This function multiplies all elements in a row*/
pub fn product_mat_rows<A:Clone+Mul<Output=A>>(width:usize, matrix: &[A], products:&mut[A]){
    fold_rows(width, matrix, products, |a, b|a*b.clone())
}

pub fn transpose<A:Copy>(matrix: &[A], rows:usize)->Vec<A>{
    let columns = matrix.len() / rows;
    let mut o = unsafe{Vec::empty_uninit(matrix.len())};
    for row in 0..rows{
        for col in 0..columns{
            let i = row * columns + col;
            let j = col * rows + row;
            o[j] = matrix[i];
        }
    }
    o
}