use std::ops::{Add, AddAssign, Mul};
use crate::init::{InitFilledCapacity};
use num_traits::{MulAdd, Zero};


pub fn dot1_t<I: num_traits::AsPrimitive<usize>, T: AddAssign + Copy + Zero>(lhs: &[T], rhs: &[I], rows:usize) -> Vec<T> {
    let mut o = Vec::full(rows, T::zero());
    dot1_t_(lhs,rhs,&mut o);
    o
}
pub fn dot1_t_<I: num_traits::AsPrimitive<usize>, T: AddAssign + Copy + Zero>(lhs: &[T], rhs: &[I], output:&mut [T])  {
    let rows = output.len();
    let columns = lhs.len() / rows;
    for row in 0..rows{
        let offset = columns * row;
        let r = &lhs[offset..offset+columns];
        output[row] = inner_product(rhs, r);
    }
}

pub fn dot1<I: num_traits::AsPrimitive<usize>, T: AddAssign + Copy + Zero>(lhs: &[I], rhs: &[T], columns:usize) -> Vec<T> {
    let mut o = Vec::full(columns, T::zero());
    dot1_(lhs,rhs,&mut o);
    o
}
pub fn dot1_<I: num_traits::AsPrimitive<usize>, T: AddAssign + Copy + Zero>(lhs: &[I], rhs: &[T], output:&mut [T])  {
    let columns = output.len();
    for row in lhs{
        let offset = columns * row.as_();
        output.iter_mut().zip(&rhs[offset..offset+columns]).for_each(|(o,&d)|*o+=d);
    }
}

pub fn dot0<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero>(lhs: &[I], rhs: &[T]) -> T {
    lhs.iter().fold(T::zero(), |sum, z| sum + rhs[z.as_()])
}

pub fn inner_product<I: num_traits::AsPrimitive<usize>, T: Add + Copy + Zero>(lhs: &[I], rhs: &[T]) -> T {
    dot0(lhs, rhs)
}


#[cfg(test)]
mod tests {
    use crate::init_rand::InitRandWithCapacity;
    use crate::mat_slice::transpose;
    use crate::rand_set;
    use super::*;

    #[test]
    fn test0() -> Result<(), String> {
        let s = vec![0,2,3];
        let m = vec![
            1f32,10.,100.,1000.,
            2.,20.,200.,2000.,
            3.,30.,300.,3000.,
            4.,40.,400.,4000.,
            5.,50.,500.,5000.,
        ];
        let m_dot_s = dot1_t(&m,&s, 5);
        assert_eq!(m_dot_s,vec![1101.,2202.,3303.,4404.,5505.]);
        Ok(())
    }
    #[test]
    fn test1() -> Result<(), String> {
        let s = vec![0,2,3];
        let m = vec![
            1f32,10.,100.,1000.,
            2.,20.,200.,2000.,
            3.,30.,300.,3000.,
            4.,40.,400.,4000.,
            5.,50.,500.,5000.,
        ];
        let s_dot_m = dot1(&s,&m, 4);
        assert_eq!(s_dot_m,vec![8.,80.,800.,8000.]);
        Ok(())
    }
    #[test]
    fn test3() -> Result<(), String> {
        let mt = vec![
            1.,4.,7.,
            2.,5.,8.,
            3.,6.,9.
        ];
        let m:Vec<f32> = (1..=9).map(|i|i as f32).collect();
        let mt2 = transpose(&m,3);
        assert_eq!(mt,mt2);
        Ok(())
    }
    #[test]
    fn test2() -> Result<(), String> {
        let rows = 5;
        let cols = 7;
        let m:Vec<f32> = Vec::rand(rows*cols);
        let s = rand_set(4,0..cols as u32);
        let mt = transpose(&m,rows);
        let m_dot_s = dot1_t(&m, &s,rows);
        let s_dot_mt = dot1(&s, &mt,rows);
        assert_eq!(m_dot_s, s_dot_mt);
        Ok(())
    }
}
