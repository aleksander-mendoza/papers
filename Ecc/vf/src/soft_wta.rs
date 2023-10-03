use std::cmp::Ordering::{Greater, Less};
use std::iter::Step;
use std::ops::{Add, Mul};
use itertools::Itertools;
use num_traits::{AsPrimitive, One, Zero};
use crate::conv_shape::{channels, height, width};
use crate::shape::Shape;
use crate::VectorFieldOne;
use crate::xyzw::z3;

pub const NULL:u8 = 2;

pub trait Swta:Copy{
    fn inhibits(t_kj:Self,s_k:f32,s_j:f32)->bool;
}
impl Swta for bool{
    fn inhibits(v_kj: bool, s_k: f32, s_j: f32) -> bool {
        v_kj && s_k > s_j
    }
}
impl Swta for f32{
    fn inhibits(u_kj: f32, s_k: f32, s_j: f32) -> bool {
        s_j + u_kj < s_k
    }
}

/**shape of s is `[height, width, channels]`, shape of u is `[channels, channels]`,
 shape of y is `[height, width, channels]`. u is C-contiguous.
 Element `s[k,j]==0` means neuron k (row) can inhibit neuron j (column).*/
pub fn top_repeated_conv_<T:Swta>(y_shape:&[usize;3], u:&[T], s:&[f32], y:&mut [u8]){
    let m = y_shape.product();
    let c = channels(y_shape).clone();
    assert_eq!(y.len(),m);
    assert_eq!(s.len(),m);
    assert_eq!(u.len(),c*c);
    for j0 in 0..y_shape[0]{
        for j1 in 0..y_shape[1]{
            let y_from_j = (j0*c+j1)*c;
            let y_to_j = y_from_j+c;
            top_slice_(u,&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }
}
/**shape of s is `[height, width, channels]`, shape of u is `[height, width, channels, channels]`,
 shape of y is `[height, width, channels]`. u is C-contiguous.
 Element `s[y,x,k,j]==0` means neuron k (row) can inhibit neuron j (column) within the minicolumn at position [y,x]. */
pub fn top_conv_<T:Swta>(y_shape:&[usize;3], u:&[T], s:&[f32], y:&mut [u8]){
    let m = y_shape.product();
    let c = channels(y_shape).clone();
    assert_eq!(y.len(),m);
    assert_eq!(s.len(),m);
    assert_eq!(u.len(),m*c);
    for j0 in 0..y_shape[0]{
        for j1 in 0..y_shape[1]{
            let y_from_j = (j0*c+j1)*c;
            let y_to_j = y_from_j+c;
            let u_from_j = y_from_j*c;
            let u_to_j = u_from_j+c*c;
            top_slice_(&u[u_from_j..u_to_j],&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }
}

/**u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column). */
pub fn top_slice<T:Swta>(u:&[T], s:&[f32]) ->Vec<bool>{
    assert_eq!(u.len(), s.len()*s.len());
    top(|k,j|u[k*s.len()+j],s)
}
pub fn ordered_top_slice<T:Swta,I:AsPrimitive<usize>>(u:&[T],s:&[f32], si:impl IntoIterator<Item=I>)->Vec<bool>{
    assert_eq!(u.len(), s.len()*s.len());
    ordered_top(|k,j|u[k*s.len()+j],s, si)
}


/**u is row-major. Element `u[k,j]==0` means neuron k (row) can inhibit neuron j (column). */
pub fn top_slice_<T:Swta>(u:&[T], s:&[f32],y:&mut [u8]){
    assert_eq!(u.len(), s.len()*s.len());
    top_(|k,j|u[k*s.len()+j],s, y)
}
pub fn ordered_top_slice_<T:Swta,I:AsPrimitive<usize>>(u:&[T],s:&[f32], si:impl IntoIterator<Item=I>, y:&mut [u8]){
    assert_eq!(u.len(), s.len()*s.len());
    ordered_top_(|k,j|u[k*s.len()+j],s, si,y)
}

pub fn top_<T:Swta>(u:impl Fn(usize,usize)->T,s:&[f32],y:&mut [u8]){
    let mut si:Vec<u32> = (0..y.len() as u32).collect();
    si.sort_by(|&a,&b|s[a as usize].total_cmp(&s[b as usize])); // sort in descending order
    ordered_top_(u,s,si,y);
    debug_assert!(!y.contains(&NULL));
}

pub fn ordered_top_<T:Swta,I:AsPrimitive<usize>>(u:impl Fn(usize,usize)->T,s:&[f32], si:impl IntoIterator<Item=I>, y:&mut [u8]){
    debug_assert_eq!(y.len(),s.len());
    for k in si.into_iter().map(I::as_){
        if y[k] == NULL {
            y[k] = 1;
            for j in 0..s.len() {
                if y[j] == NULL && T::inhibits(u(k, j),s[k],s[j]) {
                    y[j] = 0;
                }
            }
        }
    }
}
pub fn top<T:Swta>(u:impl Fn(usize,usize)->T,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![NULL;s.len()];
    top_(u,s,&mut y);
    unsafe{std::mem::transmute(y)}
}
pub fn ordered_top<T:Swta, I:AsPrimitive<usize>>(u:impl Fn(usize,usize)->T,s:&[f32], si:impl IntoIterator<Item=I>)->Vec<bool>{
    let mut y:Vec<u8> = vec![NULL;s.len()];
    ordered_top_(u,s,si,&mut y);
    y.iter_mut().for_each(|y|if *y==NULL{*y=0});
    unsafe{std::mem::transmute(y)}
}



#[cfg(test)]
mod tests {
    use crate::init_rand::InitRandWithCapacity;
    use crate::VectorFieldPartialOrd;
    use super::*;


    #[test]
    fn test_real(){
        let l = 20;
        for _ in 0..10{
            let s = Vec::<f32>::rand(l);
            let u = Vec::<f32>::rand(l*l);
            let y = top_slice(&u,&s);
            assert!(y.contains(&true));
            for (j, ye) in y.iter().cloned().enumerate(){
                if !ye{
                    let mut shunned = false;
                    for k in 0..l{
                        if s[k] - s[j] < u[k*l+j]{
                            shunned = true;
                            break;
                        }
                    }
                    assert!(shunned);
                }
            }
        }

    }


    #[test]
    fn test_bool(){
        let l = 20;
        for _ in 0..10{
            let s = Vec::<f32>::rand(l);
            let u = Vec::<bool>::rand(l*l);
            let y = top_slice(&u,&s);
            assert!(y.contains(&true));
            for (j, ye) in y.iter().cloned().enumerate(){
                if !ye{
                    let mut shunned = false;
                    for k in 0..l{
                        if u[k*l+j]  && s[k] > s[j] {
                            shunned = true;
                            break;
                        }
                    }
                    assert!(shunned);
                }
            }
        }

    }
}