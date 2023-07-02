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

/**shape of s is [height, width, channels], shape of v is [channels, channels],
 shape of y is [height, width, channels]. u is C-contiguous.
 Element s[k,j]==0 means neuron k (row) can inhibit neuron j (column).*/
pub fn top_v_repeated_conv_(y_shape:&[usize;3], v:&[bool], s:&[f32], y:&mut [u8]){
    let m = y_shape.product();
    let c = channels(y_shape).clone();
    assert_eq!(y.len(),m);
    assert_eq!(s.len(),m);
    assert_eq!(v.len(),m*c);
    for j0 in 0..y_shape[0]{
        for j1 in 0..y_shape[1]{
            let y_from_j = (j0*c+j1)*c;
            let y_to_j = y_from_j+c;
            top_v_slice_(&v,&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }

}

/**shape of s is [height, width, channels], shape of v is [height, width, channels, channels],
 shape of y is [height, width, channels]. u is C-contiguous.
 Element s[y,x,k,j]==1 means neuron k (row) can inhibit neuron j (column) within the minicolumn at position [y,x].*/
pub fn top_v_conv_(y_shape:&[usize;3], v:&[bool], s:&[f32], y:&mut [u8]){
    let m = y_shape.product();
    let c = channels(y_shape).clone();
    assert_eq!(y.len(),m);
    assert_eq!(s.len(),m);
    assert_eq!(v.len(),c*c);
    for j0 in 0..y_shape[0]{
        for j1 in 0..y_shape[1]{
            let y_from_j = (j0*c+j1)*c;
            let y_to_j = y_from_j+c;
            let v_from_j = y_from_j*c;
            let v_to_j = v_from_j+c*c;
            top_v_slice_(&v[v_from_j..v_to_j],&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }

}

/**u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column). */
pub fn top_v_slice(v:&[bool], s:&[f32]) ->Vec<bool>{
    assert_eq!(v.len(), s.len()*s.len());
    top_v(|k,j|v[k*s.len()+j],s)
}

/**u is row-major. Element v[k,j]==1 means neuron k (row) can inhibit neuron j (column). */
pub fn top_v_slice_(v:&[bool], s:&[f32], y:&mut [u8]){
    assert_eq!(v.len(), s.len()*s.len());
    top_v_(|k,j|v[k*s.len()+j],s, y)
}

pub fn top_v(v:impl Fn(usize,usize)->bool,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![2;s.len()];
    top_v_(v,s,&mut y);
    unsafe{std::mem::transmute(y)}
}
pub fn top_v_(v:impl Fn(usize,usize)->bool,s:&[f32], y:&mut [u8]){
    while let Some(k) = y.iter().cloned().enumerate().filter(|&(_,o)|o==NULL).map(|(k,_)|k).max_by(|&k,&j|if s[k] < s[j]{Less}else{Greater}){
        debug_assert_eq!(y[k],NULL);
        y[k] = 1;
        for j in 0..s.len(){
            if y[j] == NULL{
                if v(k,j) && s[k] > s[j]{
                    y[j] = 0;
                }
            }
        }
    }
    debug_assert!(!y.contains(&NULL));
}

/**shape of s is [height, width, channels], shape of u is [channels, channels],
 shape of y is [height, width, channels]. u is C-contiguous.
 Element s[k,j]==0 means neuron k (row) can inhibit neuron j (column).*/
pub fn top_u_repeated_conv_(y_shape:&[usize;3], u:&[f32], s:&[f32], y:&mut [u8]){
    let m = y_shape.product();
    let c = channels(y_shape).clone();
    assert_eq!(y.len(),m);
    assert_eq!(s.len(),m);
    assert_eq!(u.len(),c*c);
    for j0 in 0..y_shape[0]{
        for j1 in 0..y_shape[1]{
            let y_from_j = (j0*c+j1)*c;
            let y_to_j = y_from_j+c;
            top_u_slice_(u,&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }

}
/**shape of s is [height, width, channels], shape of u is [height, width, channels, channels],
 shape of y is [height, width, channels]. u is C-contiguous.
 Element s[y,x,k,j]==0 means neuron k (row) can inhibit neuron j (column) within the minicolumn at position [y,x]. */
pub fn top_u_conv_(y_shape:&[usize;3], u:&[f32], s:&[f32], y:&mut [u8]){
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
            top_u_slice_(&u[u_from_j..u_to_j],&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }
}

/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn top_u_slice(u:&[f32], s:&[f32]) ->Vec<bool>{
    assert_eq!(u.len(), s.len()*s.len());
    top_u(|k,j|u[k*s.len()+j],s)
}


/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn top_u_slice_(u:&[f32], s:&[f32],y:&mut [u8]){
    assert_eq!(u.len(), s.len()*s.len());
    top_u_(|k,j|u[k*s.len()+j],s, y)
}

pub fn top_u_(u:impl Fn(usize,usize)->f32,s:&[f32],y:&mut [u8]){
    while let Some(k) = y.iter().cloned().enumerate().filter(|&(_,o)|o==NULL).map(|(k,_)|k).max_by(|&k,&j|if s[k] < s[j]{Less}else{Greater}){
        debug_assert_eq!(y[k],NULL);
        y[k] = 1;
        for j in 0..s.len(){
            if y[j] == NULL{
                if s[j] + u(k,j) < s[k]{
                    y[j] = 0;
                }
            }
        }
    }
    debug_assert!(!y.contains(&NULL));
}
pub fn top_u(u:impl Fn(usize,usize)->f32,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![NULL;s.len()];
    top_u_(u,s,&mut y);
    unsafe{std::mem::transmute(y)}
}



/**shape of s is [height, width, channels], shape of u is [channels, channels],
 shape of y is [height, width, channels].
u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn multiplicative_top_u_repeated_conv_(y_shape:&[usize;3], u:&[f32], s:&[f32],y:&mut [u8]){
    let m = y_shape.product();
    let c = channels(y_shape).clone();
    assert_eq!(y.len(),m);
    assert_eq!(s.len(),m);
    assert_eq!(u.len(),c*c);
    for j0 in 0..y_shape[0]{
        for j1 in 0..y_shape[1]{
            let y_from_j = (j0*c+j1)*c;
            let y_to_j = y_from_j+c;
            multiplicative_top_u_slice_(u,&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }
}

/**shape of s is [height, width, channels], shape of u is [height, width, channels, channels],
 shape of y is [height, width, channels].
 u is C-contiguous.
 Element s[y,x,k,j]==0 means neuron k (row) can inhibit neuron j (column) within the minicolumn at position [y,x]. */
pub fn multiplicative_top_u_conv_(y_shape:&[usize;3], u:&[f32], s:&[f32],y:&mut [u8]){
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
            multiplicative_top_u_slice_(&u[u_from_j..u_to_j],&s[y_from_j..y_to_j],&mut y[y_from_j..y_to_j])
        }
    }
}

/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn multiplicative_top_u_slice(u:&[f32], s:&[f32]) ->Vec<bool>{
    assert_eq!(u.len(), s.len()*s.len());
    multiplicative_top_u(|k,j|u[k*s.len()+j],s)
}

/**u is row-major. Element u[k,j]==0 means neuron k (row) can inhibit neuron j (column). */
pub fn multiplicative_top_u_slice_(u:&[f32], s:&[f32],y:&mut [u8]){
    assert_eq!(u.len(), s.len()*s.len());
    multiplicative_top_u_(|k,j|u[k*s.len()+j],s, y)
}

pub fn multiplicative_top_u_(u:impl Fn(usize,usize)->f32,s:&[f32],y:&mut [u8]){
    while let Some(k) = y.iter().cloned().enumerate().filter(|&(_,o)|o==NULL).map(|(k,_)|k).max_by(|&k,&j|if s[k] < s[j]{Less}else{Greater}){
        debug_assert_eq!(y[k],NULL);
        y[k] = 1;
        for j in 0..s.len(){
            if y[j] == NULL{
                if s[k] * s[j] > u(k,j) {
                    y[j] = 0;
                }
            }
        }
    }
    debug_assert!(!y.contains(&NULL));
}
pub fn multiplicative_top_u(u:impl Fn(usize,usize)->f32,s:&[f32])->Vec<bool>{
    let mut y:Vec<u8> = vec![NULL;s.len()];
    multiplicative_top_u_(u,s,&mut y);
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
            let y = top_u_slice(&u,&s);
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
            let y = top_v_slice(&u,&s);
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