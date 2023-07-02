use std::mem::MaybeUninit;
use crate::init::empty;
use crate::slice_as_arr;

pub unsafe fn write_slice<X>(from:&[X], to:&mut [X]){
    assert_eq!(from.len(),to.len());
    from.as_ptr().copy_to(to.as_mut_ptr(),from.len());
}
pub unsafe fn write_maybe_uninit_slice<X>(from:&[X], to:&mut [MaybeUninit<X>]){
    assert_eq!(from.len(),to.len());
    // we can cast *MaybeUninit<X> to *X because MaybeUninit has repr(transparent)
    from.as_ptr().copy_to(to.as_mut_ptr() as *mut X,from.len());
}
pub fn concat<X:Copy,const L1:usize, const L2:usize>(a:&[X;L1], b:&[X;L2]) ->[X;{L1+L2}]{
    let mut arr:[X;{L1+L2}] = empty();
    arr[..L1].copy_from_slice(a.as_slice());
    arr[L1..].copy_from_slice(b.as_slice());
    arr
}
pub fn append<X:Copy,const L1:usize>(a:&[X;L1], b:&X) ->[X;{L1+1}]{
    concat(a,slice_as_arr::<_,1>(std::slice::from_ref(b)))
}
pub fn _concat<X,const L1:usize, const L2:usize>(a:[X;L1], b:[X;L2]) ->[X;{L1+L2}]{
    unsafe{
        let mut arr:[MaybeUninit<X>;{L1+L2}] = MaybeUninit::uninit_array();
        write_maybe_uninit_slice(a.as_slice(),&mut arr[..L1]);
        write_maybe_uninit_slice(b.as_slice(),&mut arr[L1..]);
        std::mem::forget(a);
        std::mem::forget(b);
        MaybeUninit::array_assume_init(arr)
    }

}
pub fn _append<X,const L1:usize>(a:[X;L1], b:X) ->[X;{L1+1}]{
    _concat(a,[b])
}

pub fn _map_arr<T,D, const DIM: usize>(arr:[T; DIM], mut f: impl FnMut(T) -> D) -> [D;DIM] {
    let mut a: [MaybeUninit<D>; DIM] = MaybeUninit::uninit_array();
    for (o,i) in a.iter_mut().zip(arr.into_iter()) {
        o.write(f(i));
    }
    unsafe{MaybeUninit::array_assume_init(a)}
}
pub fn map_arr<T,D, const DIM: usize>(arr:&[T; DIM], mut f: impl FnMut(&T) -> D) -> [D;DIM] {
    let mut a: [MaybeUninit<D>; DIM] = MaybeUninit::uninit_array();
    for i in 0..DIM {
        a[i].write(f(&arr[i]));
    }
    unsafe{MaybeUninit::array_assume_init(a)}
}
pub fn zip_arr<T,D,E, const DIM: usize>(arr1:&[T; DIM],arr2:&[D; DIM], mut f: impl FnMut(&T, &D) -> E) -> [E;DIM] {
    let mut arr: [MaybeUninit<E>; DIM] = MaybeUninit::uninit_array();
    for i in 0..DIM {
        arr[i].write(f(&arr1[i], &arr2[i]));
    }
    unsafe{MaybeUninit::array_assume_init(arr)}
}
pub fn zip3_arr<T,D,E,F, const DIM: usize>(arr1:&[T; DIM],arr2:&[D; DIM],arr3:&[E; DIM], mut f: impl FnMut(&T, &D, &E) -> F) -> [F;DIM] {
    let mut arr: [MaybeUninit<F>; DIM] = MaybeUninit::uninit_array();
    for i in 0..DIM {
        arr[i].write(f(&arr1[i], &arr2[i], &arr3[i]));
    }
    unsafe{MaybeUninit::array_assume_init(arr)}
}
pub fn _zip_arr<T,D,E, const DIM: usize>(arr1:[T; DIM],arr2:[D; DIM], mut f: impl FnMut(T, D) -> E) -> [E;DIM] {
    let mut arr: [MaybeUninit<E>; DIM] = MaybeUninit::uninit_array();
    for ((l,r),o) in arr1.into_iter().zip(arr2.into_iter()).zip(arr.iter_mut()) {
        o.write(f(l,r));
    }
    unsafe{MaybeUninit::array_assume_init(arr)}
}
pub fn _zip3_arr<T,D,E,F, const DIM: usize>(arr1:[T; DIM],arr2:[D; DIM],arr3:[E; DIM], mut f: impl FnMut(T, D, E) -> F) -> [F;DIM] {
    let mut arr: [MaybeUninit<F>; DIM] = MaybeUninit::uninit_array();
    for (((l,m),r),o) in arr1.into_iter().zip(arr2.into_iter()).zip(arr3.into_iter()).zip(arr.iter_mut()) {
        o.write(f(l,m,r));
    }
    unsafe{MaybeUninit::array_assume_init(arr)}
}