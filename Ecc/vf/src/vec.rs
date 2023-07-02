use crate::SameSize;

pub fn _map_vec<A:SameSize<B>,B:SameSize<A>>(mut v: Vec<A>, mut f:impl FnMut(A)->B) -> Vec<B> {
    for a in &mut v{
        unsafe{
            let ptr = a as *mut A;
            (ptr as *mut B).write(f(ptr.read()))
        }
    }
    unsafe{std::mem::transmute(v)}
}
pub fn _map_box<A:SameSize<B>,B:SameSize<A>>(mut v: Box<A>, mut f:impl FnMut(A)->B) -> Box<B> {
    unsafe{
        let ptr = v.as_mut() as *mut A;
        (ptr as *mut B).write(f(ptr.read()));
        std::mem::transmute(v)
    }
}
pub fn _map_boxed_slice<A:SameSize<B>,B:SameSize<A>>(mut v: Box<[A]>, mut f:impl FnMut(A)->B) -> Box<[B]> {
    for a in v.as_mut(){
        unsafe{
            let ptr = a as *mut A;
            (ptr as *mut B).write(f(ptr.read()))
        }
    }
    unsafe{std::mem::transmute(v)}
}
pub fn _map_boxed_arr<A:SameSize<B>,B:SameSize<A>,const DIM:usize>(mut v: Box<[A;DIM]>, mut f:impl FnMut(A)->B) -> Box<[B;DIM]> {
    for a in v.as_mut(){
        unsafe{
            let ptr = a as *mut A;
            (ptr as *mut B).write(f(ptr.read()))
        }
    }
    unsafe{std::mem::transmute(v)}
}

pub fn flatten_vec<T, const DIM: usize>(v: Vec<[T; DIM]>) -> Vec<T> {
    flatten_box(v.into_boxed_slice()).into_vec()
}

pub fn flatten_box<T, const DIM: usize>(mut v: Box<[[T; DIM]]>) -> Box<[T]> {
    // SAFETY: raw pointer is created from Box and
    // *mut [[T; N]] has the same aligment as *mut [T]
    let len = v.len() * DIM;
    let ptr = std::ptr::slice_from_raw_parts_mut(Box::into_raw(v).cast(), len);
    unsafe { Box::from_raw(ptr) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test6() {
        let v: Box<[[u32; 13]]> = unsafe { Box::<[[u32; 13]]>::new_uninit_slice(7).assume_init() };
        let l = v.len();
        let v = flatten_box(v);
        assert_eq!(v.len(), l * 13);
    }
}