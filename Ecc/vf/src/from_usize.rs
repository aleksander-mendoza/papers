pub trait FromUsize {
    fn from_usize(u: usize) -> Self;
}
macro_rules! from_usize {
    ($t:ident) => {
        impl FromUsize for $t {
            fn from_usize(u: usize) -> Self {
                u as $t
            }
        }
    };
}
impl FromUsize for bool {
    fn from_usize(u: usize) -> Self {
        u != 0
    }
}
from_usize!(f32);
from_usize!(f64);
from_usize!(usize);
from_usize!(u8);
from_usize!(u16);
from_usize!(u32);
from_usize!(u64);
from_usize!(isize);
from_usize!(i8);
from_usize!(i16);
from_usize!(i32);
from_usize!(i64);
