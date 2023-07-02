use std::fmt::Debug;
use std::ops::{Add, Bound, Div, Mul, Range, RangeBounds, Rem, Sub};
use num_traits::{One, Zero};
use crate::{VectorFieldOne, VectorFieldPartialOrd, VectorFieldSub};
use crate::shape::Shape;

pub fn foreach2d<T: Copy>(range: &Range<[T; 2]>, mut f: impl FnMut([T; 2])) where Range<T>: Iterator<Item=T> {
    for p0 in range.start[0]..range.end[0] {
        for p1 in range.start[1]..range.end[1] {
            f([p0, p1])
        }
    }
}

pub fn contains<T: Copy + PartialOrd, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> bool {
    range.start.all_le(element) && element.all_lt(&range.end)
}

pub fn size<T: Copy + Sub<Output=T> + One + Mul<Output=T>, const DIM: usize>(range: &Range<[T; DIM]>) -> T {
    shape(range).product()
}

pub fn shape<T: Copy + Sub<Output=T>, const DIM: usize>(range: &Range<[T; DIM]>) -> [T; DIM] {
    range.end.sub(&range.start)
}

pub fn relative_to<T: Copy + Sub<Output=T>, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> [T; DIM] {
    element.sub(&range.start)
}

pub fn translate<T: Debug+Rem<Output=T> + Div<Output=T> + Mul<Output=T> + Add<Output=T> + Copy + Zero + One + Ord + Sub<Output=T> + std::cmp::PartialOrd, const DIM: usize>(range: &Range<[T; DIM]>, element: &[T; DIM]) -> Option<T> {
    if contains(range, element) {
        Some(shape(range).idx(&relative_to(range, element)))
    } else {
        None
    }
}

pub fn resolve<T: Add<Output=T> + Copy + One + Zero + PartialOrd>(input_size: T, input_range: impl RangeBounds<T>) -> Range<T> {
    let b = match input_range.start_bound() {
        Bound::Included(&x) => x,
        Bound::Excluded(&x) => x + T::one(),
        Bound::Unbounded => T::zero()
    };
    let e = match input_range.end_bound() {
        Bound::Included(&x) => x + T::one(),
        Bound::Excluded(&x) => x,
        Bound::Unbounded => input_size
    };
    debug_assert!(b <= e, "Input range starts later than it ends");
    debug_assert!(e <= input_size, "Input range exceeds input size");
    b..e
}