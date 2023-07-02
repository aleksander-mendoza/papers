use num_traits::{Float, NumAssign};
use crate::Dist;

/**This is really same thing as Bezier curve.*/
pub type PiecewiseLinear<F: Float, const DIM: usize> = [[F; DIM]];

///
/// Returns the length of the piecewise-linear curve. Notice that you can plug bezier curve here as well
/// and obtain length of the line spanned by its control points (this is called Piecewise linear approximation of Bézier curves)
///
pub fn curve_length<F: NumAssign + Float + Copy, const DIM: usize>(piecewise_linear: &PiecewiseLinear<F, DIM>) -> F where for<'a> &'a [F;DIM]: Dist<Output=F> {
    let mut prev = &piecewise_linear[0];
    let mut sum = F::zero();
    for next in &piecewise_linear[1..] {
        sum += prev.dist(next);
        prev = next;
    }
    sum
}