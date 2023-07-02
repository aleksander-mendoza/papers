use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::process::Output;
use std::simd;
use std::simd::{f32x16, Simd, SimdElement};
use itertools::Itertools;
use num_traits::{AsPrimitive, Float, MulAdd, MulAddAssign, NumAssign, One, Zero};
use crate::init::InitRFoldWithCapacity;
use crate::{Dist, piecewise_linear, tri_len, VectorField, VectorFieldMulOwned, VectorFieldAdd, VectorFieldMulAssign, VectorFieldAddAssign, VectorFieldAddOwned, VectorFieldInitZero, VectorFieldMul, VectorFieldMulAdd, VectorFieldSub};

pub type Bezier<F: Float, const DIM: usize> = [[F; DIM]];

pub fn pos<F: NumAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t: F) -> [F; DIM] where usize: AsPrimitive<F> {
    pos_with(|i|bezier_curve[i],bezier_curve.len(),t)
}
/** Returns position of point that lies at distance t on the curve (where t is normalized to range between 0 and 1).
It uses Bernstein basis polynomials to efficiently compute the position. See here
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#General_definition and here
https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
Returns $\sum_{i=0}^{n-1} \binom{n-1}{i} (1 - t)^{n-1-i} t^i P_i$ where P_i is the point at bezier_curve[i].
 If you can't see the LaTeX equation copy-paste it to https://latex.codecogs.com/eqneditor/editor.php */
pub fn pos_with<F: NumAssign + Float + Copy + 'static, const DIM: usize>(mut bezier_curve: impl FnMut(usize)->[F; DIM], bezier_curve_length:usize, t: F) -> [F; DIM] where usize: AsPrimitive<F> {
    let n = bezier_curve_length;
    /**b_i = (1-t)^{n-1-i}*/
    let b = Vec::init_rfold(
        /*0 \le i < */n,
        /*b_{n-1}=*/F::one(),
        /** b_{i-1} = (1-t)^{n-1-i+1} = (1-t)^{n-1-i}(1-t) = b_i (1-t) */
        |/*b_i=*/b, i| /*b_{i-1=}*/b * (F::one() - t),
    );
    let mut v = [F::zero(); DIM];
    /**c_i = \frac{(n-1)!}{(n-1-i)!} */
    let mut c = 1; // c_0 = 1
    /**i!*/
    let mut i_factorial = 1;
    /**t^i*/
    let mut t_power_i = F::one();
    for i in 0..n {
        /**b_i = (1-t)^{n-1-i}*/
        let b = b[i];
        /**P_i*/
        let p = bezier_curve(i);
        /**\binom{n-1}{i} = \frac{(n-1)!}{i! (n-1-i)!} = \frac{c_i}{i!}*/
        let n_minus_1_choose_i = c / i_factorial;
        /**\binom{n-1}{i} (1 - t)^{n-1-i} t^i */
        let coefficient = n_minus_1_choose_i.as_() * t_power_i * b;
        v.add_(&p.mul_scalar(coefficient));
        t_power_i *= t;
        i_factorial *= i;
        /*c_{i+1} = \frac{(n-1)!}{(n-1-i-1)!} = \frac{(n-1)!(n-1-i)}{(n-1-i-1)!(n-1-i)} = \frac{(n-1)!(n-1-i)}{(n-1-i)!} = c_i (n-1-i)*/
        c *= (n - 1 - i);
    }
    v
}
/**Returns the derivative curve (hodograph). https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html*/
pub fn derivative<F: Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> Vec<[F; DIM]> where usize: AsPrimitive<F> {
    let mut prev = &bezier_curve[0];
    let n:F = bezier_curve.len().as_();
    bezier_curve[1..].iter().map(|next|{
        let diff = next.sub(prev)._mul_scalar(n);
        prev = next;
        diff
    }).collect()
}

/**Evaluates the derivative curve at a specific `t`, without computing the derivative curve explicitly. https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html*/
pub fn tangent<F: NumAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t:F) -> [F; DIM] where usize: AsPrimitive<F> {
    let n:F = bezier_curve.len().as_();
    pos_with(|i| bezier_curve[i+1].sub(&bezier_curve[i])._mul_scalar(n),bezier_curve.len()-1,t)
}


/**de Casteljau's algorithm using dynamic programming. https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html
https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Recursive_definition .

 This function returns the entire triangular table as on [this picture](https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/b-sub-chart.jpg).
Table T where T[i]*/
pub fn de_casteljau<F: NumAssign + MulAdd<Output=F> + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t: F) -> Vec<[F; DIM]> {
    let mut out = Vec::with_capacity(tri_len(bezier_curve.len()));
    let mut prev = &bezier_curve[0];
    let d = F::one() - t;
    for next in &bezier_curve[1..] {
        out.push(prev.linear_comb(d, next, t));
        prev = next;
    }
    let mut from = 1;
    let mut to = out.len();
    while from < to {
        for i in from..to {
            let comb = out[i - 1].linear_comb(d, &out[i], t);
            out.push(comb);
        }
        from = to + 1;
        to = out.len();
    }
    out
}

/**same as de_casteljau but t==0.5,  which allows for some optimisations*/
pub fn de_casteljau_in_half<F: NumAssign + MulAdd<Output=F> + MulAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> Vec<[F; DIM]> {
    let mut out = Vec::with_capacity(tri_len(bezier_curve.len()));
    let mut prev = &bezier_curve[0];
    let half = F::one() / (F::one() + F::one());
    for next in &bezier_curve[1..] {
        out.push(prev.add_mul_scalar(next, half));
        prev = next;
    }
    let mut from = 1;
    let mut to = out.len();
    while from < to {
        for i in from..to {
            let mut comb = out[i - 1].add_mul_scalar(&out[i], half);
            out.push(comb);
        }
        from = to + 1;
        to = out.len();
    }
    out
}

/**Straight-line length between endpoints*/
fn chord_length<F: Float + Copy + NumAssign, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> F where for<'a> &'a [F;DIM]: Dist<Output=F> {
    bezier_curve[0].dist(&bezier_curve[bezier_curve.len() - 1])
}

/** Takes table obtained from de_casteljau or de_casteljau_in_half. Produces two curves that subdivide original one.
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html */
pub fn de_casteljau_table_to_sub_curves<F:Copy, const DIM: usize>(table: Vec<[F;DIM]>, bezier_curve_len:usize) -> (Vec<[F; DIM]>, Vec<[F; DIM]>){
    let mut c1 = Vec::with_capacity(bezier_curve_len);
    let mut c2 = Vec::with_capacity(bezier_curve_len);
    let mut i = 0;
    for step in (0..bezier_curve_len).rev(){
        c1.push(table[i]);
        i += step;
        c2.push(table[i]);
        i += 1;
    }
    (c1,c2)
}
/** Produces two curves that subdivide original one.
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html */
pub fn subdivide<F: NumAssign + MulAdd<Output=F> + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>, t: F) -> (Vec<[F; DIM]>, Vec<[F; DIM]>) {
    de_casteljau_table_to_sub_curves(de_casteljau(bezier_curve, t), bezier_curve.len())
}
/** Produces two curves that subdivide original one.
https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-sub.html */
pub fn subdivide_in_half<F: NumAssign + MulAdd<Output=F> + MulAssign + Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> (Vec<[F; DIM]>, Vec<[F; DIM]>) {
    de_casteljau_table_to_sub_curves(de_casteljau_in_half(bezier_curve), bezier_curve.len())
}
///
/// Computes the length of a section of a bezier curve
///
fn curve_length<F: NumAssign + MulAdd<Output=F>+ Float + Copy + 'static, const DIM: usize>(bezier_curve: &Bezier<F, DIM>) -> F
    where for<'a> &'a [F;DIM]: Dist<Output=F>, usize:AsPrimitive<F>, f32:AsPrimitive<F>
{
    // This algorithm is described in Graphics Gems V IV.7
    //////////////// Setting up constants ////////////////
    let polygon_len:F = piecewise_linear::curve_length(bezier_curve);
    let max_error: F = (polygon_len * 1e-4.as_()).max(1e-8.as_());
    let n:F = bezier_curve.len().as_(); // degree of bezier curve + 1
    let nm2:F = (bezier_curve.len()-2).as_(); // degree of bezier curve - 1
    let two:F = F::one()+F::one();
    let half:F = F::one() / two;
    let coeff_chord:F = two/n;
    let coeff_poly:F = nm2/n;
    //////////////// First iteration works with borrowed slice ////////////////
    let chord_len:F = chord_length(bezier_curve);
    let error:F = polygon_len - chord_len;
    let error_square:F = error*error;
    let mut waiting = Vec::new();
    if error_square < max_error {
        return MulAdd::mul_add(chord_len,coeff_chord, coeff_poly * polygon_len);
    } else {
        // Subdivide the curve (each half has half the error tolerance)
        let (left,right) = subdivide_in_half(bezier_curve);
        let subsection_error = max_error * half;
        waiting.push((left, subsection_error));
        waiting.push((right, subsection_error));
    }
    //////////////// Next iterations works with owned vec ////////////////
    // Algorithm is recursive, but we use a vec as a stack to avoid overflowing (and to make the number of iterations easy to count)

    let mut total_length = F::zero();

    while let Some((section, max_error)) = waiting.pop() {
        // Estimate the error for the length of the curve
        let polygon_len:F = piecewise_linear::curve_length(&section);
        let chord_len:F = chord_length(&section);
        let error:F = polygon_len - chord_len;
        let error_square:F = error*error;

        // If the error is low enough, return the estimated length
        if error_square < max_error {
            total_length += MulAdd::mul_add(chord_len, coeff_chord, coeff_poly * polygon_len);
        } else {
            // Subdivide the curve
            let (left,right) = subdivide_in_half(&section);
            // each half has half the error tolerance
            let subsection_error: F = max_error * half;
            waiting.push((left, subsection_error));
            waiting.push((right, subsection_error));
        }
    }

    total_length
}

