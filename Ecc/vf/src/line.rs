use std::ops::Mul;
use num_traits::{Float, MulAdd};
use crate::{dot_mad_arr::dot0, VectorFieldMulAdd, VectorFieldSub};


/**Pair [P, U] defines line gives by equation `P+s*U` where s is scalar and P,U are points*/
pub type Line<F, const DIM: usize> = [[F; DIM]; 2];

/**Finds two points on lines a and b such that the distance between them is minimized. Those points must lie on a third line that is
 * perpendicular to both of the lines a and b. Returns pair `[s,t]` such that `line::pos(a,s)` and `line::pos(b,t)` are the coordinates of the two
 closest points. If both lines are parallel, then the third perpendicular line is drawn at `t==0`*/
pub fn closest_points<F: Float + MulAdd<Output=F>, const DIM: usize>(a: &Line<F, DIM>, b: &Line<F, DIM>) -> [F; 2] {
    let qp = b[0].sub(&a[0]);
    let u = &a[1];
    let v = &b[1];
    if u.eq(v) {
        // similar to https://math.stackexchange.com/questions/1347604/find-3d-distance-between-two-parallel-lines-in-simple-way
        // but we ue cos(theta) to compute s, while t=0 is assumed , thus is becomes just
        // https://en.wikipedia.org/wiki/Vector_projection#Scalar_projection_2
        let s = dot0(u, &qp);
        let t = F::zero();
        [s, t]
    } else {
        // https://math.stackexchange.com/questions/1033419/line-perpendicular-to-two-other-lines-data-sufficiency
        let uu = dot0(u,u);
        let vv = dot0(v,v);
        let uv = dot0(u, v);

        let uqp = dot0(u, &qp);
        let vqp = dot0(v, &qp);
        // s * uu - t * uv =  uqp
        // s * uv - t * vv =  vqp
        //
        // - s * uv + t * uv^2/uu =  - uqp * uv / uu
        // s * uv - t * vv =  vqp
        //
        // t * uv^2/uu - t * vv =  - uqp * uv / uu + vqp
        //
        // t * (uv^2/uu - vv) =  - uqp * uv / uu + vqp
        //
        // t  =  - (uqp * uv / uu + vqp) / (uv^2/uu - vv)
        //
        // t  =  - (uqp * uv + vqp * uu) / (uv^2 - vv * uu)
        //

        let t = -(uqp * uv + vqp * uu) / (uv * uv - vv * uu);
        // s * uu - t * uv =  uqp
        //s = (uqp + t * uv) / uu
        let s = (uqp + t * uv) / uu;
        [s, t]
    }
}

pub fn pos<F:Float+Copy+MulAdd<Output=F>,const DIM: usize>(a: &Line<F, DIM>, t:F) -> [F; DIM] {
    a[1].mul_scalar_add(t,&a[0])
}