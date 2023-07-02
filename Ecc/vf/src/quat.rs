use num_traits::Float;
use crate::{l2, mat3_to_mat4, VectorFieldMul, VectorFieldNeg, xyz4_};

/**Quaternion*/
pub type Quat<F: Float> = [F; 4];

/**Quaternion conjugate*/
pub fn conjugate<F: Float + Copy>(q: &Quat<F>) -> Quat<F> {
    let &[x, y, z, w] = q;
    [-x, -y, -z, w]
}

/**Quaternion conjugate*/
pub fn conjugate_<F: Float>(q: &mut Quat<F>) -> &mut Quat<F> {
    xyz4_(q).neg_();
    q
}

/**Quaternion conjugate*/
pub fn _conjugate<F: Float>(mut q: Quat<F>) -> Quat<F> {
    conjugate_(&mut q);
    q
}

/**Quaternion multiplication*/
pub fn mul<F: Float + Copy>(q1: &Quat<F>, q2: &Quat<F>) -> Quat<F> {
    [
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
        -q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3] + q1[3] * q2[2],
        -q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] + q1[3] * q2[3],
    ]
}

/**Quaternion to rotation matrix*/
pub fn mat3<F: Float + Copy>(q: &Quat<F>) -> [[F; 3]; 3] {
    let o = F::one();
    let t = o + o;
    let &[qx, qy, qz, qw] = q;
    [
        [o - t * qy * qy - t * qz * qz, t * qx * qy - t * qz * qw, t * qx * qz + t * qy * qw],
        [t * qx * qy + t * qz * qw, o - t * qx * qx - t * qz * qz, t * qy * qz - t * qx * qw],
        [t * qx * qz - t * qy * qw, t * qy * qz + t * qx * qw, o - t * qx * qx - t * qy * qy],
    ]
}

/**Quaternion to rotation matrix*/
pub fn mat4<F: Float + Copy>(q: &Quat<F>) -> [[F; 4]; 4] {
    mat3_to_mat4(mat3(q))
}

/** Rotates about a specific axis. Assumes the axis vector is already normalised */
pub fn about<F: Float>(axis: [F; 3], angle: F) -> Quat<F> {
    let o = F::one();
    let t = o + o;
    let a = angle / t;
    let s = a.sin();
    let [x, y, z] = axis;
    [
        x * s,
        y * s,
        z * s,
        a.cos(),
    ]
}

/** Rotates about x axis. Assumes the axis vector is already normalised */
pub fn about_x<F: Float>(angle: F) -> Quat<F> {
    about([F::one(), F::zero(), F::zero()], angle)
}

/** Rotates about y axis. Assumes the axis vector is already normalised */
pub fn about_y<F: Float>(angle: F) -> Quat<F> {
    about([F::zero(), F::one(), F::zero()], angle)
}

/** Rotates about z axis. Assumes the axis vector is already normalised */
pub fn about_z<F: Float>(angle: F) -> Quat<F> {
    about([F::zero(), F::zero(), F::one()], angle)
}