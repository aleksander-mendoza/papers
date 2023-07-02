use std::ops::{Add, AddAssign, MulAssign, Neg, Sub};
use std::process::Output;
use num_traits::{Float, FloatConst, Num};
use crate::mat_arr::{mat2_add_column, mat3_add_column, mat3x2_add_row, mat4x3_add_row, mul_row_wise_};
use crate::{mat3_to_mat4, VectorFieldAddAssign, VectorFieldMulAddAssign, VectorFieldMulAdd, xyz4, xyz4_};

pub type Translation<S, const DIM: usize> = [S; DIM];
pub type Scaling<S, const DIM: usize> = [S; DIM];
/**When DIM=2, it's an array [[S;2];1] holding one normal vector of length 2, which determines where the X axis lies.
The Y axis can then be obtained by rotating X axis 90 degrees clockwise.
 When DIM=3 it's an array [[S;3];2] holding two normal vector of length 2, which determine where the X and Y axes lie.
The Z axis can then be obtained by taking cross product of X and Y.
 And so on for DIM>3*/
pub type AlignmentAxis<S, const DIM: usize> = [[S; DIM]; { DIM - 1 }];
pub type EulerRotation<S, const DIM: usize> = [S; DIM];
pub type Quaternion<S> = [S; 4];

/**Trait that encapsulates to_radians() and to_degrees() so that we can be agnostic to number of bits in f32 and f64*/
pub trait DegreesRadians: FloatConst {
    /**radians to degrees*/
    fn deg(self) -> Self;
    /**degrees to radians*/
    fn rad(self) -> Self;
}

impl DegreesRadians for f32 {
    fn deg(self) -> Self {
        self.to_degrees()
    }

    fn rad(self) -> Self {
        self.to_radians()
    }
}


/**clockwise rotation*/
pub fn rotate90deg_cw<S: Neg<Output=S>>(vector: [S; 2]) -> [S; 2] {
    let [x, y] = vector;
    [y, -x]
}

/**counter clockwise rotation*/
pub fn rotate90deg_ccw<S: Neg<Output=S>>(vector: [S; 2]) -> [S; 2] {
    let [x, y] = vector;
    [-y, x]
}

/**clockwise rotation in radians*/
pub fn rotate2d(radians: f32, vector: [f32; 2]) -> [f32; 2] {
    let s = radians.sin();
    let c = radians.cos();
    let [x, y] = vector;
    [c * x - s * x, s * x + c * y]
}

/**cross product of two vectors*/
pub fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/**Given a normal vector of X axis, produces a 2x2 rotation matrix R such that  `M·[1,0]^T = X_axis`
 and `M·[0, 1]^T = Y_axis`*/
pub fn rotation2d_from_axis(x_axis: [f32; 2]) -> [[f32; 2]; 2] {
    let y_axis = rotate90deg_cw(x_axis);
    [
        [x_axis[0], y_axis[0]],
        [x_axis[1], y_axis[1]],
    ]
}

/**Given a normal vector of X axis, produces a 2x2 rotation matrix R such that  `M·[1,0]^T = X_axis`
 and `M·[0, 1]^T = Y_axis`*/
pub fn rotation2d(angle_in_radians: f32) -> [[f32; 2]; 2] {
    let s = angle_in_radians.sin();
    let c = angle_in_radians.cos();
    [
        [c, -s],
        [s, c]
    ]
}

/**Given a 2 normal vectors, one of X axis and other of Y , produces a 3x3 rotation matrix R such that ` M·[1,0,0]^T = X_axis`
 , `M·[0, 1, 0]^T = Y_axis` and `M·[0, 0, 1]^T = Z_axis`*/
pub fn rotation3d_from_axis(xy_axes: [[f32; 3]; 2]) -> [[f32; 3]; 3] {
    let [x_axis, y_axis] = xy_axes;
    let z_axis = cross(&x_axis, &y_axis);
    [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]
}

/**Given an angle, produces a 3x3 rotation matrix R such that ` M·[1,0,0]^T = X_axis`
 , `M·[0, 1, 0]^T = Y_axis` and `M·[0, 0, 1]^T = Z_axis`*/
pub fn rotation3d_about_x(angle_in_radians: f32) -> [[f32; 3]; 3] {
    // https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
    let s = angle_in_radians.sin();
    let c = angle_in_radians.cos();
    [
        [1., 0., 0.],
        [0., c, -s],
        [0., s, c],
    ]
}

/**Given an angle, produces a 3x3 rotation matrix R such that ` M·[1,0,0]^T = X_axis`
 , `M·[0, 1, 0]^T = Y_axis` and `M·[0, 0, 1]^T = Z_axis`*/
pub fn rotation3d_about_y(angle_in_radians: f32) -> [[f32; 3]; 3] {
    // https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
    let s = angle_in_radians.sin();
    let c = angle_in_radians.cos();
    [
        [c, 0., s],
        [0., 1., 0.],
        [-s, 0., c],
    ]
}

/**Given an angle, produces a 3x3 rotation matrix R such that ` M·[1,0,0]^T = X_axis`
 , `M·[0, 1, 0]^T = Y_axis` and `M·[0, 0, 1]^T = Z_axis`*/
pub fn rotation3d_about_z(angle_in_radians: f32) -> [[f32; 3]; 3] {
    // https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
    let s = angle_in_radians.sin();
    let c = angle_in_radians.cos();
    [
        [c, -s, 0.],
        [s, c, 0.],
        [0., 0., 1.],
    ]
}

/**Given an angle and an axis normal vector, produces a 3x3 rotation matrix R such that ` M·[1,0,0]^T = X_axis`
 , `M·[0, 1, 0]^T = Y_axis` and `M·[0, 0, 1]^T = Z_axis`*/
pub fn rotation3d_about(angle_in_radians: f32, axis: &[f32; 3]) -> [[f32; 3]; 3] {
    let s = angle_in_radians.sin();
    let c = angle_in_radians.cos();
    let t = 1. - c;
    let [x, y, z] = axis;
    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
    [
        [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
        [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
        [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
    ]
}



/**Given a 2 normal vectors, one of X axis and other of Y , produces a 3x3 rotation matrix R such that ` M·[1,0,0,1]^T = X`
 , `M·[0, 1, 0,1]^T = Y` and `M·[0, 0, 1,1]^T = Z` where Z is a cross product of X and Y*/
pub fn rotation4d_from_axis(xy_axes: [[f32; 3]; 2]) -> [[f32; 4]; 4] {
    mat3_to_mat4(rotation3d_from_axis(xy_axes))
}


/**Given an angle, produces a 3x3 rotation matrix R such that ` M·[1,0,0,1]^T = X_axis`
 , `M·[0, 1, 0,1]^T = Y_axis` and `M·[0, 0, 1,1]^T = Z_axis`*/
pub fn rotation4d_about_x(angle_in_radians: f32) -> [[f32; 4]; 4] {
    mat3_to_mat4(rotation3d_about_x(angle_in_radians))
}


/**Given an angle, produces a 3x3 rotation matrix R such that ` M·[1,0,0,1]^T = X_axis`
 , `M·[0, 1, 0,1]^T = Y_axis` and `M·[0, 0, 1,1]^T = Z_axis`*/
pub fn rotation4d_about_y(angle_in_radians: f32) -> [[f32; 4]; 4] {
    mat3_to_mat4(rotation3d_about_y(angle_in_radians))
}


/**Given an angle, produces a 3x3 rotation matrix R such that ` M·[1,0,0,1]^T = X_axis`
 , `M·[0, 1, 0,1]^T = Y_axis` and `M·[0, 0, 1,1]^T = Z_axis`*/
pub fn rotation4d_about_z(angle_in_radians: f32) -> [[f32; 4]; 4] {
    mat3_to_mat4(rotation3d_about_z(angle_in_radians))
}

/**Given an angle and an axis normal vector, produces a 3x3 rotation matrix R such that ` M·[1,0,0, 1]^T = X_axis`
 , `M·[0, 1, 0, 1]^T = Y_axis` and `M·[0, 0, 1, 1]^T = Z_axis`*/
pub fn rotation4d_about(angle_in_radians: f32, axis: &[f32; 3]) -> [[f32; 4]; 4] {
    mat3_to_mat4(rotation3d_about(angle_in_radians,axis))
}


/**Operates on 4x4 matrix `M` that represents affine transformation. It is assumed that vectors are
 applied from right `M·v` and translation should be applied to the result `M·v+translation_vector`*/
pub fn translate4d_(mat: &mut [[f32;4]; 4],translation_vector:&[f32;3]){
    let [r0,r1,r2,r3] = mat;
    let &[x,y,z] = translation_vector;
    xyz4_(r0).add_mul_scalar_(xyz4(r3),x);
    xyz4_(r1).add_mul_scalar_(xyz4(r3),y);
    xyz4_(r2).add_mul_scalar_(xyz4(r3),z);
    let q = r3[3];
    r0[3]+=x*q;
    r1[3]+=y*q;
    r2[3]+=z*q;
}


/**Produces 4x4 matrix that represents affine transformation*/
pub fn translation4d_(vector:[f32;3])->[[f32;4]; 4]{
    let [x,y,z] = vector;
    [
        [1.,0.,0.,x],
        [0.,1.,0.,y],
        [0.,0.,1.,z],
        [0.,0.,0.,1.],
    ]
}
/**Produces 4x4 matrix that represents affine transformation. Column major version of translation4d_*/
pub fn translation4d_cm_(vector:[f32;3])->[[f32;4]; 4]{
    let [x,y,z] = vector;
    [
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [x,y,z,1.],
    ]
}
/**FOV angle is in radians*/
pub fn perspective_proj<F: Float + Copy>(fov: F, aspect: F, near: F, far: F) -> [[F; 4]; 4] {
    let o = F::one();
    let two = o+o;
    let tan_half_fov_y = o / (fov / two).tan();
    let nearmfar = near - far;
    let z = F::zero();
    [
        [o / (aspect*tan_half_fov_y), z, z, z],
        [z, o / tan_half_fov_y, z, z],
        [z, z, (far + near) / nearmfar, two * far * near / nearmfar],
        [z, z, -o, z]
    ]
}
/**FOV angle is in radians. Column major version of perspective_proj*/
pub fn perspective_proj_cm<F: Float + Copy>(fov: F, aspect: F, near: F, far: F) -> [[F; 4]; 4] {
    let o = F::one();
    let two = o+o;
    let tan_half_fov_y = o / (fov / two).tan();
    let nearmfar = near - far;
    let z = F::zero();
    [
        [o / (aspect*tan_half_fov_y), z, z, z],
        [z, o / tan_half_fov_y, z, z],
        [z, z, (far + near) / nearmfar, -o],
        [z, z, two * far * near / nearmfar, z]
    ]
}

pub fn orthographic_proj<F: Float + Copy>(bottom: F, top: F, left: F, right: F, near: F, far: F) -> [[F; 4]; 4] {
    let o = F::one();
    let two = o + o;
    let rl = right - left;
    let tb = top - bottom;
    let nf = far - near;
    let z = F::zero();
    [
        [two / rl, z, z, -(right + left) / rl],
        [z, two / tb, z, -(top + bottom) / tb],
        [z, z, -two / nf, -(far + near) / nf],
        [z, z, z, o],
    ]
}
/**Column major version of orthographic_proj*/
pub fn orthographic_proj_cm<F: Float + Copy>(bottom: F, top: F, left: F, right: F, near: F, far: F) -> [[F; 4]; 4] {
    let o = F::one();
    let two = o + o;
    let rl = right - left;
    let tb = top - bottom;
    let nf = far - near;
    let z = F::zero();
    [
        [two / rl, z, z, z],
        [z, two / tb, z, z],
        [z, z, -two / nf, z],
        [-(right + left) / rl, -(top + bottom) / tb, -(far + near) / nf, o],
    ]
}

pub trait AffineTransformation<S, const DIM: usize>: Clone {
    fn compose_(&mut self, other: &Self) -> &mut Self;
    fn compose(&self, other: &Self) -> Self {
        let mut s = self.clone();
        s.compose_(other);
        s
    }
    fn inverse_(&mut self) -> &mut Self;
    fn inverse(&self) -> Self {
        let mut s = self.clone();
        s.inverse_();
        s
    }
    fn scale_(&mut self, scaling: &Scaling<S, DIM>) -> &mut Self;
    fn scale(&self, scaling: &Scaling<S, DIM>) -> Self {
        let mut s = self.clone();
        s.scale_(scaling);
        s
    }
    fn rotate_(&mut self, rot: &EulerRotation<S, DIM>) -> &mut Self;
    fn rotate(&self, rot: &EulerRotation<S, DIM>) -> Self {
        let mut s = self.clone();
        s.rotate_(rot);
        s
    }
    fn translate_(&mut self, translation: &Translation<S, DIM>) -> &mut Self;
    fn translate(&self, translation: &Translation<S, DIM>) -> Self {
        let mut s = self.clone();
        s.translate_(translation);
        s
    }
}

/**A linear transformation resulting from first applying scaling, then rotation to align with axis, then translation*/
#[derive(Clone, Debug)]
pub struct AffTrans<S, const DIM: usize> where [(); { DIM - 1 }]: Sized {
    /**This holds both scaling and rotation information. Each vector's direction represents rotation axis,
                                         and it's length represents scaling. The last axis must be perpendicular to all other, which means
                                         that there are only DIM - 1 degrees of freedom. Therefore we do not store the last vector here.
                                         It can be computer as needed. Once computed, we obtain an orthogonal axis.*/
    axis: AlignmentAxis<S, DIM>,
    translation: Translation<S, DIM>,
}

impl<S: Copy + MulAssign + AddAssign, const DIM: usize> AffineTransformation<S, DIM> for AffTrans<S, DIM> where [(); { DIM - 1 }]: Sized {
    fn compose_(&mut self, other: &Self) -> &mut Self {
        todo!()
    }

    fn inverse_(&mut self) -> &mut Self {
        todo!()
    }

    fn inverse(&self) -> Self {
        todo!()
    }

    fn scale_(&mut self, scaling: &Scaling<S, DIM>) -> &mut Self {
        mul_row_wise_(&mut self.axis, scaling);
        self
    }
    fn rotate_(&mut self, rot: &EulerRotation<S, DIM>) -> &mut Self {
        // self.axis
        self
    }

    fn translate_(&mut self, translation: &Translation<S, DIM>) -> &mut Self {
        self.translation.add_(translation);
        self
    }
}

impl AffTrans<f32, 2> {
    pub fn rot2(&self) -> [[f32; 2]; 2] {
        rotation2d_from_axis(self.axis[0])
    }
    pub fn mat3(&self) -> [[f32; 3]; 3] {
        let mut mat = self.rot2();
        let mat = mat2_add_column(mat, self.translation);
        mat3x2_add_row(mat, [0., 0., 1.])
    }
}

impl AffTrans<f32, 3> {
    pub fn rot3(&self) -> [[f32; 3]; 3] {
        rotation3d_from_axis(self.axis)
    }
    pub fn mat4(&self) -> [[f32; 4]; 4] {
        let mut mat = self.rot3();
        let mat = mat3_add_column(mat, self.translation);
        mat4x3_add_row(mat, [0., 0., 0., 1.])
    }
}