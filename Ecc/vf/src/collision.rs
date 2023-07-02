use std::ops::Mul;
use num_traits::{Float, MulAdd};
use crate::{AffTrans, VectorFieldSub};
use crate::dot_arr::dot0;

pub fn line_segment_to_line_segment() {

}

pub fn cylinder_to_cylinder() {}


/**Vector holding length of x,y,z radii.*/
pub type Ellipsoid = [f32; 3];

pub fn ellipsoid_to_plane(ell1: Ellipsoid, tran1: AffTrans<f32, 3>, ell2: Ellipsoid, tran2: AffTrans<f32, 3>) {}

/** https://matthias-research.github.io/pages/publications/orientedParticles.pdf */
pub fn ellipsoid_to_ellipsoid(ell1: Ellipsoid, tran1: AffTrans<f32, 3>, ell2: Ellipsoid, tran2: AffTrans<f32, 3>) {}