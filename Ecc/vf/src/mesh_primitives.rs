pub mod uv {
    use crate::init::InitEmptyWithCapacity;

    /**number of rings is the longitude resolution (number of latitudal lines), number of sectors is the latitude resolution (number of longitudal lines)*/
    pub fn sphere(rings: usize, sectors: usize) -> Vec<[f32; 2]> {
        let mut texcoords = Vec::with_capacity(rings * sectors);
        for r in (0..rings).map(|r| r as f32) {
            for s in (0..sectors).map(|s| s as f32) {
                let uv = [1. - (s / sectors as f32), 1. - (r / rings as f32)];
                texcoords.push(uv);
            }
        }
        texcoords
    }
}

pub mod norm {
    use std::f32::consts::PI;
    use crate::init::InitEmptyWithCapacity;

    /**number of rings is the longitude resolution (number of latitudal lines), number of sectors is the latitude resolution (number of longitudal lines)*/
    pub fn sphere(rings: usize, sectors: usize) -> Vec<[f32; 3]> {
        let mut normals = Vec::with_capacity(rings * sectors);
        for r in (0..rings).map(|r| r as f32) {
            let theta = r * PI / rings as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            for s in (0..sectors).map(|s| s as f32) {
                let phi = s * 2. * PI / sectors as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let normal = [cos_phi * sin_theta, cos_theta, sin_phi * sin_theta];
                normals.push(normal)
            }
        }
        normals
    }
}

/**Functions that return vertex data. In order to make sense of those vertices (VBO), you also need
 indices ()*/
pub mod vert {
    use std::f32::consts::{FRAC_PI_2, PI};
    use crate::init::InitEmptyWithCapacity;
    use crate::VectorFieldMulAssign;

    pub fn cube(size: [f32; 3]) -> [[f32; 3]; 8] {
        let [x, y, z] = size;
        [
            [-x, -y, z], //0
            [x, -y, z], //1
            [-x, y, z], //2
            [x, y, z], //3
            [-x, -y, -z], //4
            [x, -y, -z], //5
            [-x, y, -z], //6
            [x, y, -z],  //7
        ]
    }

    /**number of rings is the longitude resolution (number of latitudal lines), number of sectors is the latitude resolution (number of longitudal lines)*/
    pub fn sphere(radius: f32, rings: usize, sectors: usize) -> Vec<[f32; 3]> {
        let mut vertices = Vec::with_capacity(rings * sectors);
        for r in (0..rings).map(|r| r as f32) {
            let theta = r * PI / rings as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            for s in (0..sectors).map(|s| s as f32) {
                let phi = s * 2. * PI / sectors as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let mut normal = [cos_phi * sin_theta, cos_theta, sin_phi * sin_theta];
                normal.mul_scalar_(radius);
                vertices.push(normal);
            }
        }
        vertices
    }
}

/**Functions that return indices*/
pub mod ind {
    use std::iter::Step;
    use num_traits::{AsPrimitive, Num, PrimInt};
    use crate::init::InitEmptyWithCapacity;

    pub const CUBE: [u32; 36] = [
        //Top
        2, 6, 7,
        2, 3, 7,

        //Bottom
        0, 4, 5,
        0, 1, 5,

        //Left
        0, 2, 6,
        0, 4, 6,

        //Right
        1, 3, 7,
        1, 5, 7,

        //Front
        0, 2, 3,
        0, 1, 3,

        //Back
        4, 6, 7,
        4, 5, 7
    ];

    /**number of rings is the longitude resolution, number of sectors is the latitude resolution*/
    pub fn sphere<T: PrimInt + Step + Copy + AsPrimitive<usize>>(rings: T, sectors: T) -> Vec<T> {
        let mut indices = Vec::with_capacity((rings * sectors).as_() * 6);
        for r in T::zero()..rings {
            for s in T::zero()..sectors {
                let first = (r * (rings + T::one())) + s;
                let second = first + sectors + T::one();

                indices.push(first);
                indices.push(second);
                indices.push(first + T::one());

                indices.push(second);
                indices.push(second + T::one());
                indices.push(first + T::one());
            }
        }
        indices
    }
}