pub fn xy_z_w4<D>(xy: [D; 2], z: D, w: D) -> [D; 4] {
    let [x, y] = xy;
    [x, y, z, w]
}

pub fn xy_zw4<D>(xy: [D; 2], zw: [D; 2]) -> [D; 4] {
    let [x, y] = xy;
    let [z, w] = zw;
    [x, y, z, w]
}

pub fn xyz_w4<D>(xyz: [D; 3], w: D) -> [D; 4] {
    let [x, y, z] = xyz;
    [x, y, z, w]
}

pub fn x_yzw4<D>(x: D, yzw: [D; 3]) -> [D; 4] {
    let [y, z, w] = yzw;
    [x, y, z, w]
}

pub fn xy4<D>(xyzw: &[D; 4]) -> &[D; 2] {
    let [ref xy @ .., _, _ ] = xyzw;
    xy
}

pub fn yz4<D>(xyzw: &[D; 4]) -> &[D; 2] {
    let [_, ref yz @ .., _  ] = xyzw;
    yz
}

pub fn zw4<D>(xyzw: &[D; 4]) -> &[D; 2] {
    let [_, _, ref zw @ ..  ] = xyzw;
    zw
}

pub fn yzw4<D>(xyzw: &[D; 4]) -> &[D; 3] {
    let [_, ref yzw @ ..  ] = xyzw;
    yzw
}


pub fn xyz4<D>(xyzw: &[D; 4]) -> &[D; 3] {
    let [ref xyz @ .., _  ] = xyzw;
    xyz
}

pub fn xyzw3<D>(x: D, y: D, z: D, w: D) -> [D; 4] {
    [x, y, z, w]
}

pub fn x4<D>(xyzw: &[D; 4]) -> &D {
    &xyzw[0]
}

pub fn y4<D>(xyzw: &[D; 4]) -> &D {
    &xyzw[1]
}

pub fn z4<D>(xyzw: &[D; 4]) -> &D {
    &xyzw[2]
}

pub fn w4<D>(xyzw: &[D; 4]) -> &D {
    &xyzw[3]
}

pub fn xy_z3<D>(xy: [D; 2], z: D) -> [D; 3] {
    let [x, y] = xy;
    [x, y, z]
}

pub fn xy3<D>(xyz: &[D; 3]) -> &[D; 2] {
    let [ref xy @ .., _ ] = xyz;
    xy
}

pub fn yz3<D>(xyz: &[D; 3]) -> &[D; 2] {
    let [_, ref yz @ ..  ] = xyz;
    yz
}

pub fn xyz3<D>(x: D, y: D, z: D) -> [D; 3] {
    [x, y, z]
}

pub fn x3<D>(xyz: &[D; 3]) -> &D {
    &xyz[0]
}

pub fn y3<D>(xyz: &[D; 3]) -> &D {
    &xyz[1]
}

pub fn z3<D>(xyz: &[D; 3]) -> &D {
    &xyz[2]
}

pub fn x2<D>(xy: &[D; 2]) -> &D {
    &xy[0]
}

pub fn y2<D>(xy: &[D; 2]) -> &D {
    &xy[1]
}

pub fn xy2<D>(x: D, y: D) -> [D; 2] {
    [x, y]
}


pub fn xy4_<D>(xyzw: &mut [D; 4]) -> &mut [D; 2] {
    let [ref mut xy @ .., _, _ ] = xyzw;
    xy
}

pub fn yz4_<D>(xyzw: &mut [D; 4]) -> &mut [D; 2] {
    let [_, ref mut yz @ .., _  ] = xyzw;
    yz
}

pub fn zw4_<D>(xyzw: &mut [D; 4]) -> &mut [D; 2] {
    let [_, _, ref mut zw @ ..  ] = xyzw;
    zw
}

pub fn yzw4_<D>(xyzw: &mut [D; 4]) -> &mut [D; 3] {
    let [_, ref mut yzw @ ..  ] = xyzw;
    yzw
}


pub fn xyz4_<D>(xyzw: &mut [D; 4]) -> &mut [D; 3] {
    let [ref mut xyz @ .., _  ] = xyzw;
    xyz
}


pub fn x4_<D>(xyzw: &mut [D; 4]) -> &mut D {
    &mut xyzw[0]
}

pub fn y4_<D>(xyzw: &mut [D; 4]) -> &mut D {
    &mut xyzw[1]
}

pub fn z4_<D>(xyzw: &mut [D; 4]) -> &mut D {
    &mut xyzw[2]
}

pub fn w4_<D>(xyzw: &mut [D; 4]) -> &mut D {
    &mut xyzw[3]
}


pub fn xy3_<D>(xyz: &mut [D; 3]) -> &mut [D; 2] {
    let [ref mut xy @ .., _ ] = xyz;
    xy
}

pub fn yz3_<D>(xyz: &mut [D; 3]) -> &mut [D; 2] {
    let [_, ref mut yz @ ..  ] = xyz;
    yz
}

pub fn x3_<D>(xyz: &mut [D; 3]) -> &mut D {
    &mut xyz[0]
}

pub fn y3_<D>(xyz: &mut [D; 3]) -> &mut D {
    &mut xyz[1]
}

pub fn z3_<D>(xyz: &mut [D; 3]) -> &mut D {
    &mut xyz[2]
}

pub fn x2_<D>(xy: &mut [D; 2]) -> &mut D {
    &mut xy[0]
}

pub fn y2_<D>(xy: &mut [D; 2]) -> &mut D {
    &mut xy[1]
}

