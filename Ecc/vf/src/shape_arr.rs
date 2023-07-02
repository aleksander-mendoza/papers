
pub fn shape1<T, const DIM0: usize>(_mat: &[T; DIM0]) -> [usize; 1] {
    [DIM0]
}

pub fn shape2<T, const DIM0: usize, const DIM1: usize>(_mat: &[[T; DIM0]; DIM1]) -> [usize; 2] {
    [DIM1, DIM0]
}

pub fn shape3<T, const DIM0: usize, const DIM1: usize, const DIM2: usize>(_mat: &[[[T; DIM0]; DIM1]; DIM2]) -> [usize; 3] {
    [DIM2, DIM1, DIM0]
}

pub fn shape4<T, const DIM0: usize, const DIM1: usize, const DIM2: usize, const DIM3: usize>(_mat: &[[[[T; DIM0]; DIM1]; DIM2]; DIM3]) -> [usize; 4] {
    [DIM3, DIM2, DIM1, DIM0]
}

pub fn shape5<T, const DIM0: usize, const DIM1: usize, const DIM2: usize, const DIM3: usize, const DIM4: usize>(_mat: &[[[[[T; DIM0]; DIM1]; DIM2]; DIM3]; DIM4]) -> [usize; 5] {
    [DIM4, DIM3, DIM2, DIM1, DIM0]
}

pub fn col_vec<T:Sized, const DIM0: usize>(mat: [T; DIM0]) -> [[T; 1]; DIM0] {
    mat.map(|x| [x])
}

pub fn row_vec<T:Sized, const DIM0: usize>(mat: [T; DIM0]) -> [[T; DIM0]; 1] {
    [mat]
}

pub fn unsqueeze2_1<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; 1]; DIM0] {
    col_vec(mat)
}

pub fn unsqueeze2_0<T, const DIM0: usize>(mat: [T; DIM0]) -> [[T; DIM0]; 1] {
    row_vec(mat)
}

pub fn squeeze2_1<T, const DIM1: usize>(mat: [[T; 1]; DIM1]) -> [T; DIM1] {
    mat.map(|[x]| x)
}

pub fn squeeze2_0<T, const DIM0: usize>(mat: [[T; DIM0]; 1]) -> [T; DIM0] {
    let [x] = mat;
    x
}

pub fn unsqueeze3_0<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; DIM0]; DIM1]; 1] {
    [mat]
}

pub fn unsqueeze3_1<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; DIM0]; 1]; DIM1] {
    unsqueeze2_1(mat)
}

pub fn unsqueeze3_2<T, const DIM0: usize, const DIM1: usize>(mat: [[T; DIM0]; DIM1]) -> [[[T; 1]; DIM0]; DIM1] {
    mat.map(|x| unsqueeze2_1(x))
}


pub fn squeeze3_0<T, const DIM0: usize, const DIM1: usize>(mat: [[[T; DIM0]; DIM1]; 1]) -> [[T; DIM0]; DIM1] {
    let [x] = mat;
    x
}

pub fn squeeze3_1<T, const DIM0: usize, const DIM2: usize>(mat: [[[T; DIM0]; 1]; DIM2]) -> [[T; DIM0]; DIM2] {
    squeeze2_1(mat)
}

pub fn squeeze3_2<T, const DIM2: usize, const DIM1: usize>(mat: [[[T; 1]; DIM1]; DIM2]) -> [[T; DIM1]; DIM2] {
    mat.map(|x| squeeze2_1(x))
}


#[cfg(test)]
mod tests {
    use crate::init::arange3;
    use crate::shape::Shape;
    use super::*;

    #[test]
    fn test3() {
        let mat = arange3::<usize,5,3,4>();
        let shape = shape3(&mat);
        let mut idx = 0;
        for i in 0..4{
            for j in 0..3{
                for k in 0..5{
                    let pos = [i,j,k];
                    let idx2 = shape.idx(&pos);
                    assert_eq!(idx, idx2);
                    let p = mat.as_ptr() as *const usize;
                    let e = unsafe{p.add(idx).read()};
                    assert_eq!(idx, e);
                    assert!(pos.lt(&shape));
                    idx += 1;
                }
            }
        }
    }
}