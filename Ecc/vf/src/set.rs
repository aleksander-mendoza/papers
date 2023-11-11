use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Step;
use num_traits::{AsPrimitive, NumAssignOps, One, PrimInt, Zero};
use std::ops::{Add, AddAssign, Range, Rem, Sub};
use partial_sort::{partial_sort, PartialSort};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use crate::from_usize::FromUsize;
use crate::init::{InitEmptyWithCapacity, InitFilledCapacity};
use crate::shape::Shape;
use crate::{VectorFieldPartialOrd, VectorFieldSub};

pub trait SetCardinality {
    /**Set cardinality is the number of its member elements*/
    fn card(&self) -> usize;
}

pub trait SetIntersection {
    type O;
    fn intersection(&self, other: &Self) -> Self::O;
}

pub trait SetUnion {
    type O;
    fn union(&self, other: &Self) -> Self::O;
}

pub trait SetSubtract {
    fn subtract(&mut self, other: &Self);
}

pub trait SetOverlap {
    fn overlap(&self, other: &Self) -> usize;
}

pub trait SetContains<N: Copy> {
    fn contains(&self, elem: N) -> bool;
}

pub trait SetSparseIndexArray {
    fn normalize(&mut self) -> &mut Self;
    fn is_normalized(&self) -> bool;
}

pub trait SetSparseMask {
    fn mask<T>(&self, destination: &mut [T], f: impl Fn(&mut T));
}

pub trait SetSparseParallelMask {
    fn mask_par<T>(&self, destination: &mut [T], f: impl Fn(&mut T) + Send + Sync);
}

// pub trait SetSparseRand<T> {
//     fn add_unique_random(&mut self, n: T, range: Range<T>);
// }
impl<N: Ord + Copy> SetCardinality for [N] {
    fn card(&self) -> usize {
        self.len()
    }
}


impl<N: num_traits::PrimInt> SetIntersection for [N] {
    type O = Vec<N>;

    /**Requires that both SDRs are normalized. The resulting SDR is already in normalized form*/
    fn intersection(&self, other: &Self) -> Vec<N> {
        let mut intersection = Vec::with_capacity(self.len() + other.len());
        let mut i = 0;
        if other.is_empty() { return intersection; }
        for &idx in self {
            while other[i] < idx {
                i += 1;
                if i >= other.len() { return intersection; }
            }
            if other[i] == idx {
                intersection.push(idx);
            }
        }
        intersection
    }
}

impl<N: Ord + Copy> SetOverlap for Vec<N> {
    /**This method requires that both sets are normalized*/
    fn overlap(&self, other: &Self) -> usize {
        if self.is_empty() || other.is_empty() { return 0; }
        let mut i1 = 0;
        let mut i2 = 0;
        let mut overlap = 0;
        let (s1, s2) = if self[0] < other[0] { (self, other) } else { (other, self) };
        loop {
            while s1[i1] < s2[i2] {
                i1 += 1;
                if i1 >= s1.len() { return overlap; }
            }
            if s1[i1] == s2[i2] {
                overlap += 1;
                i1 += 1;
                if i1 >= s1.len() { return overlap; }
            }
            while s1[i1] > s2[i2] {
                i2 += 1;
                if i2 >= s2.len() { return overlap; }
            }
            if s1[i1] == s2[i2] {
                overlap += 1;
                i2 += 1;
                if i2 >= s2.len() { return overlap; }
            }
        }
    }
}

impl<N: Ord + Copy> SetUnion for [N] {
    type O = Vec<N>;

    fn union(&self, other: &Self) -> Self::O {
        let mut union = Vec::with_capacity(self.len() + other.len());
        let mut i1 = 0;
        let mut i2 = 0;
        if self.len() > 0 && other.len() > 0 {
            'outer: loop {
                while self[i1] < other[i2] {
                    union.push(self[i1]);
                    i1 += 1;
                    if i1 >= self.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    union.push(self[i1]);
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
                while self[i1] > other[i2] {
                    union.push(other[i2]);
                    i2 += 1;
                    if i2 >= other.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    union.push(other[i2]);
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
            }
        }
        if i1 < self.len() {
            union.extend_from_slice(&self[i1..])
        } else {
            union.extend_from_slice(&other[i2..])
        }
        union
    }
}

impl<N: Ord + Copy> SetSubtract for Vec<N> {
    fn subtract(&mut self, other: &Self) {
        let mut i1 = 0;
        let mut i2 = 0;
        let mut j = 0;
        if self.len() > 0 && other.len() > 0 {
            'outer: loop {
                while self[i1] < other[i2] {
                    self[j] = self[i1];
                    j += 1;
                    i1 += 1;
                    if i1 >= self.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
                while self[i1] > other[i2] {
                    i2 += 1;
                    if i2 >= other.len() { break 'outer; }
                }
                if self[i1] == other[i2] {
                    i1 += 1;
                    i2 += 1;
                    if i1 >= self.len() || i2 >= other.len() { break 'outer; }
                }
            }
        }
        while i1 < self.len() {
            self[j] = self[i1];
            j += 1;
            i1 += 1;
        }
        self.truncate(j);
    }
}

impl<N: AsPrimitive<usize> + Copy> SetSparseMask for [N] {
    fn mask<T>(&self, destination: &mut [T], f: impl Fn(&mut T)) {
        for i in self {
            f(&mut destination[i.as_()])
        }
    }
}

impl<N: AsPrimitive<usize> + Copy> SetSparseIndexArray for Vec<N> {
    fn normalize(&mut self) -> &mut Self {
        self.sort_by_key(|a| a.as_());
        self.dedup_by_key(|a| a.as_());
        self
    }

    fn is_normalized(&self) -> bool {
        self.windows(2).all(|a| a[0].as_() < a[0].as_())
    }
}

pub fn add_unique_random<N: PrimInt+NumAssignOps+AsPrimitive<usize>+Step+Hash>(mut cache:impl FnMut(N)->bool, n: N, range: Range<N>) where Standard: Distribution<N>{
    let len = range.end - range.start;
    assert!(len >= n);
    for _ in N::zero()..n {
        let mut r = range.start + rand::random::<N>() % len;
        while cache(r) { // cache says whether r already occurred before
            r += N::one();
            if r >= range.end {
                r = range.start;
            }
        }
    }
}
pub fn sparse_gt(dense:&[f32], threshold:f32)->Vec<u32>{
    let mut indices = Vec::new();
    for (i,v) in dense.iter().cloned().enumerate(){
        if v > threshold{
            indices.push(i as u32);
        }
    }
    indices.sort_by(|&a,&b|dense[b as usize].total_cmp(&dense[a as usize]));// a and b are reversed to sort in descending order
    indices
}
pub fn sparse_top_k(dense:&[f32], k:u32)->Vec<u32>{
    let mut perm:Vec<u32> = (0..dense.len() as u32).collect();
    perm.partial_sort(k as usize,|&a,&b|dense[b as usize].total_cmp(&dense[a as usize]));// a and b are reversed to sort in descending order
    perm.truncate(k as usize);
    perm
}
pub fn rand_set<N:PrimInt+AsPrimitive<usize>+Step+Hash+NumAssignOps>(cardinality: N, range: Range<N>) -> Vec<N> where Standard: Distribution<N>{
    let mut set = HashSet::new();
    add_unique_random(|n|!set.insert(n),cardinality, range);
    set.into_iter().collect()
}
pub fn rand_set_sorted<N:PrimInt+AsPrimitive<usize>+Step+Hash+NumAssignOps>(cardinality: N, range: Range<N>) -> Vec<N> where Standard: Distribution<N>{
    let mut r = rand_set(cardinality,range);
    r.sort();
    r
}
pub fn rand_dense<N:PrimInt+AsPrimitive<usize>+Step+Hash+NumAssignOps>(cardinality: N, length:N) -> Vec<bool> where Standard: Distribution<N>{
    let mut s = Vec::full(length.as_(),false);
    add_unique_random(|n| {
        let prev = s[n.as_()];
        s[n.as_()] = true;
        prev
    },cardinality, N::zero()..length);
    s
}

//     /**Randomly picks some neurons that a present in other SDR but not in self SDR.
//     Requires that both SDRs are already normalized.
//     It will only add so many elements so that self.len() <= n*/
//     pub fn randomly_extend_from(&mut self, other: &Self, n: usize) {
//         debug_assert!(self.is_normalized());
//         debug_assert!(other.is_normalized());
//         assert!(other.len() <= n, "The limit {} is less than the size of SDR {}", n, other.len());
//         self.subtract(other);
//         while self.len() + other.len() > n {
//             let idx = rand::random::<usize>() % self.0.len();
//             self.0.swap_remove(idx);
//         }
//         self.0.extend_from_slice(other.as_slice());
//         self.0.sort()
//     }
// }
// {
//     pub fn subregion(&self, total_shape: &[Idx; 3], subregion_range: &Range<[Idx; 3]>) -> Vec {
//         Vec(self.iter().cloned().filter_map(|i| range_translate(subregion_range, &total_shape.pos(i))).collect())
//     }
//
//     pub fn subregion2d(&self, total_shape: &[Idx; 3], subregion_range: &Range<[Idx; 2]>) -> Vec {
//         Vec(self.iter().cloned().filter_map(|i| range_translate(subregion_range, total_shape.pos(i).grid())).collect())
//     }
//
//     pub fn conv_rand_subregion(&self, shape: &ConvShape, rng: &mut impl Rng) -> Vec {
//         self.conv_subregion(shape, &shape.out_grid().rand_vec(rng))
//     }
//
//     pub fn conv_subregion(&self, shape: &ConvShape, output_column_position: &[Idx; 2]) -> Vec {
//         let mut s = Vec::new();
//         let r = shape.in_range(output_column_position);
//         let kc = shape.kernel_column();
//         for &i in self.iter() {
//             let pos = shape.in_shape().pos(i);
//             if range_contains(&r, pos.grid()) {
//                 let pos_within_subregion = pos.grid().sub(&r.start).add_channels(pos.channels());
//                 s.push(kc.idx(pos_within_subregion))
//             }
//         }
//         s
//     }
//
// }
pub fn sparse_subtensor<const DIM:usize,Idx: PrimInt+Copy+Debug>(sparse_tensor:&[Idx], tensor_shape:&[Idx;DIM], subtensor:Range<[Idx;DIM]>)->Vec<Idx>{
    sparse_tensor.iter().cloned().filter(|&i|{
        let pos = tensor_shape.pos(i);
        subtensor.start.all_le(&pos) && pos.all_lt(&subtensor.end)
    }).collect()
}
pub fn sparse_subtensor_reindexed<const DIM:usize,Idx: PrimInt+Copy+Debug>(sparse_tensor:&[Idx], tensor_shape:&[Idx;DIM], subtensor:Range<[Idx;DIM]>)->Vec<Idx>{
    let subshape = subtensor.end.sub(&subtensor.start);
    sparse_tensor.iter().cloned().filter_map(|i|{
        let pos = tensor_shape.pos(i);
        if subtensor.start.all_le(&pos) && pos.all_lt(&subtensor.end){
            Some(subshape.idx(&pos.sub(&subtensor.start)))
        }else{
            None
        }
    }).collect()
}
pub fn join_sets<Idx:Copy, I:AsRef<[Idx]>+Send+Sync>(sparse:&[I])->(Vec<Idx>,Vec<usize>){
    let mut offsets = Vec::with_capacity(sparse.len()+1);
    let mut offset = 0;
    offsets.push(offset);
    for s in sparse{
        offset += s.as_ref().len();
        offsets.push(offset);
    }
    let mut indices:Vec<Idx> = Vec::empty(offset);
    let ptr = indices.as_mut_ptr() as usize;
    sparse.into_par_iter().enumerate().for_each(|(i,s)|{
        let offset = offsets[i];
        let ptr = ptr as *mut Idx;
        for (i, val) in s.as_ref().iter().cloned().enumerate(){
            unsafe{ptr.offset((offset+i) as isize).write(val)}
        }
    });
    (indices, offsets)
}
/**Returns a single vector containing indices of all true boolean values.
The second vector contains offsets to the first one. It works just like Vec<Vec<Idx>> but is flattened.
 To get the i-th slice just do `indices[offsets[i]..offsets[i+1]]` . The number of slices is `offsets.len()-1` */
pub fn batch_dense_to_sparse<Idx: FromUsize>(batch_size: usize, bools: &[bool]) -> (Vec<Idx>, Vec<usize>) {
    assert_eq!(bools.len() % batch_size, 0);
    let mut from = 0;
    let mut indices = Vec::<Idx>::new();
    let mut offsets = Vec::<usize>::with_capacity(bools.len() / batch_size);
    offsets.push(indices.len());
    while from < bools.len() {
        let to = from + batch_size;
        dense_to_sparse_(&bools[from..to], &mut indices);
        offsets.push(indices.len());
        from = to;
    }
    (indices, offsets)
}

pub fn dense_to_sparse_<Idx: FromUsize>(bools: &[bool], output: &mut Vec<Idx>) {
    bools.iter().cloned().enumerate().filter(|(_, b)| *b).map(|(i, _)| Idx::from_usize(i)).collect_into(output);
}

pub fn dense_to_sparse<Idx: FromUsize>(bools: &[bool]) -> Vec<Idx> {
    let mut v = Vec::<Idx>::new();
    dense_to_sparse_(bools, &mut v);
    v
}

pub fn sparse_to_dense_<Idx: AsPrimitive<usize>>(indices: &[Idx], output: &mut [bool]) {
    for i in indices{
        output[i.as_()] = true
    }
}
pub fn sparse_to_dense<Idx: AsPrimitive<usize>>(indices: &[Idx], length:usize) -> Vec<bool> {
    let mut v:Vec<bool> = (0..length).map(|_|false).collect();
    sparse_to_dense_(indices, &mut v);
    v
}

pub fn mat_to_rle<Idx: Zero+AddAssign+One+Copy>(bools: &[bool]) -> Vec<Idx> {
    let mut v = Vec::<Idx>::new();
    let mut prev = false;
    let mut running_length = Idx::zero();
    for &b in bools{
        if b == prev{
            running_length += Idx::one()
        }else{
            v.push(running_length);
            prev = b;
            running_length = Idx::one();
        }
    }
    v.push(running_length);
    v
}

pub fn rle_to_mat<Idx: AsPrimitive<usize>>(rle:&[Idx], bools: &mut [bool]){
    let mut value = false;
    let mut offset = 0;
    for &running_length in rle{
        let end = offset+running_length.as_();
        bools[offset..end].fill(value);
        value = !value;
        offset = end;
    }
    assert_eq!(offset,bools.len(),"RLE counts do not match shape of the boolean tensor")
}

#[cfg(test)]
mod tests {
    use more_asserts::{assert_ge, assert_le};
    use super::*;
    use rand::SeedableRng;
    use crate::init::InitEmptyWithCapacity;
    use crate::init_rand::InitRandWithCapacity;
    use crate::VectorFieldSubAssign;


    #[test]
    fn test6() -> Result<(), String> {
        fn overlap(a: &[u32], b: &[u32]) -> usize {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.overlap(&sdr2)
        }
        assert_eq!(overlap(&[1, 5, 6, 76], &[1]), 1);
        assert_eq!(overlap(&[1, 5, 6, 76], &[]), 0);
        assert_eq!(overlap(&[], &[]), 0);
        assert_eq!(overlap(&[], &[1]), 0);
        assert_eq!(overlap(&[1, 5, 6, 76], &[1, 5, 6, 76]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[5, 76, 6, 1]), 4);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1]), 2);
        assert_eq!(overlap(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]), 3);
        Ok(())
    }

    #[test]
    fn test7() -> Result<(), String> {
        fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.intersection(&sdr2)
        }
        assert_eq!(intersect(&[1, 5, 6, 76], &[1]).as_slice(), &[1]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[]).as_slice(), &[]);
        assert_eq!(intersect(&[], &[1]).as_slice(), &[]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 6]);
        assert_eq!(intersect(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 5, 6]);
        Ok(())
    }

    #[test]
    fn test7_union() -> Result<(), String> {
        fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.union(&sdr2)
        }
        assert_eq!(union(&[1, 5, 6, 76], &[1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[], &[]).as_slice(), &[]);
        assert_eq!(union(&[1], &[]).as_slice(), &[1]);
        assert_eq!(union(&[], &[1]).as_slice(), &[1]);
        assert_eq!(union(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[1, 5, 6, 53, 76, 746]);
        assert_eq!(union(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[1, 3, 5, 6, 7, 53, 76, 78, 746]);
        Ok(())
    }

    #[test]
    fn test7_subtract() -> Result<(), String> {
        fn subtract(a: &[u32], b: &[u32]) -> Vec<u32> {
            let mut sdr1 = a.to_vec();
            let mut sdr2 = b.to_vec();
            sdr1.normalize();
            sdr2.normalize();
            sdr1.subtract(&sdr2);
            sdr1
        }
        assert_eq!(subtract(&[1, 5, 6, 76], &[1]).as_slice(), &[5, 6, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[]).as_slice(), &[1, 5, 6, 76]);
        assert_eq!(subtract(&[], &[]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[]).as_slice(), &[1]);
        assert_eq!(subtract(&[], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[1]).as_slice(), &[]);
        assert_eq!(subtract(&[1], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[1, 2], &[2]).as_slice(), &[1]);
        assert_eq!(subtract(&[2, 3], &[2]).as_slice(), &[3]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[1, 5, 6, 76]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[5, 76, 6, 1]).as_slice(), &[]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1]).as_slice(), &[5, 76]);
        assert_eq!(subtract(&[1, 5, 6, 76], &[53, 746, 6, 1, 5, 78, 3, 6, 7]).as_slice(), &[76]);
        Ok(())
    }

    #[test]
    fn test8() {
        let mat = Vec::<bool>::rand(265);
        let rle:Vec<usize> = mat_to_rle(&mat);
        let mut mat2 = Vec::empty(mat.len());
        rle_to_mat(&rle, &mut mat2);
        assert_eq!(&mat,&mat2);
    }

    #[test]
    fn test9() {
        let i = rand_set(100,0..300);
        let len = 30;
        let o = rand::random::<i32>() % (300-len);
        let mut s1 = sparse_subtensor(&i,&[300], [o]..[o+len]);
        s1.sub_scalar_(o);
        let s2 = sparse_subtensor_reindexed(&i,&[300], [o]..[o+len]);
        assert_eq!(s1,s2);
        for i in s1{
            assert_le!(0,i);
        }
    }
    #[test]
    fn test10() {
        let c:u32 = 1000;
        let r = 20;
        let ii:Vec<Vec<u32>> = (0..r).map(|_|rand_set_sorted(c/10,0..c)).collect();
        let b:Vec<bool> = ii.iter().map(|i|sparse_to_dense(&i,c as usize)).flatten().collect();
        let (i1,o1) = batch_dense_to_sparse::<u32>(c as usize,&b);
        let (i2,o2) = join_sets(&ii);
        assert_eq!(i1,i2,"indices");
        assert_eq!(o1,o2,"offsets");
    }
}