use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Less};

pub fn top_small_k_indices<V: Copy + PartialOrd>(mut k: usize, n: usize, f: impl Fn(usize) -> V) -> Vec<(usize, V)> {
    debug_assert!(k <= n);
    let mut heap: Vec<(usize, V)> = (0..k).map(&f).enumerate().collect();
    heap.sort_by(|v1, v2| if v1.1 > v2.1 { Greater } else { Less });
    for (idx, v) in (k..n).map(f).enumerate() {
        let idx = idx + k;
        if v > heap[0].1 {
            let mut i = 1;
            while i < k && v > heap[i].1 {
                heap[i - 1] = heap[i];
                i += 1
            }
            heap[i - 1] = (idx, v);
        }
    }
    heap
}

pub fn top_large_k_indices<T>(mut k: usize, values: &[T], candidates_per_value: &mut [usize], f: fn(&T) -> usize, mut output: impl FnMut(usize)) {
    debug_assert!(candidates_per_value.iter().all(|&e| e == 0));
    values.iter().for_each(|v| candidates_per_value[f(v)] += 1);
    let mut min_candidate_value = 0;
    for (value, candidates) in candidates_per_value.iter_mut().enumerate().rev() {
        if k <= *candidates {
            *candidates = k;
            min_candidate_value = value;
            break;
        }
        k -= *candidates;
    }
    candidates_per_value[0..min_candidate_value].fill(0);
    for (i, v) in values.iter().enumerate() {
        let v = f(v);
        if candidates_per_value[v] > 0 {
            output(i);
            candidates_per_value[v] -= 1;
        }
    }
}


#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::Rng;
    use super::*;

    #[test]
    fn test8() {
        let mut rng = rand::thread_rng();
        let max = 128usize;
        for _ in 0..54 {
            let k = rng.gen_range(2usize..8);
            let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
            let mut candidates = vec![0; max];
            let mut o = Vec::new();
            top_large_k_indices(k, &arr, &mut candidates, |&a| a, |t| o.push(t));
            let mut top_values1: Vec<usize> = o.iter().map(|&i| arr[i]).collect();
            let mut arr_ind: Vec<(usize, usize)> = arr.into_iter().enumerate().collect();
            arr_ind.sort_by_key(|&(_, v)| v);
            let top_values2: Vec<usize> = arr_ind[64 - k..].iter().map(|&(_, v)| v).collect();
            top_values1.sort();
            assert_eq!(top_values1, top_values2)
        }
    }

    #[test]
    fn test9() {
        let mut rng = rand::thread_rng();
        let max = 128usize;
        for _ in 0..54 {
            let k = rng.gen_range(2usize..8);
            let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
            let o = top_small_k_indices(k, arr.len(), |i| arr[i]);
            let mut top_values1: Vec<usize> = o.into_iter().map(|(i, v)| v).collect();
            let mut arr_ind: Vec<(usize, usize)> = arr.into_iter().enumerate().collect();
            arr_ind.sort_by_key(|&(_, v)| v);
            let top_values2: Vec<usize> = arr_ind[64 - k..].iter().map(|&(_, v)| v).collect();
            top_values1.sort();
            assert_eq!(top_values1, top_values2)
        }
    }

    #[test]
    fn test10() {
        let mut rng = rand::thread_rng();
        let max = 128usize;
        for _ in 0..54 {
            let arr: Vec<usize> = (0..64).map(|_| rng.gen_range(0..max)).collect();
            let o = top_small_k_indices(1, arr.len(), |i| arr[i]);
            let (top_idx, top_val) = o[0];
            assert_eq!(top_val, *arr.iter().max().unwrap());
            assert_eq!(top_idx, arr.len() - 1 - arr.iter().rev().position_max().unwrap());
        }
    }
}
pub fn argsort<T>(data: &[T], compare:impl Fn(&T,&T)->Ordering) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&i,&j| compare(&data[i],&data[j]));
    indices
}