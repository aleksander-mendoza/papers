use std::ops::{Add, Div, Mul, Range, Sub};
use num_traits::{Num, Zero};
use crate::*;
use crate::from_usize::FromUsize;


pub fn interpolate_gaps(histogram: &mut [f32]) {
    assert_eq!(histogram.len() % 256, 0);
    let mut offset = 0;

    while offset < histogram.len() {
        let mut prev_non_zero = 0.;
        let mut i = 0;
        while i < 256 {
            debug_assert!(i < 256);
            debug_assert!((i>0 && histogram[i - 1] == prev_non_zero) || (i == 0 && prev_non_zero == 0.));
            let hist_i = histogram[offset + i];
            if hist_i == 0. {
                let mut j = i;
                debug_assert_eq!(histogram[offset + i], hist_i);
                debug_assert_eq!(hist_i, 0.);
                debug_assert!(j < 256);
                let mut hist_j = 0.;
                while j < 256 {
                    hist_j = histogram[offset + j];
                    if hist_j == 0. {
                        j += 1;
                    } else {
                        debug_assert_eq!(histogram[offset + j], hist_j);
                        debug_assert_ne!(hist_j, 0.);
                        break;
                    }
                }
                debug_assert!((histogram[offset + j] == hist_j && hist_j != 0.) || (hist_j == 0. && j == 256));
                debug_assert!(j >= i);
                let delta_x = (j+1-i) as f32;
                debug_assert!((i>0 && histogram[offset + i - 1] == prev_non_zero) || (i == 0 && prev_non_zero == 0.));
                let delta_y = hist_j - prev_non_zero;
                let b = delta_y / delta_x;
                for k in i..j {
                    // i =< k < j
                    // i-1 < k < j
                    // i-1 - k < 0 < j - k
                    // 0 < k - i+1
                    // 0 < k - i+1 < delta_x
                    // prev_non_zero < prev_non_zero + (k - i+1)*b < prev_non_zero + delta_x*b == prev_non_zero + delta_y == hist_j
                    debug_assert!(if b >= 0. {prev_non_zero + (k+1 - i) as f32 *b < hist_j} else{hist_j < prev_non_zero + (k+1 - i) as f32 *b});
                    debug_assert!(if b >= 0. {prev_non_zero < prev_non_zero + (k - i+1) as f32 *b}else{prev_non_zero + (k - i+1) as f32 *b < prev_non_zero});
                    // by induction: prev_non_zero != 0 or (i == 0 and prev_non_zero == 0)
                    histogram[offset+k] = prev_non_zero + (k - i + 1) as f32 * b;
                    debug_assert!(if b >=0. {prev_non_zero < histogram[offset+k]}else{histogram[offset+k] < prev_non_zero});
                    debug_assert!(if b >= 0. {histogram[offset+k] < hist_j}else{hist_j < histogram[offset+k]});
                }
                i = j + 1;
                debug_assert_ne!(histogram[offset+i-1],0.);
                debug_assert_ne!(hist_j, 0.);
                prev_non_zero = hist_j ;

            }else{
                i += 1;
                debug_assert_ne!(histogram[offset+i-1], 0.);
                debug_assert_eq!(hist_i, histogram[offset+i-1]);
                prev_non_zero = hist_i;
            }
        }
        debug_assert_eq!(i,256);
        offset = i;
    }
}


pub fn histogram(image: &[u8], stride: usize, source_mask: impl Fn(usize) -> bool) -> Box<[u32; 256]> {
    let mut slice = Box::new([0; 256]);
    for (idx, pixel) in image.iter().step_by(stride).cloned().enumerate() {
        if source_mask(idx) {
            slice[pixel as usize] += 1
        }
    }
    slice
}

/**`image` shape is `[height,width,channels]`*/
pub fn histograms(image: &[u8], channels: usize, source_mask: impl Fn(usize) -> bool) -> Box<[[u32; 256]]> {
    let mut slice = vec![[0u32; 256]; channels];
    for channel in 0..channels {
        let subslice = &mut slice[channel];
        for (idx, pixel) in image[channel..].iter().step_by(channels).cloned().enumerate() {
            if source_mask(idx) {
                subslice[pixel as usize] += 1;
            }
        }
    }
    slice.into_boxed_slice()
}

pub fn _normalize_histogram(histogram: Box<[u32; 256]>) -> Box<[f32; 256]> {
    let sum_inv = 1. / histogram.iter().sum::<u32>() as f32;
    _map_boxed_arr(histogram, |a| a as f32 * sum_inv)
}

pub fn normalize_histogram(histogram: &[u32; 256]) -> Box<[f32; 256]> {
    let sum_inv = 1. / histogram.iter().sum::<u32>() as f32;
    Box::new(map_arr(histogram, |&a| a as f32 * sum_inv))
}
//
// /**Histograms with multiple channels*/
// pub fn _normalize_histograms(histograms: Box<[u32]>) -> Box<[f32]> {
//     _map_boxed_slice(histograms,|a| {
//         let sum_inv = 1. / a.sum() as f32;
//         a._transmute(|a| a as f32 * sum_inv)
//     })
// }

/**Histograms with multiple channels*/
pub fn normalize_histograms(histograms: &[u32]) -> Box<[f32]> {
    assert_eq!(histograms.len() % 256, 0);
    let channels: usize = histograms.len() / 256;
    let mut o = Vec::with_capacity(histograms.len());
    for channel in 0..channels {
        let offset = channel * 256;
        let sum_inv = 1. / histograms[offset..offset + 256].iter().sum::<u32>() as f32;
        for i in offset..offset + 256 {
            debug_assert_eq!(o.len(), i);
            o.push(histograms[i] as f32 * sum_inv);
        }
    }
    o.into_boxed_slice()
}

pub fn match_histogram(source: &[u8], src_stride: usize, reference: &[u8], ref_stride: usize, source_mask: impl Fn(usize) -> bool) -> Vec<u8> {
    let mut o = Vec::with_capacity(source.len());
    unsafe { o.set_len(o.capacity()) }
    match_histogram_(source, src_stride, reference, ref_stride, &mut o, src_stride, source_mask);
    o
}

pub fn match_histogram_(source: &[u8], src_stride: usize, reference: &[u8], ref_stride: usize, output: &mut [u8], out_stride: usize, source_mask: impl Fn(usize) -> bool) {
    let hist_ref = _normalize_histogram(histogram(reference, ref_stride, &source_mask));
    match_precomputed_histogram_(source, src_stride, hist_ref.as_slice(), output, out_stride, source_mask)
}

pub fn match_precomputed_histogram(source: &[u8], src_stride: usize, hist_ref: &[f32], source_mask: impl Fn(usize) -> bool) -> Vec<u8> {
    let mut o = Vec::with_capacity(source.len());
    unsafe { o.set_len(o.capacity()) }
    match_precomputed_histogram_(source, src_stride, hist_ref, &mut o, src_stride, source_mask);
    o
}

pub fn match_precomputed_histogram_(source: &[u8], src_stride: usize, hist_ref: &[f32], output: &mut [u8], out_stride: usize, source_mask: impl Fn(usize) -> bool) {
    let hist_src = histogram(source, src_stride, &source_mask);
    match_2precomputed_histogram_(source, src_stride, hist_src.as_slice(), &hist_ref, output, out_stride, source_mask)
}

/**You can use source_mask to apply histogram matching partially only to a masked region of image. `source_mask(idx)` takes as input `idx=y*img_width+x`.
 IMPORTANT! `hist_src` must sum up to
```
hist_src.iter().sum()==source.iter().step_by(src_stride).enumerate().filter(|(idx,pix_val)|source_mask(idx)).count()
```
so it must be recomputed specifically for every given mask
 */
pub fn match_2precomputed_histogram_(source: &[u8], src_stride: usize, hist_src: &[u32], hist_ref: &[f32], output: &mut [u8], out_stride: usize, source_mask: impl Fn(usize) -> bool) {
    debug_assert_eq!(hist_ref.len(), 256);
    debug_assert_eq!(hist_src.len(), 256);

    let sum_src: u32 = hist_src.iter().sum();

    let mut i_ref = 0;
    let mut stack: Vec<(/*reference value to replace source value*/u8, /*how much source value to replace*/usize)> = Vec::new();
    /**for each source pixel value stores a slice (offset,end) into stack that tells us how to replace that value*/
    let mut stack_offsets = Box::new([(0usize, 0usize); 256]);
    let mut stack_offset = 0;
    let mut popped_src = 0i32;
    for i_src in 0..256 {
        if popped_src <= 0 {
            let to_pop = hist_src[i_src] as i32;
            let to_replace = to_pop.min(-popped_src);
            if to_replace > 0 {
                stack.push(((i_ref - 1) as u8, to_replace as usize));//we should replace `to_replace` pixels of value `i_src` with value `i_ref`
            }
            popped_src += to_pop;
        }
        while popped_src > 0 {
            if i_ref >= 256 {
                stack.push((255u8, popped_src as usize));
                popped_src = 0;
                break;
            }
            let popped_ref = hist_ref[i_ref];
            debug_assert!(popped_ref >= 0., "reference histogram has negative value {} at {}", popped_ref, i_ref);
            debug_assert!(popped_ref <= 1., "reference histogram has value {} exceeding 1 at {}(are you sure it is normalized?)", popped_ref, i_ref);
            let corresponding_src = (popped_ref * sum_src as f32) as i32; // it is done in this way for better numerical stability
            let replaced_src = corresponding_src.min(popped_src);
            popped_src -= corresponding_src;
            if replaced_src > 0 {
                stack.push((i_ref as u8, replaced_src as usize));//we should replace `replaced_src` pixels of value `i_src` with value `i_ref`
            }
            i_ref += 1;
        }

        debug_assert_eq!(stack[stack_offset..].iter().map(|&(_, s)| s).sum::<usize>(), hist_src[i_src] as usize);
        stack_offsets[i_src] = (stack_offset, stack.len()); // stack tells us how many pixels of value `i_src` should be replaced into various other values of `i_ref`
        stack_offset = stack.len();
    }
    // debug_assert!(true);
    // If at this point i_ref < 256, that can only be due to floating-point imprecision. (which is unlikely as we use f64)
    // A few pixels might be improperly replaced but that's fine. Nobody will notice.

    'outer: for ((idx, src_i), out_i) in source
        .iter().step_by(src_stride).cloned()
        .enumerate()
        .zip(output.iter_mut().step_by(out_stride)) {
        if source_mask(idx) { // we can partially apply histogram matching only to a masked region of an image
            let (mut offset, end) = stack_offsets[src_i as usize];
            while offset < end {
                let (ref_i, to_replace) = stack[offset];
                if to_replace > 0 {
                    *out_i = ref_i;
                    stack[offset].1 = to_replace - 1;
                    continue 'outer;
                } else {
                    offset += 1;
                    stack_offsets[src_i as usize].0 = offset;
                }
            }
            // coming here should rarely happen. Only possible due to FP imprecision. Usually we hit continue statement
        }
        *out_i = src_i;
    }
}

pub fn blend(scalar1: f32, histogram1: &[f32], scalar2: f32, histogram2: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(histogram1.len());
    unsafe { out.set_len(out.capacity()) }
    blend_(scalar1, histogram1, scalar2, histogram2, &mut out);
    out
}

pub fn blend_(scalar1: f32, histogram1: &[f32], scalar2: f32, histogram2: &[f32], output_histogram: &mut [f32]) {
    assert_eq!(histogram1.len(), histogram2.len());
    assert_eq!(output_histogram.len(), histogram2.len());
    assert!(scalar1 <= 1., "Scalar1={} is greater than 1", scalar1);
    assert!(scalar2 <= 1., "Scalar2={} is greater than 1", scalar2);
    assert!(0. <= scalar1, "Scalar1={} is less than 0", scalar1);
    assert!(0. <= scalar2, "Scalar2={} is less than 0", scalar2);
    for i in 0..histogram1.len() {
        debug_assert!(histogram1[i] <= 1., "histogram1[{}] is {}", i, histogram1[i]);
        debug_assert!(histogram2[i] <= 1., "histogram2[{}] is {}", i, histogram2[i]);
        debug_assert!(histogram1[i] >= 0., "histogram1[{}] is {}", i, histogram1[i]);
        debug_assert!(histogram2[i] >= 0., "histogram2[{}] is {}", i, histogram2[i]);
        output_histogram[i] = histogram1[i].mul_add(scalar1, histogram2[i] * scalar2)
    }
}

/**shape == [height, width, channels]*/
pub fn match_images(source: &[u8], src_shape: &[usize; 3], reference: &[u8], ref_shape: &[usize; 3], source_mask: impl Fn(usize) -> bool) -> Box<[u8]> {
    assert_eq!(src_shape[2], ref_shape[2]);
    let channels = src_shape[2];
    let len = src_shape[0] * src_shape[1];
    let mut out = Vec::<u8>::with_capacity(len * channels);
    unsafe { out.set_len(out.capacity()) }
    for channel in 0..channels {
        match_histogram_(&source[channel..], channels, &reference[channel..], channels, &mut out[channel..], channels, &source_mask);
    }
    out.into_boxed_slice()
}

/**shape == [height, width, channels],  hist_ref:[channels, 256]*/
pub fn match_precomputed_images(source: &[u8], src_shape: &[usize; 3], hist_ref: &[f32], source_mask: impl Fn(usize) -> bool) -> Box<[u8]> {
    let hist_src = histograms(source, src_shape[2], &source_mask);
    match_2precomputed_images(source, src_shape, hist_src.flatten(), hist_ref, source_mask)
}

/**shape == [height, width, channels], hist_src:[channels,256], hist_ref:[channels, 256]*/
pub fn match_2precomputed_images(source: &[u8], src_shape: &[usize; 3], hist_src: &[u32], hist_ref: &[f32], source_mask: impl Fn(usize) -> bool) -> Box<[u8]> {
    assert_eq!(hist_src.len(), hist_ref.len());
    let channels = src_shape[2];
    let len = src_shape[0] * src_shape[1];
    let mut out = Vec::<u8>::with_capacity(len * channels);
    unsafe { out.set_len(out.capacity()) }
    for channel in 0..channels {
        let hist_offset = channel * 256;
        match_2precomputed_histogram_(&source[channel..], channels, &hist_src[hist_offset..hist_offset + 256], &hist_ref[hist_offset..hist_offset + 256], &mut out[channel..], channels, &source_mask);
    }
    out.into_boxed_slice()
}

/**Finds reference histogram that is closest to the source histogram. The distance is mean square error.
 Returned distance is between 0 and the number of channels (because it's between 0 and 1 for each channel and then summed up)*/
pub fn find_closest(hist_src: &[[u32; 256]], batch: usize, references: &[f32]) -> (usize, f32) {
    let channels = hist_src.len();
    let channels256 = channels * 256;
    assert_eq!(references.len(), batch * channels * 256);
    let mut min_square_diff = f32::INFINITY;
    let mut best_ref_idx = 0;
    for ref_idx in 0..batch {
        let ref_offset = ref_idx * channels256;
        /**`ref_hists` shape is `[channels, 256]`*/
        let ref_hists = &references[ref_offset..ref_offset + channels256];
        let mut square_diff = 0.;
        for channel in 0..channels {
            let offset = channel * 256;
            let hist_src = &hist_src[channel];
            let hist_ref = &ref_hists[offset..offset + 256];
            let s_inv = 1. / hist_src.iter().sum::<u32>() as f32;
            fn sq(a: f32) -> f32 {
                a * a
            }
            fn f(a: &u32) -> f32 { *a as f32 }
            let channel_square_diff = hist_src.iter().map(f).zip(hist_ref.iter()).map(|(s, r)| sq(s * s_inv - r)).sum::<f32>();
            square_diff += channel_square_diff;
        }
        if square_diff < min_square_diff {
            min_square_diff = square_diff;
            best_ref_idx = ref_idx;
        }
    }
    (best_ref_idx, min_square_diff)
}

/**Finds reference histogram that is closest to the source histogram. The distance is mean square error.*/
pub fn find_closest_n(hist_src: &[f32], batch: usize, references: &[f32]) -> (usize, f32) {
    assert_eq!(hist_src.len() % 256, 0);
    let channels = hist_src.len() / 256;
    ;
    let channels256 = hist_src.len();
    assert_eq!(references.len(), batch * channels * 256);
    let mut min_square_diff = f32::INFINITY;
    let mut best_ref_idx = 0;
    for ref_idx in 0..batch {
        let ref_offset = ref_idx * channels256;
        /**`ref_hists` shape is `[channels, 256]`*/
        let ref_hists = &references[ref_offset..ref_offset + channels256];
        let mut square_diff = 0.;
        for channel in 0..channels {
            let offset = channel * 256;
            let hist_src = &hist_src[offset..offset + 256];
            let hist_ref = &ref_hists[offset..offset + 256];
            fn sq(a: f32) -> f32 {
                a * a
            }
            let channel_square_diff = hist_src.iter().zip(hist_ref.iter()).map(|(s, r)| sq(s - r)).sum::<f32>();
            square_diff += channel_square_diff;
        }
        if square_diff < min_square_diff {
            min_square_diff = square_diff;
            best_ref_idx = ref_idx;
        }
    }
    (best_ref_idx, min_square_diff)
}

/**shape == [height, width, channels], references:[batch,channels,256]*/
pub fn match_best_images(source: &[u8], src_shape: &[usize; 3], batch: usize, references: &[f32], source_mask: impl Fn(usize) -> bool) -> (Box<[u8]>, usize, f32) {
    assert_eq!(source.len(), src_shape.iter().product());
    let channels = src_shape[2];
    let hist_src = histograms(source, channels, &source_mask);
    let channels256 = channels * 256;
    let (best_ref_idx, min_square_diff) = find_closest(&hist_src, batch, references);
    let ref_offset = best_ref_idx * channels256;
    let best_ref_hists = &references[ref_offset..ref_offset + channels256];
    (match_2precomputed_images(source, src_shape, hist_src.flatten(), best_ref_hists, source_mask), best_ref_idx, min_square_diff)
}

/**Find if there exists `x1` and `x2` such that
```
h/w >= ratio
```
where
```
w = (x2-x1-1)/histogram.len();
y_max = histogram[x1..x2].max();
y_min = histogram[x1].max(histogram[x2]);
h = y_max-y_min;
```
and `ratio>1`
 */
pub fn find_n_histogram_anomaly<T: Copy + Zero + PartialOrd + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + FromUsize>(histogram: &[T], ratio: T) -> Vec<Range<isize>> {
    let ratio = ratio / T::from_usize(histogram.len());
    // w = (x2-x1-1)/histogram.len()
    // h/w >= ratio
    // h/((x2-x1-1)/histogram.len()) >= ratio
    // histogram.len()*h/(x2-x1-1) >= ratio
    // h/(x2-x1-1) >= ratio/histogram.len()
    find_histogram_anomaly(histogram, ratio)
}

/**Find if there exists `x1` and `x2` such that
```
h/w >= ratio
```
where
```
w = x2-x1-1;
y_max = histogram[x1..x2].max();
y_min = histogram[x1].max(histogram[x2]);
h = y_max-y_min;
```
and `ratio>1`
 */
pub fn find_histogram_anomaly<T: Copy + Zero + PartialOrd + Sub<Output=T> + Mul<Output=T> + FromUsize>(histogram: &[T], ratio: T) -> Vec<Range<isize>> {
    let mut out = Vec::new();
    let mut x1: isize = -1;
    let mut y1 = T::zero();
    let l = histogram.len() as isize;
    let mut to_be_pushed = x1..x1;
    'outer: while x1 + 1 < l {
        let next_y = histogram[(x1 + 1) as usize];
        if next_y >= y1 {
            let mut y_max = next_y;
            for x2 in x1 + 2..l + 1 {
                let y2 = if x2 < l { histogram[x2 as usize] } else { T::zero() };
                if y2 > y_max {
                    y_max = y2;
                }

                let y_min = if y1 > y2 { y1 } else { y2 };
                let w = T::from_usize((x2 - x1 - 1) as usize);
                let h = y_max - y_min;
                let r = w * ratio;
                if h >= r {
                    if to_be_pushed.start < to_be_pushed.end && to_be_pushed.end != x2 {
                        out.push(to_be_pushed);
                    }
                    to_be_pushed = x1..x2;
                    break;
                }
            }
        }
        y1 = next_y;
        x1 += 1;
    }
    if to_be_pushed.start < to_be_pushed.end {
        out.push(to_be_pushed);
    }
    out
}

pub fn find_histograms_anomaly<T: Copy + Zero + PartialOrd + Sub<Output=T> + Mul<Output=T> + FromUsize>(histograms: &[T], ratio: T) -> Vec<(usize, Range<isize>)> {
    let mut out = Vec::new();
    assert_eq!(histograms.len() % 256, 0);
    let channels = histograms.len() / 256;
    for channel in 0..channels {
        let offset = channel * 256;
        let a = find_histogram_anomaly(&histograms[offset..offset + 256], ratio);
        out.extend(a.into_iter().map(|r| (channel, r)));
    }
    out
}

pub fn find_n_histograms_anomaly<T: Copy + Zero + PartialOrd + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + FromUsize>(histograms: &[T], ratio: T) -> Vec<(usize, Range<isize>)> {
    let ratio = ratio / T::from_usize(histograms.len());
    find_histograms_anomaly(histograms, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use crate::init_rand::InitRandWithCapacity;


    #[test]
    fn test6() {
        let s = vec![0, 1, 2];
        let r = vec![3, 4, 5];
        let o = match_images(&s, &[s.len(), 1, 1], &r, &[r.len(), 1, 1], |_| true);
        assert_eq!(o.as_ref(), &[3, 4, 5]);
    }

    #[test]
    fn test5() {
        let s = vec![0, 1, 2, 2, 5, 7, 3, 4, 6];
        let r = vec![3, 4, 5];
        let o = match_images(&s, &[s.len(), 1, 1], &r, &[r.len(), 1, 1], |_| true);
        assert_eq!(o.as_ref(), &[3, 3, 3, 4, 5, 5, 4, 4, 5]);
    }

    #[test]
    fn test4() {
        let o = find_n_histogram_anomaly(&[0., 0., 0.1, 0.1, 0.4, 0.1, 0.1, 0.1], 2.);
        assert_eq!(o, vec![3..5]);
    }

    #[test]
    fn test3() {
        let o = find_n_histogram_anomaly(&[0.4, 0., 0., 0.1, 0.1, 0.1, 0.1, 0.1], 2.);
        assert_eq!(o, vec![-1..1]);
    }

    #[test]
    fn test2() {
        let o = find_n_histogram_anomaly(&[0.0, 0., 0., 0.1, 0.1, 0.1, 0.1, 0.4], 2.);
        assert_eq!(o, vec![6..8]);
    }

    #[test]
    fn test1() {
        let mut a = vec![0.0, 1., 0., 3.0, 8.0, 0.0, 0.0, 2.];
        let mut b = Vec::rand(256);
        for i in rand_set(100,a.len()..256){
            b[i] = 0.;
        }
        b[..a.len()].copy_from_slice(&a);
        interpolate_gaps(&mut b);
        assert_eq!(&b[..a.len()], &[0.5,1.,2.,3.,8.,6.0,4.0,2.]);
    }
}