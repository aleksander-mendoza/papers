use num_traits::AsPrimitive;

/**Cayley graph is an array of shape `[group_elements, group_generators]` such that each group element keeps track of its
neighbouring states for each generating element.
The element at index 0 is neutral by convention.
If `a`-th generator element has an inverse, then by convention it is either `a+1` or `a-1`*/
pub type CayleyGraph = [usize];

pub fn cyclic_monoid(n: usize) -> (Vec<usize>, usize) {
    let mut b:Vec<usize> = (0..n).map(|i|i+1).collect();
    b[n - 1] = 0;
    (b, 1)
}

/**`0` and `1` are inverse generator elements*/
pub fn cyclic_group(n: usize) -> (Vec<usize>, usize) {
    let mut b:Vec<usize> = Vec::with_capacity(n * 2);
    for i in 0..n {
        b.push(i+1); // next element
        b.push( i.wrapping_sub(1)); // inverse
        // b[i * 2 + 0] = i + 1;
        // b[i * 2 + 1] = i.wrapping_sub(1);
    }
    b[(n - 1) * 2 + 0] = 0;
    b[0 * 2 + 1] = n - 1;
    (b, 2)
}

/**`a` is of shape `[group_elements_a, group_generators_a]`.
`b` is of shape `[group_elements_b, group_generators_b]`.
computes direct product `a×b` of two monoids by mapping `i`-th element of `a` and `j`-th element of `b` to `i*group_elements_b+j`-th element of `a×b`.
If `x` is the `i`-th generator of `a`  then `x` is mapped to `i`-th generator `(x,e)` of `a×b`.
If `x` is the `i`-th generator of `b`  then `x` is mapped to `group_generators_a+i`-th generator `(e,x)` of `a×b`.
 `a×b` is of shape `[group_elements_a*group_elements_b, group_generators_a+group_generators_b]`*/
pub fn direct_product(a: &[usize], gen_len_a: usize, b: &[usize], gen_len_b: usize) -> (Vec<usize>, usize) {
    let gen_len_c = gen_len_a + gen_len_b;
    let mut c = Vec::with_capacity(a.len() * b.len() * gen_len_c);
    let a_size = a.len() / gen_len_a;
    let b_size = b.len() / gen_len_b;
    for element_a in 0..a_size {
        for element_b in 0..b_size {
            for action_a in 0..gen_len_a {
                let neighbour_a = a[element_a * gen_len_a + action_a];
                let neighbour_c = neighbour_a * b_size + element_b;
                assert_eq!(c.len(), (element_a * b_size + element_b) * gen_len_c + action_a);
                c.push(neighbour_c);
            }
            for action_b in 0..gen_len_b {
                let neighbour_b = b[element_b * gen_len_b + action_b];
                let neighbour_c = element_a * b_size + neighbour_b;
                assert_eq!(c.len(), (element_a * b_size + element_b) * gen_len_c + gen_len_a + action_b);
                c.push(neighbour_c);
            }
        }
    }

    (c, gen_len_c)
}

/**`state_space` has shape `[states,generator_elements]`, `w` has shape `[states,quotient_monoid_elements]` and its rows sum up to `1`.
 `u` has shape `[generator_elements,quotient_monoid_elements,quotient_monoid_elements]`*/
pub fn learn_u(state_space: &[usize], generator_elements: usize, w: &[f32], quotient_monoid_elements: usize, u: &mut [f32]) {
    assert_eq!(state_space.len() % generator_elements, 0);
    let n = state_space.len() / generator_elements;
    let m = quotient_monoid_elements;
    assert_eq!(w.len(), n * m);
    assert_eq!(u.len(), generator_elements * m * m);
    for a in 0..generator_elements {
        for g in 0..m {
            for next_g in 0..m {
                let mut sum_wh_wha = 0.;
                let mut sum_wh = 0.;
                for h in 0..n {
                    let ha = state_space[h * generator_elements + a];
                    if ha < n {
                        let w_ha = w[ha * m + next_g];
                        let w_h = w[h * m + g];
                        sum_wh_wha += w_ha * w_h;
                        sum_wh += w_h;
                    }
                }
                u[(a * m + g) * m + next_g] = sum_wh_wha / sum_wh;
            }
        }
    }
}

/**`state_space` has shape `[states,generator_elements]`, `w` has shape `[states,quotient_monoid_elements]` and its rows sum up to `1`.
 `u` has shape `[generator_elements,quotient_monoid_elements,quotient_monoid_elements]`. `new_w` is of same shape as `w` and is filled with zeroes*/
pub fn learn_w(state_space: &[usize], generator_elements: usize, w: &[f32], quotient_monoid_elements: usize, u: &[f32], new_w: &mut [f32]) {
    assert_eq!(state_space.len() % generator_elements, 0);
    let n = state_space.len() / generator_elements;
    let m = quotient_monoid_elements;
    assert_eq!(w.len(), n * m);
    assert_eq!(w.len(), new_w.len());
    assert_eq!(u.len(), generator_elements * m * m);
    let mut neighbour_count = vec![0; new_w.len()];

    for a in 0..generator_elements {
        for h in 0..n {
            let ha = state_space[h * generator_elements + a];
            if ha < n {
                for next_g in 0..m {
                    let mut sum = 0.;
                    for g in 0..m {
                        let w_h = w[h * m + g];
                        let u_a = u[(a * m + g) * m + next_g];
                        sum += u_a * w_h;
                    }
                    new_w[ha * m + next_g] += sum;
                    neighbour_count[ha * m + next_g] += 1;
                }
            }
        }
    }
    new_w.iter_mut().zip(neighbour_count.iter()).for_each(|(w, &n)| *w /= n as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test6() {
        let (c1, g1) = cyclic_group(3);
        let (c2, g2) = cyclic_group(4);
        let (c12, g12) = direct_product(&c1, g1, &c2, g2);
    }
}