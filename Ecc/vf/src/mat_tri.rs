
/**
Returns number of non-zero elements in triangular matrix. An example below has side length 5
and 10 non-zero elements.
```
0  a9  a8  a7  a6
0   0  a5  a4  a3
0   0   0  a2  a1
0   0   0   0  a0
0   0   0   0   0
```
 */
pub fn tri_len(side_length:usize)->usize{
    (side_length * (side_length-1)) / 2
}
/**
Given number of non-zero elements in triangular matrix returns the side length.
An example below has 10 non-zero elements and side length 5.
https://en.wikipedia.org/wiki/Triangular_number#Triangular_roots_and_tests_for_triangular_numbers
```
0  a9  a8  a7  a6
0   0  a5  a4  a3
0   0   0  a2  a1
0   0   0   0  a0
0   0   0   0   0
```
 */
pub fn tri_side_len(len:usize)->f32{
    (8.*len as f32-1.).sqrt()/2.
}
/**
Returns number of non-zero elements in triangular matrix including diagonal. An example below has side length 4
and 10 non-zero elements.
```
a9  a8  a7  a6
 0  a5  a4  a3
 0   0  a2  a1
 0   0   0  a0
```
 */
pub fn trid_len(side_length:usize)->usize{
    (side_length * (side_length+1)) / 2
}

/**
Index when given x,y coordinate. Example pos=[0,3] yields idx=2
```
0  a0  a1  a2  a3
0   0  a4  a5  a6
0   0   0  a7  a8
0   0   0   0  a9
0   0   0   0   0
```
 */
pub fn triu_idx(pos:[usize;2], side_length:usize)->usize{
    let n = side_length;
    let [i,j] = pos;
    (n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1
}

/**
x,y coordinate when given index. Example idx=8 yields pos=[2,4]
```
0  a0  a1  a2  a3
0   0  a4  a5  a6
0   0   0  a7  a8
0   0   0   0  a9
0   0   0   0   0
```
 */
pub fn triu_pos(idx:usize, side_length:usize)->[usize;2]{
    let n = side_length;
    let i = n - 2 - (((4*n*(n-1)-8*idx-7) as f64).sqrt()/2.0 - 0.5) as usize;
    let j = idx + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
    [i,j]
}