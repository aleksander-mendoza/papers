pub fn slice_as_arr<X,const L:usize>(a:&[X])->&[X;L]{
    assert_eq!(a.len(),L);
    unsafe{&*(a.as_ptr() as *const [X; L])}
}
pub fn arr1<X>(a:(X))->[X;1]{
    let (a) = a;
    [a]
}
pub fn arr2<X>(a:(X,X))->[X;2]{
    let (a,b) = a;
    [a,b]
}
pub fn arr3<X>(a:(X,X,X))->[X;3]{
    let (a,b,c) = a;
    [a,b,c]
}
pub fn arr4<X>(a:(X,X,X,X))->[X;4]{
    let (a,b,c,d) = a;
    [a,b,c,d]
}
pub fn arr5<X>(a:(X,X,X,X,X))->[X;5]{
    let (a,b,c,d,e) = a;
    [a,b,c,d,e]
}
pub fn arr6<X>(a:(X,X,X,X,X,X))->[X;6]{
    let (a,b,c,d,e, f) = a;
    [a,b,c,d,e,f]
}

pub fn tup1<X>(a:[X;1])->(X){
    let [a] = a;
    (a)
}
pub fn tup2<X>(a:[X;2])->(X,X){
    let [a,b] = a;
    (a,b)
}
pub fn tup3<X>(a:[X;3])->(X,X,X){
    let [a,b,c] = a;
    (a,b,c)
}
pub fn tup4<X>(a:[X;4])->(X,X,X,X){
    let [a,b,c,d] = a;
    (a,b,c,d)
}
pub fn tup5<X>(a:[X;5])->(X,X,X,X,X){
    let [a,b,c,d,e] = a;
    (a,b,c,d,e)
}
pub fn tup6<X>(a:[X;6])->(X,X,X,X,X,X){
    let [a,b,c,d,e, f] = a;
    (a,b,c,d,e,f)
}