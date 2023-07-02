use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::random;
use crate::init::{filled2, filled3, InitWithCapacity};

pub  trait InitRandWithCapacity{
    fn rand(capacity:usize)->Self;
}

impl <T> InitRandWithCapacity for Vec<T> where Standard: Distribution<T>{
    fn rand(capacity: usize) -> Self {
        Vec::init_with(capacity, |_|random())
    }
}

pub fn rand2<A:Copy, const W:usize, const H:usize>()->[[A;W];H] where Standard: Distribution<A>{
    filled2(|_|rand::random())
}

pub fn rand3<A:Copy, const W:usize, const H:usize,const D:usize>()->[[[A;W];H];D] where Standard: Distribution<A>{
    filled3(|_|rand::random())
}