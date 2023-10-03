#![feature(generic_const_exprs)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array)]
#![feature(step_trait)]
#![feature(iter_collect_into)]
#![feature(portable_simd)]
#![feature(slice_flatten)]
#![feature(const_ptr_read)]
#![feature(const_refs_to_cell)]
#![feature(new_uninit)]

extern crate core;



pub mod conv;
pub mod dot_arr;
pub mod top_k;
pub mod static_layout;
pub mod init;
pub mod dot_slice;
pub mod shaped_tensor_mad;
pub mod dot_sparse_arr;
pub mod shape;
pub mod layout;
pub mod soft_wta;
pub mod init_rand;
pub mod conv_shape;
pub mod vec_range;
pub mod from_usize;
pub mod mat_slice;

pub mod cayley;
pub mod blas_safe;

mod mat_arr;
pub use mat_arr::*;
pub use statrs::*;
pub use levenshtein::*;
mod mat_tri;
pub use mat_tri::*;
pub use levenshtein::*;
mod arr_concat;
pub use arr_concat::*;
mod shape_arr;
pub use shape_arr::*;
mod tup_arr;
pub use tup_arr::*;
mod norm;
pub use norm::*;
mod xyzw;
pub use xyzw::*;
mod vector_field;
pub use vector_field::*;
mod vector_field_arr;
pub use vector_field_arr::*;
mod vector_field_vec;
pub use vector_field_vec::*;
mod vector_field_slice;
pub use vector_field_slice::*;
mod set;
pub use set::*;
pub mod histogram;
mod num;
mod vec;
pub mod dot_sparse_slice;
pub mod ecc_layer;


pub use vec::*;
pub use num::*;
