#![doc = include_str!("../README.md")]
#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs, generic_arg_infer)]

extern crate log;
extern crate num_traits as libnum;
extern crate bytemuck;

// Once trait aliases are stabilized can replace these two lines with:
// pub trait FloatType = libnum::Float + libnum::Zero + libnum::One + libnum::NumAssignOps + libnum::NumOps + core::fmt::Debug;
pub trait FloatType: libnum::Float + libnum::Zero + libnum::One + libnum::NumAssignOps + libnum::NumOps + core::fmt::Debug + bytemuck::Pod {}
impl<F: libnum::Float + libnum::Zero + libnum::One + libnum::NumAssignOps + libnum::NumOps + core::fmt::Debug + bytemuck::Pod> FloatType for F {}

pub fn array1d_to_array2d<const NDIM: usize, F>(vec: &[F]) -> &[[F; NDIM]] 
where 
    F: crate::FloatType,
{
    assert!(vec.len() % NDIM == 0);
    bytemuck::try_cast_slice(vec).unwrap()
}

pub fn array1d_to_array2d_mut<const NDIM: usize, F>(vec: &mut [F]) -> &mut [[F; NDIM]] 
where 
    F: crate::FloatType,
{
    assert!(vec.len() % NDIM == 0);
    bytemuck::try_cast_slice_mut(vec).unwrap()
}

pub fn array2d_to_array1d<const NDIM: usize, F>(mat: &[[F; NDIM]]) -> &[F] 
where 
    F: crate::FloatType,
{
    bytemuck::try_cast_slice(mat).unwrap()
}

pub fn array2d_to_array1d_mut<const NDIM: usize, F>(mat: &mut [[F; NDIM]]) -> &mut [F] 
where 
    F: crate::FloatType,
{
    bytemuck::try_cast_slice_mut(mat).unwrap()
}


/// Contains the basic error types for our solvers whether they're either from our nonlinear or linear solvers.
pub mod helix_error;
/// Contains our various nonlinear solvers and functionality related to these solvers such as the traits
/// necessary to define our nonlinear problem that can be used within the solvers.
/// It should be noted that these solvers are designed around small problem size think dimensions less than
/// 1 - 100. As one increases the dimension, limitations start to appear due to the array size and the linear solvers.
pub mod nonlinear_solver;

#[cfg(not(feature = "linear_algebra"))]
pub(crate) mod linear_algebra;

/// Contains our linear solvers and various BLAS-like operations that are useful for solving nonlinear problems. These
/// methods are not meant for large dimension problems, but they should be fine for small system sizes.
#[cfg(feature = "linear_algebra")]
pub mod linear_algebra;
