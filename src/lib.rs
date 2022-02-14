#![doc = include_str!("../README.md")]

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate log;
extern crate num_traits as libnum;
extern crate anyhow;

/// Contains our various nonlinear solvers and functionality related to these solvers such as the traits
/// necessary to define our nonlinear problem that can be used within the solvers.
/// It should be noted that these solvers are designed around small problem size think dimensions less than
/// 1 - 100. As one increases the dimension, limitations start to appear due to the array size and the linear solvers.
pub mod nonlinear_solver;
pub mod helix_error;

#[cfg(not(feature = "linear_algebra"))]
pub(crate) mod linear_algebra;

/// Contains our linear solvers and various BLAS-like operations that are useful for solving nonlinear problems. These
/// methods are not meant for large dimension problems, but they should be fine for small system sizes.
#[cfg(feature = "linear_algebra")]
pub mod linear_algebra;
