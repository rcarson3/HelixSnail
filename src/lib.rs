#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate num_traits as libnum;
extern crate log;

pub mod nonlinear_solver;

#[cfg(not(feature = "linear_algebra"))]
pub(crate) mod linear_algebra;

#[cfg(feature = "linear_algebra")]
pub mod linear_algebra;