#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate num_traits as libnum;
extern crate log;

pub(crate) mod linear_algebra;
pub mod nonlinear_solver;
