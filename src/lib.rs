//!
//!# HelixSnail
//!A small nonlinear solver library written in Rust. It is based on the c++ [snls](https://github.com/LLNL/SNLS) library by LLNL. This crate is largely an experiment to see how we can make performant code in a safer language where we don't have to resort to pointer filled code. Additionally, this code is also generic over the float type which the original one was only suitable to run with f64/double types.
//!
//!It is written such that it can be used in `no_std` environments, so it can be used on the GPU using crates such as [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA).
//!
//!# Example
//!
//! ```Rust
//! #![allow(incomplete_features)]
//! #![feature(generic_const_exprs)]
//! extern crate env_logger;
//! extern crate helix_snail;
//! extern crate num_traits as libnum;
//!
//! use helix_snail::nonlinear_solver::*;
//! use libnum::{Float, NumAssignOps, NumOps, One, Zero};
//! use log::info;
//! // This doesn't need to be a global value.
//! // I just had it for testing purposes.
//! // A value less than or equal to 0 does not log anything
//! // Any value greater than 0 will cause logs to be produced
//! const LOGGING_LEVEL: i32 = 1;
//!
//! struct Broyden<F>
//! where
//! F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug,
//! {
//! lambda: F,
//! pub logging_level: i32,
//! }
//!
//! impl<F> NonlinearProblem<F> for Broyden<F>
//! where
//! F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug,
//! {
//! const NDIM: usize = 8;
//! fn compute_resid_jacobian(&mut self, fcn_eval: &mut [F], opt_jacobian: Option<&mut [F]>, x: &[F]) -> bool {
//!     assert!(fcn_eval.len() >= Self::NDIM);
//!     assert!(x.len() >= Self::NDIM);
//!
//!     let two: F = F::from(2.0).unwrap();
//!     let three: F = F::from(3.0).unwrap();
//!     let four: F = F::from(4.0).unwrap();
//!
//!     if self.logging_level > 0 {
//!         info!("Evaluating at x = ");
//!         for i in 0..Self::NDIM {
//!             info!(" {:?} ", x[i]);
//!         }
//!     }
//!
//!     fcn_eval[0] = (three - two * x[0]) * x[0] - two * x[1] + F::one();
//!     for i in 1..(Self::NDIM - 1) {
//!         fcn_eval[i] = (three - two * x[i]) * x[i] - x[i - 1] - two * x[i + 1] + F::one();
//!     }
//!
//!     let fcn =
//!         (three - two * x[Self::NDIM - 1]) * x[Self::NDIM - 1] - x[Self::NDIM - 2] + F::one();
//!
//!     fcn_eval[Self::NDIM - 1] = (F::one() - self.lambda) * fcn + self.lambda * fcn * fcn;
//!
//!     if let Some(jacobian) = opt_jacobian {
//!         assert!(
//!             jacobian.len() >= Self::NDIM * Self::NDIM,
//!             "length {:?}",
//!             jacobian.len()
//!         );
//!
//!         for ij in 0..(Self::NDIM * Self::NDIM) {
//!             jacobian[ij] = F::zero();
//!         }
//!         
//!         jacobian[0 * Self::NDIM + 0] = three - four * x[0];
//!         jacobian[0 * Self::NDIM + 1] = -two;
//!         // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
//!         for i in 1..(Self::NDIM - 1) {
//!             jacobian[i * Self::NDIM + i - 1] = -F::one();
//!             jacobian[i * Self::NDIM + i] = three - four * x[i];
//!             jacobian[i * Self::NDIM + i + 1] = -two;
//!         }
//!
//!         let dfndxn = three - four * x[Self::NDIM - 1];
//!         // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
//!         jacobian[(Self::NDIM - 1) * Self::NDIM + (Self::NDIM - 1)] =
//!             (F::one() - self.lambda) * dfndxn + self.lambda * two * dfndxn * fcn;
//!         jacobian[(Self::NDIM - 1) * Self::NDIM + (Self::NDIM - 2)] =
//!             (-F::one() + self.lambda) * F::one() - self.lambda * two * fcn;    
//!     }
//!
//!     true
//! }
//! }
//!
//! fn main() {
//!     let _ = env_logger::builder().is_test(true).try_init();
//!     
//!     let mut broyden = Broyden::<f64> {
//!         lambda: $lambda,
//!         logging_level: LOGGING_LEVEL,
//!     };
//!
//!     let dc = TrustRegionDeltaControl::<f64> {
//!         delta_init: 1.0,
//!         ..Default::default()
//!     };
//!
//!     let mut solver = TrustRegionDoglegSolver::<f64, Broyden<f64>>::new(&dc, &mut broyden);
//!
//!     for i in 0..Broyden::<f64>::NDIM {
//!         solver.x[i] = 0.0;
//!     }
//!
//!     solver.set_logging_level(Some(LOGGING_LEVEL));
//!     solver.setup_options(Broyden::<$type>::NDIM * 10, $tolerance, Some(LOGGING_LEVEL));
//!
//!     let status = solver.solve();
//!
//!     assert!(
//!         status == NonlinearSolverStatus::Converged,
//!         "Solution did not converge"
//!     );
//! }
//! ```

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate log;
extern crate num_traits as libnum;

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
