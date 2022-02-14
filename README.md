# HelixSnail
This crate is designed to solve nonlinear systems of equations that are on the smaller size (aka between 1..100 dimensions). It is based on my experience as a developer of the c++ [snls](https://github.com/LLNL/SNLS) library by LLNL. I will mention that the work here is no-way or shape associated with my work on that project. This crate is purely a personal project of mine to further explore numerical/scientific coding in Rust while not subjective to design choices of existing libraries. While designs made here are similar to the original C++ library, I am making several major changes such as making the crate generic enough to allow any float type be used by the library. The original library restricted things to `double \ f64` types. Next, I am exploring a number of different design choices using Rust trait system that allow us to depart from the original library. I have found it relatively simple to define basic traits that a nonlinear solver and objects related to them should share. This is a sharp departure from my work on the SNLS library where I found adding various aspects to SNLS to be tougher. Although, I will admit that supporting batch solvers within SNLS should be simpler than within Rust thanks to the RAJA abstraction library.

HelixSnail is written with a `no_std` environment being the default environment. This choice was deliberately done, so we can easily run on the GPU using crates such as [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA).

# Solvers

* `TrustRegionDoglegSolver` - this nonlinear solver makes use of a model trust-region method that makes use of a dogleg solver for the sub-problem of the nonlinear problem. It reduces down to taking a full newton raphson step when a given step is near the solution.

* The hybrid version of the trust region method with a dogleg solver from `SNLS` will also be ported at some point.

* Variations of these solvers that make use of different `DeltaControl` objects will also be supported at some point. However, it will require additional research into acceptable design choices of the nonlinear solvers in order to reduce code duplication.

# Optional features
By enabling the `linear_algebra` feature, external libraries / applications can also make use of the linear solvers and various BLAS-like functions written for this crate. These methods can be useful while writing out necessary functions for small-sized nonlinear problems. It is important to note these methods are not meant for large dimension problems, but they should be fine for small system sizes.

# Example
The below example is taken from the test suit, but it shows how to define your nonlinear problem (`Broyden` structure down below) and run the current solver.

 ```rust
 #![allow(incomplete_features)]
 #![feature(generic_const_exprs)]
 extern crate env_logger;
 extern crate helix_snail;
 extern crate num_traits as libnum;

 use helix_snail::nonlinear_solver::*;
 use libnum::{Float, NumAssignOps, NumOps, One, Zero};
 use log::{info, error};
 // This doesn't need to be a global value.
 // I just had it for testing purposes.
 // A value less than or equal to 0 does not log anything
 // Any value greater than 0 will cause logs to be produced
 const LOGGING_LEVEL: i32 = 1;

 struct Broyden<F>
 where
 F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug,
 {
 lambda: F,
 pub logging_level: i32,
 }

 impl<F> NonlinearProblem<F> for Broyden<F>
 where
 F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug,
 {
 const NDIM: usize = 8;
 fn compute_resid_jacobian(&mut self, fcn_eval: &mut [F], opt_jacobian: Option<&mut [F]>, x: &[F]) -> bool {
     assert!(fcn_eval.len() >= Self::NDIM);
     assert!(x.len() >= Self::NDIM);

     let two: F = F::from(2.0).unwrap();
     let three: F = F::from(3.0).unwrap();
     let four: F = F::from(4.0).unwrap();

     if self.logging_level > 0 {
         info!("Evaluating at x = ");
         for i in 0..Self::NDIM {
             info!(" {:?} ", x[i]);
         }
     }

     fcn_eval[0] = (three - two * x[0]) * x[0] - two * x[1] + F::one();
     for i in 1..(Self::NDIM - 1) {
         fcn_eval[i] = (three - two * x[i]) * x[i] - x[i - 1] - two * x[i + 1] + F::one();
     }

     let fcn =
         (three - two * x[Self::NDIM - 1]) * x[Self::NDIM - 1] - x[Self::NDIM - 2] + F::one();

     fcn_eval[Self::NDIM - 1] = (F::one() - self.lambda) * fcn + self.lambda * fcn * fcn;

     if let Some(jacobian) = opt_jacobian {
         assert!(
             jacobian.len() >= Self::NDIM * Self::NDIM,
             "length {:?}",
             jacobian.len()
         );

         for ij in 0..(Self::NDIM * Self::NDIM) {
             jacobian[ij] = F::zero();
         }

         jacobian[0 * Self::NDIM + 0] = three - four * x[0];
         jacobian[0 * Self::NDIM + 1] = -two;
         // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
         for i in 1..(Self::NDIM - 1) {
             jacobian[i * Self::NDIM + i - 1] = -F::one();
             jacobian[i * Self::NDIM + i] = three - four * x[i];
             jacobian[i * Self::NDIM + i + 1] = -two;
         }

         let dfndxn = three - four * x[Self::NDIM - 1];
         // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
         jacobian[(Self::NDIM - 1) * Self::NDIM + (Self::NDIM - 1)] =
             (F::one() - self.lambda) * dfndxn + self.lambda * two * dfndxn * fcn;
         jacobian[(Self::NDIM - 1) * Self::NDIM + (Self::NDIM - 2)] =
             (-F::one() + self.lambda) * F::one() - self.lambda * two * fcn; 
     }

     true
 }
 }

 fn main() {
     let _ = env_logger::builder().is_test(true).try_init();

     let mut broyden = Broyden::<f64> {
         lambda: 0.9999,
         logging_level: LOGGING_LEVEL,
     };

     let dc = TrustRegionDeltaControl::<f64> {
         delta_init: 1.0,
         ..Default::default()
     };

     let mut solver = TrustRegionDoglegSolver::<f64, Broyden<f64>>::new(&dc, &mut broyden);

     for i in 0..Broyden::<f64>::NDIM {
         solver.x[i] = 0.0;
     }

     solver.set_logging_level(Some(LOGGING_LEVEL));
     solver.setup_options(Broyden::<f64>::NDIM * 10, 1e-12, Some(LOGGING_LEVEL));

     let err = solver.solve();

     let status = match err {
         Ok(()) => true,
         Err(e) => {
             error!("Solution did not converge with following error {:?}", e);
             false
         }
     };

     assert!(
         status == true,
         "Solution did not converge"
     );
 }
 ```

 # License

 This crate is licensed under the MIT license.