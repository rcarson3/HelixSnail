#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
extern crate env_logger;
extern crate helix_snail;
extern crate num_traits as libnum;

use helix_snail::nonlinear_solver::*;
use log::{error, info};
// Making use of past here makes it slightly nicer to write the necessary test macro
use paste::paste;

const LOGGING_LEVEL: i32 = 1;

/**
  Comment as in the Trilinos NOX package:

  This test problem is a modified extension of the "Broyden
  Tridiagonal Problem" from Jorge J. More', Burton S. Garbow, and
  Kenneth E. Hillstrom, Testing Unconstrained Optimization Software,
  ACM TOMS, Vol. 7, No. 1, March 1981, pp. 14-41.  The modification
  involves squaring the last equation fn(x) and using it in a
  homotopy-type equation.
  The parameter "lambda" is a homotopy-type parameter that may be
  varied from 0 to 1 to adjust the ill-conditioning of the problem.
  A value of 0 is the original, unmodified problem, while a value of
  1 is that problem with the last equation squared.  Typical values
  for increasingly ill-conditioned problems might be 0.9, 0.99,
  0.999, etc.
  The standard starting point is x(i) = -1, but setting x(i) = 0 tests
  the selected global strategy.
*/
struct Broyden<F>
where
    F: helix_snail::FloatType,
{
    lambda: F,
    pub logging_level: i32,
}

impl<F> NonlinearProblem<F> for Broyden<F>
where
    F: helix_snail::FloatType,
{
    const NDIM: usize = 8;
    fn compute_resid_jacobian<const NDIM: usize>(
        &mut self,
        x: &[F],
        fcn_eval: &mut [F],
        opt_jacobian: &mut Option<&mut [[F; NDIM]]>,
    ) -> bool {
        assert!(
            Self::NDIM == NDIM,
            "Self::NDIM and const NDIMs are not equal..."
        );
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
            assert!(jacobian.len() >= Self::NDIM, "length {:?}", jacobian.len());

            // zero things out first
            for item in jacobian.iter_mut().take(NDIM) {
                for val in item.iter_mut() {
                    *val = F::zero();
                }
            }

            jacobian[0][0] = three - four * x[0];
            jacobian[0][1] = -two;
            // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
            for i in 1..(Self::NDIM - 1) {
                jacobian[i][i - 1] = -F::one();
                jacobian[i][i] = three - four * x[i];
                jacobian[i][i + 1] = -two;
            }

            let dfndxn = three - four * x[Self::NDIM - 1];
            // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
            jacobian[Self::NDIM - 1][Self::NDIM - 1] =
                (F::one() - self.lambda) * dfndxn + self.lambda * two * dfndxn * fcn;
            jacobian[Self::NDIM - 1][Self::NDIM - 2] =
                (-F::one() + self.lambda) * F::one() - self.lambda * two * fcn;
        }

        true
    }
}

/// Test macro for the trust region method that uses a dogleg solver for the subspace
/// for the broyden test problem.
/// Inputs for this are the extended name we want to go with the initial name broyden_tr_dogleg_$name_$type
/// $type is either f32 or f64 for the solver
/// $lambda is the lambda we want the Broyden class to use and be associated with
/// $tolerance is the tolerance for the solver
/// Note: we make use of the paste macro in order to be able to actually append names to the function
/// as this makes our naming convention for things simpler...
macro_rules! broyden_tr_dogleg_tests {
    ($(($name:ident, $type:ident, $lambda:expr, $tolerance:expr),)*) => {
        $(
            paste! {
                #[test]
                fn [< broyden_tr_dogleg_ $name _ $type >]() {
                    let _ = env_logger::builder().is_test(true).try_init();

                    let mut broyden = Broyden::<$type> {
                        lambda: $lambda,
                        logging_level: LOGGING_LEVEL,
                    };

                    let dc = TrustRegionDeltaControl::<$type> {
                        delta_init: 1.0,
                        ..Default::default()
                    };
                    {
                        let mut solver = TrustRegionDoglegSolver::<{Broyden::<$type>::NDIM}, $type, Broyden<$type>>::new(&dc, &mut broyden);

                        for i in 0..Broyden::<$type>::NDIM {
                            solver.x[i] = 0.0;
                        }

                        solver.set_logging_level(Some(LOGGING_LEVEL));
                        solver.setup_options(Broyden::<$type>::NDIM * 10, $tolerance, Some(LOGGING_LEVEL));

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
                }
            }
        )*
    }
}

broyden_tr_dogleg_tests! {
    (lambda_0_9_r4, f64, 0.9999, 1e-12),
    (lambda_0_9_r4, f32, 0.9999, 1e-6),
    (lambda_0_9_r8, f64, 0.99999999, 1e-12),
    (lambda_0_9_r8, f32, 0.99999999, 1e-6),
    (lambda_0_9_r2, f64, 0.99, 1e-12),
    (lambda_0_9_r2, f32, 0.99, 1e-6),
}
