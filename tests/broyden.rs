#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
extern crate env_logger;
extern crate helix_snail;
extern crate num_traits as libnum;

use helix_snail::nonlinear_solver::*;
use libnum::{Float, NumAssignOps, NumOps, One, Zero};
use log::info;

const LOGGING_LEVEL: i32 = 1;

struct Broyden<F>
where
    F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug + core::convert::From<f64>,
{
    lambda: F,
    pub logging_level: i32,
}

impl<F> NonlinearProblem<F> for Broyden<F>
where
    F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug + core::convert::From<f64>,
    f64: Into<F>,
{
    const NDIM: usize = 8;
    fn compute_resid_jacobian(&mut self, fcn_eval: &mut [F], jacobian: &mut [F], x: &[F]) -> bool {
        assert!(fcn_eval.len() >= Self::NDIM);
        assert!(
            jacobian.len() >= Self::NDIM * Self::NDIM,
            "length {:?}",
            jacobian.len()
        );
        assert!(x.len() >= Self::NDIM);

        if self.logging_level > 0 {
            info!("Evaluating at x = ");
            for i in 0..Self::NDIM {
                info!(" {:?} ", x[i]);
            }
        }

        for ij in 0..(Self::NDIM * Self::NDIM) {
            jacobian[ij] = F::zero();
        }

        fcn_eval[0] = (3.0.into() - 2.0.into() * x[0]) * x[0] - 2.0.into() * x[1] + F::one();
        for i in 1..(Self::NDIM - 1) {
            fcn_eval[i] =
                (3.0.into() - 2.0.into() * x[i]) * x[i] - x[i - 1] - 2.0.into() * x[i + 1]
                    + F::one();
        }

        let fcn = (3.0.into() - 2.0.into() * x[Self::NDIM - 1]) * x[Self::NDIM - 1]
            - x[Self::NDIM - 2]
            + F::one();

        fcn_eval[Self::NDIM - 1] = (F::one() - self.lambda) * fcn + self.lambda * fcn * fcn;

        jacobian[0 * Self::NDIM + 0] = 3.0.into() - 4.0.into() * x[0];
        jacobian[0 * Self::NDIM + 1] = -2.0.into();
        // F(i) = (3-2*x[i])*x[i] - x[i-1] - 2*x[i+1] + 1;
        for i in 1..(Self::NDIM - 1) {
            jacobian[i * Self::NDIM + i - 1] = -F::one();
            jacobian[i * Self::NDIM + i] = 3.0.into() - 4.0.into() * x[i];
            jacobian[i * Self::NDIM + i + 1] = -2.0.into();
        }

        let dfndxn = 3.0.into() - 4.0.into() * x[Self::NDIM - 1];
        // F(n-1) = ((3-2*x[n-1])*x[n-1] - x[n-2] + 1)^2;
        jacobian[(Self::NDIM - 1) * Self::NDIM + (Self::NDIM - 1)] =
            (F::one() - self.lambda) * dfndxn + self.lambda * 2.0.into() * dfndxn * fcn;
        jacobian[(Self::NDIM - 1) * Self::NDIM + (Self::NDIM - 2)] =
            (-F::one() + self.lambda) * F::one() - self.lambda * 2.0.into() * fcn;

        true
    }
}

#[test]
fn broyden_a() {
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

    let status = solver.solve();

    assert!(
        status == NonlinearSolverStatus::Converged,
        "Solution did not converge"
    );
}

#[test]
fn broyden_b() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut broyden = Broyden::<f64> {
        lambda: 0.99999999,
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

    let status = solver.solve();

    assert!(
        status == NonlinearSolverStatus::Converged,
        "Solution did not converge"
    );
}

#[test]
fn broyden_c() {
    let _ = env_logger::builder().is_test(true).try_init();

    let mut broyden = Broyden::<f64> {
        lambda: 0.99,
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

    let status = solver.solve();

    assert!(
        status == NonlinearSolverStatus::Converged,
        "Solution did not converge"
    );
}
