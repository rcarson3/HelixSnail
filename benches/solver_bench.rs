#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
extern crate divan;
extern crate env_logger;
extern crate helix_snail;
extern crate num_traits as libnum;

use divan::black_box;
use helix_snail::linear_algebra::math::*;
use helix_snail::nonlinear_solver::*;
use log::{error, info};

const LOGGING_LEVEL: i32 = 0;

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

#[divan::bench_group (
    min_time = 1.0,
    max_time = 5.0, // seconds
    sample_size = 500,
    sample_count = 1000,
)]
mod nonlinear_solver {
    use crate::*;

    #[divan::bench(name = "broyden_f64")]
    fn broyden_bench_f64() {
        let _ = env_logger::builder().is_test(true).try_init();

        let mut broyden = Broyden::<f64> {
            lambda: 0.9999,
            logging_level: LOGGING_LEVEL,
        };

        let dc = TrustRegionDeltaControl::<f64> {
            delta_init: 1.0,
            ..Default::default()
        };

        let mut solver =
            TrustRegionDoglegSolver::<{ Broyden::<f64>::NDIM }, f64, Broyden<f64>>::new(
                &dc,
                &mut broyden,
            );

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

        assert!(status == true, "Solution did not converge");
    }

    #[divan::bench(name = "broyden_f32")]
    fn broyden_bench_f32() {
        let _ = env_logger::builder().is_test(true).try_init();

        let mut broyden = Broyden::<f32> {
            lambda: 0.9999,
            logging_level: LOGGING_LEVEL,
        };

        let dc = TrustRegionDeltaControl::<f32> {
            delta_init: 1.0,
            ..Default::default()
        };

        let mut solver =
            TrustRegionDoglegSolver::<{ Broyden::<f32>::NDIM }, f32, Broyden<f32>>::new(
                &dc,
                &mut broyden,
            );

        for i in 0..Broyden::<f32>::NDIM {
            solver.x[i] = 0.0;
        }

        solver.set_logging_level(Some(LOGGING_LEVEL));
        solver.setup_options(Broyden::<f32>::NDIM * 10, 1e-6, Some(LOGGING_LEVEL));

        let err = solver.solve();

        let status = match err {
            Ok(()) => true,
            Err(e) => {
                error!("Solution did not converge with following error {:?}", e);
                false
            }
        };

        assert!(status == true, "Solution did not converge");
    }
}

#[divan::bench_group (
    min_time = 1.0,
    max_time = 5.0, // seconds
    sample_size = 500,
    sample_count = 1000,
)]
mod math {
    use crate::*;
    const NDIM: usize = 12;
    #[divan::bench(name = "outer_prod_instantiation")]
    fn outer_prod_instantiation() -> ([f64; NDIM], [f64; NDIM], [[f64; NDIM]; NDIM]) {
        let mut vec1: [f64; NDIM] = [0.0; NDIM];
        let mut vec2: [f64; NDIM] = [0.0; NDIM];
        let matrix2: [[f64; NDIM]; NDIM] = [[0.0; NDIM]; NDIM];

        for i in 0..NDIM {
            vec1[i] = i as f64 + 1.0_f64;
            vec2[i] = i as f64;
        }
        (vec1, vec2, matrix2)
    }

    #[divan::bench(name = "outer_prod_whole")]
    fn outer_prod_local() {
        let mut vec1: [f64; NDIM] = [0.0; NDIM];
        let mut vec2: [f64; NDIM] = [0.0; NDIM];
        let mut matrix2: [[f64; NDIM]; NDIM] = [[0.0; NDIM]; NDIM];

        for i in 0..NDIM {
            vec1[i] = i as f64 + 1.0_f64;
            vec2[i] = i as f64;
        }
        outer_prod::<{ NDIM }, { NDIM }, f64>(
            black_box(&vec1),
            black_box(&vec2),
            black_box(&mut matrix2),
        )
    }
}

fn main() {
    divan::main();
}
