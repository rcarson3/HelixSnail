#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
extern crate env_logger;
extern crate helix_snail;
extern crate num_traits as libnum;

#[macro_use]
extern crate criterion;

use helix_snail::nonlinear_solver::*;
use helix_snail::linear_algebra::math::*;
use libnum::{Float, NumAssignOps, NumOps, One, Zero};
use log::{info, error};
use criterion::{black_box, Criterion};


const LOGGING_LEVEL: i32 = 0;


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
    fn compute_resid_jacobian<const NDIM: usize>(
        &mut self,
        x: &[F],
        fcn_eval: &mut [F],
        opt_jacobian: Option<&mut [[F; NDIM]]>,
    ) -> bool {
        assert!(Self::NDIM == NDIM, "Self::NDIM and const NDIMs are not equal...");
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
                jacobian.len() >= Self::NDIM,
                "length {:?}",
                jacobian.len()
            );

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
            jacobian[(Self::NDIM - 1)][(Self::NDIM - 1)] =
                (F::one() - self.lambda) * dfndxn + self.lambda * two * dfndxn * fcn;
            jacobian[(Self::NDIM - 1)][(Self::NDIM - 2)] =
                (-F::one() + self.lambda) * F::one() - self.lambda * two * fcn;
        }

        true
    }
}

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

    let mut solver = TrustRegionDoglegSolver::<f32, Broyden<f32>>::new(&dc, &mut broyden);

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

    assert!(
        status == true,
        "Solution did not converge"
    );
}

pub fn outer_prod_2<const NDIM: usize, const MDIM: usize, F>(vec1: &[F], vec2: &[F], matrix: &mut [[F; MDIM]])
where
    F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug,
{
    assert!(matrix.len() >= NDIM);
    assert!(vec1.len() >= MDIM);
    assert!(vec2.len() >= NDIM);

    for i_n in 0..NDIM {
        for j_m in 0..MDIM {
            matrix[i_n][j_m] = vec1[i_n] * vec2[j_m];
        }
    }
}

fn nl_solver_wrapper(c: &mut Criterion) {
    let mut group = c.benchmark_group("nl_solver");

    group.sample_size(500);
    group.bench_function("broyden_bench_f64", |b| {
        b.iter(|| broyden_bench_f64())
    });
    group.bench_function("broyden_bench_f32", |b| {
        b.iter(|| broyden_bench_f32())
    });
    group.finish();
}

fn math_wrapper(c: &mut Criterion) {
    let mut group = c.benchmark_group("math");

    const NDIM: usize = 12;

    let mut vec1: [f64; NDIM] = [0.0; NDIM];
    let mut vec2: [f64; NDIM] = [0.0; NDIM];
    let mut matrix2: [[f64; NDIM]; NDIM] = [[0.0; NDIM]; NDIM];

    for i in 0..NDIM {
        vec1[i] = i as f64 + 1.0_f64;
        vec2[i] = i as f64;
    }

    group.sample_size(500);
    group.bench_function("outer_prod_f64", |b| {
        b.iter(|| outer_prod::<{NDIM}, {NDIM}, f64>(black_box(&vec1), black_box(&vec2), black_box(&mut matrix2)))
    });

    group.finish();
}

criterion_group!(benches, nl_solver_wrapper, math_wrapper);
criterion_main!(benches);