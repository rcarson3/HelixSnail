#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
extern crate env_logger;
extern crate helix_snail;
extern crate num_traits as libnum;

use helix_snail::nonlinear_solver::*;
use log::{error, info};

struct NewtonProblem<F>
where
    F: helix_snail::FloatType,
{
    func: fn(&F, &mut F, Option<&mut F>) -> bool,
}

impl<F> NonlinearSystemSize for NewtonProblem<F>
where
    F: helix_snail::FloatType,
{
    const NDIM: usize = 1;
}

impl<F> Nonlinear1DProblem<F> for NewtonProblem<F>
where
    F: helix_snail::FloatType,
{
    #[inline(always)]
    fn compute_resid_jacobian(
        &mut self,
        x: &F,
        fcn_eval: &mut F,
        opt_jacobian: Option<&mut F>,
    ) -> bool {
        (self.func)(x, fcn_eval, opt_jacobian)
    }
}

fn newton_func_a<F>(tol: F)
where
    F: helix_snail::FloatType,
{
    let _ = env_logger::builder().is_test(true).try_init();

    let lambda = |x: &F, fcn_eval: &mut F, opt_jacobian: Option<&mut F>| -> bool {
        let alpha = F::from(5.0).unwrap();
        let x_sol = F::from(2.345).unwrap();
        let arg = alpha * (*x - x_sol);
        *fcn_eval = arg.tanh();
        if let Some(jac) = opt_jacobian {
            let temp = F::one() / arg.cosh();
            *jac = alpha * temp * temp;
        }
        true
    };

    let mut np = NewtonProblem::<F> {
        func: lambda,
    };

    let mut newton_bb = NewtonBisectionBracketedSolver::<F, NewtonProblem<F>>::new(false, &mut np);

    newton_bb.x = F::zero();
    newton_bb.set_bounds(F::zero(), F::zero());

    let err = newton_bb.solve();

    info!("Function evaluations: {:?}", &newton_bb.get_num_fcn_evals());
    info!("Final residual: {:?}", &newton_bb.get_l2_error());
    info!("Final solution: {:?}", &newton_bb.x);

    let status = match err {
        Ok(()) => true,
        Err(e) => {
            error!("Solution did not converge with following error {:?}", e);
            false
        }
    };
    let x_sol = F::from(2.345).unwrap();
    assert!(newton_bb.x - x_sol <= tol, "Expected solution is outside of expected bounds");

    assert!(
        status == true,
        "Solution did not converge"
    );
}

fn newton_func_xsinx<F>(tol: F)
where
    F: helix_snail::FloatType,
{
    let _ = env_logger::builder().is_test(true).try_init();

    let lambda = |x: &F, fcn_eval: &mut F, opt_jacobian: Option<&mut F>| -> bool {
        let level = F::from(0.75).unwrap();
        *fcn_eval = *x * x.sin() - level;
        if let Some(jac) = opt_jacobian {
            *jac = x.sin() + *x * x.cos();
        }
        true
    };

    let mut np = NewtonProblem::<F> {
        func: lambda,
    };

    let mut newton_bb = NewtonBisectionBracketedSolver::<F, NewtonProblem<F>>::new(false, &mut np);

    newton_bb.x = F::zero();
    newton_bb.set_bounds(F::zero(), F::zero());

    let err = newton_bb.solve();

    info!("Function evaluations: {:?}", &newton_bb.get_num_fcn_evals());
    info!("Final residual: {:?}", &newton_bb.get_l2_error());
    info!("Final solution: {:?}", &newton_bb.x);

    let status = match err {
        Ok(()) => true,
        Err(e) => {
            error!("Solution did not converge with following error {:?}", e);
            false
        }
    };

    let x_sol = F::from(-0.933308).unwrap();
    assert!(newton_bb.x - x_sol <= tol, "Expected solution is outside of expected bounds");

    assert!(
        status == true,
        "Solution did not converge"
    );
}

#[test]
fn newton_func_xsinx_f32() {
    let tol = 1e-6_f32;
    newton_func_xsinx(tol);
}

#[test]
fn newton_func_xsinx_f64() {
    let tol = 1e-7_f32;
    newton_func_xsinx(tol);
}

#[test]
fn newton_func_a_f32() {
    let tol = 1e-6_f32;
    newton_func_a(tol);
}

#[test]
fn newton_func_a_f64() {
    let tol = 1e-7_f32;
    newton_func_a(tol);
}