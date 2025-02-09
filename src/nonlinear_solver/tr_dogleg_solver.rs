#![allow(dead_code)]
#![allow(unused_variables)]

use crate::linear_algebra::{dot_prod, lup_solver, mat_t_vec_mult, mat_vec_mult, norm};
use crate::nonlinear_solver::*;

use core::result::Result;
use log::info;

/// This nonlinear solver makes use of a model trust-region method that makes use of a dogleg solver
/// for the sub-problem of the nonlinear problem. It reduces down to taking a full newton raphson step
/// when a given step is near the solution.
pub struct TrustRegionDoglegSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F> + Sized,
    [(); NP::NDIM]: Sized,
{
    /// The field we're solving for. Although, we typically are solving for a scaled version of this in order to have
    /// a numerically stable system of equations.
    pub x: [F; NP::NDIM],
    /// This controls the step size that our solver takes while iterating for a solution
    delta_control: &'a TrustRegionDeltaControl<F>,
    /// The total number of function evaluations our solver took
    function_evals: usize,
    /// The total number of jacobian evaluations our solver took
    jacobian_evals: usize,
    /// The number of iterations it took to solve the nonlinear system
    num_iterations: usize,
    /// The maximum number of iterations we want our solver to take before failing
    max_iterations: usize,
    /// The tolerance we want on our solution
    solution_tolerance: F,
    /// The L2 error of our solution = || F(x) ||_L2
    l2_error: F,
    /// The step size our solver is allowed to take
    delta: F,
    /// The normalized ratio between our predicted error and our actual error
    rho_last: F,
    /// The structure that can calculate the residual / func evaluation and the jacobian of the residual
    crj: &'a mut NP,
    /// The converged status of our nonlinear solve
    converged: bool,
    /// The logging level where any level below 1 is considered off
    logging_level: i32,
}

impl<'a, F, NP> TrustRegionDoglegSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM]: Sized,
{
    /// The size of the jacobian
    const NDIM2: usize = NP::NDIM * NP::NDIM;

    /// Creates a new solver with default values for a number of fields when provided the delta control
    /// and the nonlinear problem structure
    ///
    /// # Arguments:
    /// * `delta_control` - controls the step size that our solver takes while iterating for a solution
    /// * `crj` - Our nonlinear problem which can calculate the residual / func evaluation and the jacobian of the residual
    ///
    /// # Outputs:
    /// * `TrustRegionDoglegSolver::<'a, F, NP>` - a new solver
    pub fn new(
        delta_control: &'a TrustRegionDeltaControl<F>,
        crj: &'a mut NP,
    ) -> TrustRegionDoglegSolver<'a, F, NP> {
        TrustRegionDoglegSolver::<'a, F, NP> {
            x: [F::zero(); NP::NDIM],
            delta_control,
            function_evals: 0,
            jacobian_evals: 0,
            num_iterations: 0,
            max_iterations: NP::NDIM * 1000,
            solution_tolerance: F::from(1e-12).unwrap(),
            l2_error: -F::one(),
            delta: F::from(1e8).unwrap(),
            rho_last: F::one(),
            crj,
            converged: false,
            logging_level: 0,
        }
    }

    /// Computes the newton step for a given iteration
    fn compute_newton_step(
        &self,
        residual: &[F],
        jacobian: &mut [[F; NP::NDIM]],
        newton_step: &mut [F],
    ) -> Result<(), crate::helix_error::SolverError>
    where
        [(); NP::NDIM + 1]: Sized,
    {
        lup_solver(residual, jacobian, newton_step)?;
        for item in newton_step.iter_mut().take(NP::NDIM) {
            *item *= -F::one();
        }
        Ok(())
    }

    /// Rejects the current iterations solution and returns the solution to its previous value
    fn reject(&mut self, delta_x: &[F]) {
        assert!(delta_x.len() >= NP::NDIM);

        for (i_x, item) in delta_x.iter().enumerate().take(NP::NDIM) {
            self.x[i_x] -= *item;
        }
    }
}

impl<F, NP> NonlinearSystemSize for TrustRegionDoglegSolver<'_, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM]: Sized,
{
    const NDIM: usize = NP::NDIM;
}

impl<F, NP> NonlinearSolver<F> for TrustRegionDoglegSolver<'_, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM + 1]: Sized,
{
    fn setup_options(&mut self, max_iter: usize, tolerance: F, output_level: Option<i32>) {
        self.converged = false;
        self.function_evals = 0;
        self.max_iterations = max_iter;
        self.solution_tolerance = tolerance;

        self.logging_level = output_level.unwrap_or_default();
    }
    fn set_logging_level(&mut self, output_level: Option<i32>) {
        self.logging_level = output_level.unwrap_or_default();
    }
    fn solve(&mut self) -> Result<(), crate::helix_error::SolverError> {
        self.converged = false;
        self.num_iterations = 0;
        self.function_evals = 0;
        self.jacobian_evals = 0;

        self.delta = self.delta_control.get_delta_initial();

        if self.logging_level > 0 {
            info!("Initial delta = {:?}", self.delta);
        }

        let mut residual = [F::zero(); NP::NDIM];
        let mut jacobian = [[F::zero(); NP::NDIM]; NP::NDIM];

        if !NonlinearNDSolver::compute_residual_jacobian(self, &mut residual, &mut jacobian) {
            return Err(crate::helix_error::SolverError::InitialEvalFailure);
        }

        self.l2_error = norm::<{ NP::NDIM }, F>(&residual);

        let mut l2_error_0 = self.l2_error;

        if self.logging_level > 0 {
            info!("Initial residual = {:?}", self.l2_error);
        }

        let mut reject_previous = false;

        let mut newton_raphson_step = [F::zero(); NP::NDIM];
        let mut gradient = [F::zero(); NP::NDIM];
        let mut delta_x = [F::zero(); NP::NDIM];
        let mut jacob_grad_2 = F::zero();

        while self.num_iterations < self.max_iterations {
            self.num_iterations += 1;

            if !reject_previous {
                mat_t_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(
                    &jacobian,
                    &residual,
                    &mut gradient,
                );
                let mut temp = [F::zero(); NP::NDIM];
                mat_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(&jacobian, &gradient, &mut temp);
                jacob_grad_2 = dot_prod::<{ NP::NDIM }, F>(&temp, &temp);
                self.compute_newton_step(&residual, &mut jacobian, &mut newton_raphson_step)?;
            }

            let mut predicted_residual = -F::one();
            let mut use_newton_raphson = false;

            let newton_raphson_l2_norm = norm::<{ NP::NDIM }, F>(&newton_raphson_step);

            dogleg::<{ NP::NDIM }, F>(
                self.delta,
                l2_error_0,
                newton_raphson_l2_norm,
                jacob_grad_2,
                &gradient,
                &newton_raphson_step,
                &mut delta_x,
                &mut self.x,
                &mut predicted_residual,
                &mut use_newton_raphson,
            );

            reject_previous = false;

            {
                let resid_jacob_success = NonlinearNDSolver::compute_residual_jacobian(
                    self,
                    &mut residual,
                    &mut jacobian,
                );
                let converged = self.delta_control.update::<{ NP::NDIM }>(
                    &residual,
                    l2_error_0,
                    predicted_residual,
                    newton_raphson_l2_norm,
                    self.solution_tolerance,
                    use_newton_raphson,
                    resid_jacob_success,
                    self.logging_level,
                    &mut self.delta,
                    &mut self.rho_last,
                    &mut self.l2_error,
                    &mut reject_previous,
                )?;
                if converged {
                    self.converged = converged;
                    break;
                }
            }

            if reject_previous {
                if self.logging_level > 0 {
                    info!("Rejecting solution");
                }

                self.l2_error = l2_error_0;
                self.reject(&delta_x);
            }

            l2_error_0 = self.l2_error;
        }

        if !self.converged {
            return Err(crate::helix_error::SolverError::UnconvergedMaxIter);
        }
        Ok(())
    }
    fn get_num_fcn_evals(&self) -> usize {
        self.function_evals
    }
    fn get_num_jacobian_evals(&self) -> usize {
        self.jacobian_evals
    }
    fn get_solver_rho(&self) -> F {
        self.rho_last
    }
    fn get_solver_delta(&self) -> F {
        self.delta
    }
    fn get_l2_error(&self) -> F {
        self.l2_error
    }
}

impl<F, NP> NonlinearNDSolver<F> for TrustRegionDoglegSolver<'_, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM + 1]: Sized,
{
    fn compute_residual_jacobian<const NDIM: usize>(
        &mut self,
        fcn_eval: &mut [F],
        jacobian: &mut [[F; NDIM]],
    ) -> bool {
        assert!(
            NP::NDIM == NDIM,
            "Self::NDIM/NP_NDIM and const NDIMs are not equal..."
        );

        self.function_evals += 1;
        self.jacobian_evals += 1;
        let jac = crate::array2d_to_array1d_mut(jacobian);

        self.crj
            .compute_resid_jacobian(&self.x, fcn_eval, Some(jac))
    }

    fn compute_residual(&mut self, fcn_eval: &mut [F]) -> bool {
        self.function_evals += 1;

        self.crj.compute_resid_jacobian(&self.x, fcn_eval, None)
    }
}
