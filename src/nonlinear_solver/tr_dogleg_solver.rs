#![allow(dead_code)]
#![allow(unused_variables)]

use crate::nonlinear_solver::*;

/// This nonlinear solver makes use of a model trust-region method that makes use of a dogleg solver
/// for the sub-problem of the nonlinear problem. It reduces down to taking a full newton raphson step
/// when a given step is near the solution.
pub struct TrustRegionDoglegSolver<'a, F, NP: NonlinearProblem<F>>
where
    F: Float + Zero + One + NumAssignOps + NumOps,
    [(); NP::NDIM]:,
{
    pub x: [F; NP::NDIM],
    delta_control: &'a TrustRegionDeltaControl<F>,
    function_evals: usize,
    jacobian_evals: usize,
    num_iterations: usize,
    max_iterations: usize,
    solution_tolerance: F,
    l2_error: F,
    delta: F,
    rho_last: F,
    crj: &'a mut NP,
    status: NonlinearSolverStatus,
    logging_level: i32,
}

impl<'a, F, NP> TrustRegionDoglegSolver<'a, F, NP>
where
    F: Float + Zero + One + NumAssignOps,
    NP: NonlinearProblem<F>,
    [(); NP::NDIM]:,
{
    const NDIM2: usize = NP::NDIM * NP::NDIM;

    pub fn new(
        delta_control: &'a TrustRegionDeltaControl<F>,
        crj: &'a mut NP,
    ) -> TrustRegionDoglegSolver<'a, F, NP>
    where
        f64: Into<F>,
    {
        TrustRegionDoglegSolver::<'a, F, NP> {
            x: [F::zero(); NP::NDIM],
            delta_control: delta_control,
            function_evals: 0,
            jacobian_evals: 0,
            num_iterations: 0,
            max_iterations: NP::NDIM * 1000,
            solution_tolerance: 1e-12.into(),
            l2_error: -1.0.into(),
            delta: 1e8.into(),
            rho_last: F::one(),
            crj: crj,
            status: NonlinearSolverStatus::Unconverged,
            logging_level: 0,
        }
    }

    fn solver_step(&mut self) {}
}

impl<'a, F, NP> NonlinearSolver<F> for TrustRegionDoglegSolver<'a, F, NP>
where
    F: Float + Zero + One + NumAssignOps,
    NP: NonlinearProblem<F>,
    [(); NP::NDIM]:,
{
    const NDIM: usize = NP::NDIM;
    fn setup_options(&mut self, max_iter: usize, tolerance: F, output_level: Option<i32>) {
        self.status = NonlinearSolverStatus::Unconverged;
        self.function_evals = 0;
        self.max_iterations = max_iter;

        self.logging_level = if let Some(output) = output_level {
            output
        } else {
            0
        };
    }
    fn set_logging_level(output_level: Option<u32>) {}
    fn solve(&mut self) -> NonlinearSolverStatus {
        NonlinearSolverStatus::Unconverged
    }
    fn get_num_fcn_evals(&self) -> usize {
        self.num_iterations
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
    fn compute_residual_jacobian(&mut self, fcn_eval: &mut [F], jacobian: &mut [F]) -> bool {
        self.crj.compute_resid_jacobian(fcn_eval, jacobian, &self.x)
    }
}
