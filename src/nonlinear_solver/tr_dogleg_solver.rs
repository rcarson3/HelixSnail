#![allow(dead_code)]
#![allow(unused_variables)]

use super::*;
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
    crj: NP,
    status: NonlinearSolverStatus,
}

impl<'a, F, NP> TrustRegionDoglegSolver<'a, F, NP>
where
    F: Float + Zero + One + NumAssignOps,
    NP: NonlinearProblem<F>,
    [(); NP::NDIM]:,
{
    const NDIM2: usize = NP::NDIM * NP::NDIM;
}

impl<'a, F, DC, NP> NonlinearSolver<F, DC> for TrustRegionDoglegSolver<'a, F, NP>
where
    F: Float + Zero + One + NumAssignOps,
    DC: DeltaControl<F>,
    NP: NonlinearProblem<F>,
    [(); NP::NDIM]:,
{
    const NDIM: usize = NP::NDIM;
    fn setup_solver(
        &mut self,
        max_iter: u32,
        tolerance: F,
        delta_control: &DC,
        output_level: Option<i32>,
    ) {
    }
    fn set_logging_level(output_level: Option<u32>) {}
    fn solve(&mut self) -> NonlinearSolverStatus {
        NonlinearSolverStatus::Unconverged
    }
    fn get_num_fcn_evals(&self) -> u32 {
        0
    }
    fn get_solver_rho(&self) -> F {
        F::zero()
    }
    fn get_solver_delta(&self) -> F {
        F::zero()
    }
    fn get_l2_error(&self) -> F {
        F::zero()
    }
    fn compute_residual_jacobian(fcn_eval: &mut [F], jacobian: &mut [F]) -> bool {
        false
    }
}
