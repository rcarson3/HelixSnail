pub mod delta_control;
pub mod tr_dogleg_solver;
pub mod dogleg;

pub use self::delta_control::*;
pub use self::tr_dogleg_solver::*;
pub use self::dogleg::*;
use libnum::{Float, Zero, One, NumAssignOps};

/// Status of nonlinear solvers - status can range from converged, failure, and unconverged.
pub enum NonlinearSolverStatus {
    Converged,
    InitialEvalFailure,
    EvalFailure,
    Unconverged,
    DeltaFailure,
    UnconvergedMaxIter,
    SlowJacobian,
    SlowConvergence,
    AlgorithmFailure,
    Unset,
}

/// Nonlinear Solver trait which contains functions that should be shared between solvers
pub trait NonlinearSolver<F, DC>
where F: Float + Zero + One + NumAssignOps,
      DC: DeltaControl<F>
{
    const NDIM: usize;
    fn setup_solver(
        &mut self,
        max_iter: u32,
        tolerance: F,
        delta_control: &DC,
        output_level: Option<i32>,
    );
    fn set_logging_level(output_level: Option<u32>);
    fn solve(&mut self) -> NonlinearSolverStatus;
    fn get_num_fcn_evals(&self) -> u32;
    fn get_solver_rho(&self) -> F;
    fn get_solver_delta(&self) -> F;
    fn get_l2_error(&self) -> F;
    fn compute_residual_jacobian(fcn_eval: &mut [F], jacobian: &mut [F]) -> bool;
}

pub trait NonlinearProblem<F>
where F: Float + Zero + One + NumAssignOps
{
    fn compute_resid_jacobian(
        &mut self,
        fcn_eval: &mut [F],
        jacobian: &mut [F],
        x: &[F],
    ) -> bool;
    const NDIM: usize;
}
