/// The defines what an acceptable step size is in our solution step size
pub mod delta_control;
/// The dogleg is an approach to solving the update step of a nonlinear system of equations.
pub mod dogleg;
/// A dogleg solver that makes use of a trust-region method
pub mod tr_dogleg_solver;

pub mod hybrid_tr_dogleg_solver;

pub mod newton_bb_solver;

pub use self::delta_control::*;
pub use self::dogleg::*;
pub use self::hybrid_tr_dogleg_solver::*;
pub use self::newton_bb_solver::*;
pub use self::tr_dogleg_solver::*;

pub trait NonlinearSystemSize {
    /// Size of nonlinear system of equations which should be consistent with nonlinear problem
    const NDIM: usize;
}

/// Nonlinear Solver trait which contains functions that should be shared between solvers. These solvers currently
/// expect a square system of equations in order to work.
pub trait NonlinearSolver<F>: NonlinearSystemSize
where
    F: crate::FloatType,
{
    /// Values required to setup solver
    ///
    /// # Arguments
    /// * `max_iter` - maximum number of iterations for the solver
    /// * `tolerance` - solution tolerance to be considered converged
    /// * `output_level` - optional parameters which controls whether or not logging should occur
    fn setup_options(&mut self, max_iter: usize, tolerance: F, output_level: Option<i32>);
    /// Sets a logging level if applicable to solver
    ///
    /// # Arguments
    /// * `output_level` - optional parameters which controls the level of logging within solver
    fn set_logging_level(&mut self, output_level: Option<i32>);
    /// Solves the nonlinear system of equations
    ///
    /// # Outputs
    /// * Nonlinear solver status which can be used to probe the success of the solve
    fn solve(&mut self) -> Result<(), crate::helix_error::SolverError>;
    /// Get the number of function evals of the system
    ///
    /// # Outputs
    /// * The number of function evals
    fn get_num_fcn_evals(&self) -> usize;
    /// Get the number of jacobian evals of the system
    ///
    /// # Outputs
    /// * The number of jacobian evals
    fn get_num_jacobian_evals(&self) -> usize;
    /// Gets the current solver's normalized ratio between the actual l2 error of the residual
    /// and the predicted l2 error of the residual
    ///
    /// This value can be useful for debugging the convergence of the nonlinear problem
    ///
    /// # Outputs
    /// * normalized ratio between the actual l2 error of the residual and the predicted l2 error of the residual
    fn get_solver_rho(&self) -> F;
    /// Gets the solver's acceptable step size change allowed by the delta control object
    ///
    /// # Outputs
    /// * acceptable step size change allowed by the delta control object
    fn get_solver_delta(&self) -> F;
    /// Gets the current l2 norm of the residual of the nonlinear problem
    ///
    /// # Outputs
    /// * l2 norm of the residual of the nonlinear problem
    fn get_l2_error(&self) -> F;
}

/// Nonlinear Solver trait which contains functions that should be shared between solvers. These solvers currently
/// expect a square system of equations in order to work.
pub trait NonlinearNDSolver<F>: NonlinearSolver<F>
where
    F: crate::FloatType,
{
    /// Computes the residual and jacobian of the nonlinear problem
    ///
    /// # Arguments
    /// * `fcn_eval` - the residual / function evaluation of the nonlinear problem
    /// * `jacobian` - the derivative of the residual with respect to the solution variable
    ///
    /// # Outputs
    /// * whether the nonlinear problem was able to successfully to evaluate these quantities with the current solution
    /// Note that nightly currently isn't flexible enough for us to have this be jacobian: &mut [[F; Self::NDIM]]
    /// so we revert to this instead... where NDIM = Self::NDIM in practice
    fn compute_residual_jacobian<const NDIM: usize>(
        &mut self,
        fcn_eval: &mut [F],
        jacobian: &mut [[F; NDIM]],
    ) -> bool;

    /// Computes the residual and jacobian of the nonlinear problem
    ///
    /// # Arguments
    /// * `fcn_eval` - the residual / function evaluation of the nonlinear problem
    ///
    /// # Outputs
    /// * whether the nonlinear problem was able to successfully to evaluate these quantities with the current solution
    fn compute_residual(&mut self, fcn_eval: &mut [F]) -> bool;
}

/// Nonlinear Solver trait which contains functions that should be shared between solvers. These solvers currently
/// expect a square system of equations in order to work.
pub trait Nonlinear1DSolver<F>: NonlinearSolver<F>
where
    F: crate::FloatType,
{
    /// Computes the residual and jacobian of the nonlinear problem
    ///
    /// # Arguments
    /// * `fcn_eval` - the residual / function evaluation of the nonlinear problem
    /// * `jacobian` - the derivative of the residual with respect to the solution variable
    ///
    /// # Outputs
    /// * whether the nonlinear problem was able to successfully to evaluate these quantities with the current solution
    /// Note that nightly currently isn't flexible enough for us to have this be jacobian: &mut [[F; Self::NDIM]]
    /// so we revert to this instead... where NDIM = Self::NDIM in practice
    fn compute_residual_jacobian(&mut self, fcn_eval: &mut F, jacobian: &mut F) -> bool;

    /// Computes the residual and jacobian of the nonlinear problem
    ///
    /// # Arguments
    /// * `fcn_eval` - the residual / function evaluation of the nonlinear problem
    ///
    /// # Outputs
    /// * whether the nonlinear problem was able to successfully to evaluate these quantities with the current solution
    fn compute_residual(&mut self, fcn_eval: &mut F) -> bool;
}

/// Nonlinear problems must implement the following trait in-order to be useable within this crates solvers
pub trait NonlinearNDProblem<F>: NonlinearSystemSize
where
    F: crate::FloatType,
{
    /// This function at a minimum computes the residual / function evaluation of the system of nonlinear equations
    /// that we are solving for. It is expected that fcn_eval and opt_jacobian have been scaled such that the solution
    /// variable x nominally remains in the neighborhood of [-1, 1] as this provides better numerical stability of
    /// the solution.
    ///
    /// # Arguments
    /// * fcn_eval - the residual / function evaluation of the nonlinear problem: size NDIM
    /// * opt_jacobian - (Optional) the derivative of the residual with respect to the solution variable: size NDIM * NDIM
    ///                   For the solvers within this library, it is expected that jacobian is provided back to us if we pass in a slice
    ///                   as we don't make use of finite difference methods to estimate the jacobian.
    /// Note that nightly currently isn't flexible enough for us to have this be jacobian: Option(&mut [[F; Self::NDIM]])
    /// so we revert to this instead... where NDIM = Self::NDIM in practice
    fn compute_resid_jacobian(
        &mut self,
        x: &[F],
        fcn_eval: &mut [F],
        opt_jacobian: Option<&mut [F]>,
    ) -> bool;
}

/// Nonlinear problems must implement the following trait in-order to be useable within this crates solvers
pub trait Nonlinear1DProblem<F>: NonlinearSystemSize
where
    F: crate::FloatType,
{
    /// This function at a minimum computes the residual / function evaluation of the system of nonlinear equations
    /// that we are solving for. It is expected that fcn_eval and opt_jacobian have been scaled such that the solution
    /// variable x nominally remains in the neighborhood of [-1, 1] as this provides better numerical stability of
    /// the solution.
    ///
    /// # Arguments
    /// * fcn_eval - the residual / function evaluation of the nonlinear problem: size NDIM
    /// * opt_jacobian - (Optional) the derivative of the residual with respect to the solution variable: size NDIM * NDIM
    ///                   For the solvers within this library, it is expected that jacobian is provided back to us if we pass in a slice
    ///                   as we don't make use of finite difference methods to estimate the jacobian.
    /// Note that nightly currently isn't flexible enough for us to have this be jacobian: Option(&mut [[F; Self::NDIM]])
    /// so we revert to this instead... where NDIM = Self::NDIM in practice
    fn compute_resid_jacobian(
        &mut self,
        x: &F,
        fcn_eval: &mut F,
        opt_jacobian: Option<&mut F>,
    ) -> bool;
}
