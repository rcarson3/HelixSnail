#![allow(dead_code)]
#![allow(unused_variables)]

use crate::nonlinear_solver::*;

use core::result::Result;
use log::info;

/// This nonlinear solver makes use of a model trust-region method that makes use of a dogleg solver
/// for the sub-problem of the nonlinear problem. It reduces down to taking a full newton raphson step
/// when a given step is near the solution.
pub struct NewtonBisectionBracketedSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: Nonlinear1DProblem<F> + Sized,
{
    /// The field we're solving for. Although, we typically are solving for a scaled version of this in order to have
    /// a numerically stable system of equations.
    pub x: F,
    /// Whether our solver is bracketed or not
    bracket: bool,
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
    /// The tolerance for our x variable
    x_tolerance: F,
    /// The L2 error of our solution = || F(x) ||_L2
    l2_error: F,
    /// The structure that can calculate the residual / func evaluation and the jacobian of the residual
    crj: &'a mut NP,
    /// The converged status of our nonlinear solve
    converged: bool,
    /// The logging level where any level below 1 is considered off
    logging_level: i32,
    /// The lower bracketed bound to use with the solver
    x_lower: F,
    /// The upper bracketed bound to use with the solver
    x_upper: F,
}

impl<'a, F, NP> NewtonBisectionBracketedSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: Nonlinear1DProblem<F>,
{
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
        bracket: bool,
        crj: &'a mut NP,
    ) -> NewtonBisectionBracketedSolver<'a, F, NP> {
        NewtonBisectionBracketedSolver::<'a, F, NP> {
            x: F::zero(),
            bracket,
            function_evals: 0,
            jacobian_evals: 0,
            num_iterations: 0,
            max_iterations: NP::NDIM * 50,
            solution_tolerance: F::from(1e-8).unwrap(),
            x_tolerance: F::from(1e-12).unwrap(),
            l2_error: -F::one(),
            crj,
            converged: false,
            logging_level: 0,
            x_lower: F::zero(),
            x_upper: F::zero(),
        }
    }
    pub fn set_x_tolerance(&mut self, x_tolerance: F) {
        self.x_tolerance = x_tolerance;
    }
    pub fn set_bounds(&mut self, x_lower: F, x_upper: F) {
        self.x_lower = x_lower;
        self.x_upper = x_upper;
    }
    /** find bounds for zero of a function for which x does not have limits ;
    *
    * xl and xh need to be set as inputs, fl and fh do not
    *
    * on exit value of true, fl and fh are consistent with xl and xh
    */
    pub fn calculate_bounds(&mut self, func_lower: &mut F, func_upper: &mut F) -> bool {
        if self.x_lower > self.x_upper {
            core::mem::swap(&mut self.x_lower, &mut self.x_upper);
        }

        let mut delta_upper = {
            let delta_x = self.x_upper - self.x_lower;
            let max_x = F::one().max(self.x_lower.abs());
            F::from(0.2).unwrap() * delta_x.max(max_x)
        };

        let mut delta_lower = delta_upper;
        let mut success: bool;
      
        let mut delta_x_upper_init;
        let mut delta_x_lower_init;

        {
            let mut jacob_upper = F::zero();
            let mut jacob_lower = F::zero();
            success = self.crj
            .compute_resid_jacobian(&self.x_upper, func_upper, &mut Some(&mut jacob_upper));
            self.function_evals += 1;
            if !success {
                return false;
            }
            success = self.crj
            .compute_resid_jacobian(&self.x_lower, func_lower, &mut Some(&mut jacob_lower));
            self.function_evals += 1;

            if (func_upper.abs() < self.solution_tolerance) || (func_lower.abs() < self.solution_tolerance) ||
                (*func_lower * *func_upper) < F::zero() {
                return true;
            }

            delta_x_upper_init = -jacob_upper / *func_upper;
            delta_x_lower_init = -jacob_lower / *func_lower;

        }

        let mut new_upper = false;
        let mut x_upper_prev = self.x_upper;
        let mut x_lower_prev = self.x_lower;

        let bound_step_growth_factor = F::from(1.2).unwrap();
        let bound_overshoot_factor= F::from(1.2).unwrap();

         // Create a lambda function to deal with the bound updates portion of things as
         // updating the lower bounds was exactly the same as doing it for the upper bounds code
         // wise...
         let mut bound_function = |success: &mut bool, func: &mut F, x_value: &mut F, x_prev: &mut F,delta: &mut F, delta_x: &mut F, x_other: &mut F, func_other: &mut F| -> bool {
            let mut jacob = F::zero();
            let func_prev = *func;
            *success = self.crj.compute_resid_jacobian(x_value, func, &mut Some(&mut jacob));
            self.function_evals += 1;
            if self.logging_level > 0 {
                info!("NewtonBB in bounding, have x, f, J : {:?}, {:?}, {:?}", &x_value, &func, &jacob);
            }

            if !(*success) {
                // try a smaller step
                *x_value = *x_prev;
                *func = func_prev;
                // delta_x is as was before
                *delta *= F::from(0.1).unwrap();
                if self.logging_level > 0 {
                    info!("NewtonBB trouble in bounding, cut delta back to = {:?}", &delta);
                }
                return false;
             }
             if func.abs() < self.solution_tolerance {
                return true;
             }
             if func_prev * *func < F::zero() {
                // bracketed
                *x_other = *x_prev;
                *func_other = func_prev;
                return true;
             }
             *delta_x = -jacob / *func;
             *delta *= bound_step_growth_factor;
             return false;
         };

         for i in 0..self.max_iterations {
            // the ordering here biases the search toward exploring smaller x values
            //
            if (i < 10) && (delta_x_lower_init < F::zero()) {
                x_lower_prev = self.x_lower;
                self.x_lower += (-delta_lower).max( bound_overshoot_factor / delta_x_lower_init);
                new_upper = false;
            }
            else if (i < 10) && (delta_x_upper_init > F::zero()) {
                x_upper_prev = self.x_upper;
                self.x_upper += (-delta_upper).max( bound_overshoot_factor / delta_x_upper_init);
                new_upper = true;
            }
            else {
                // take turns
                if new_upper { 
                    x_lower_prev = self.x_lower;
                    self.x_lower -= delta_lower;
                    new_upper = false;
                }
                else {
                    x_upper_prev = self.x_upper;
                    self.x_upper -= delta_upper;
                    new_upper = true;
                }           
            }
            if new_upper {
                let return_val = bound_function(&mut success, func_upper, &mut self.x_upper, &mut x_upper_prev, &mut delta_upper, &mut delta_x_upper_init, &mut self.x_lower, func_lower);
                if return_val {
                    return true; 
                }
                if !success { 
                    continue;
                }
             }
             else {
                let return_val = bound_function(&mut success, func_lower, &mut self.x_lower, &mut x_lower_prev, &mut delta_lower, &mut delta_x_lower_init, &mut self.x_upper, func_upper);
                if return_val {
                    return true; 
                }
                if !success { 
                    continue;
                }
             }
         }
        false
    }
}

impl<'a, F, NP> NonlinearSystemSize for NewtonBisectionBracketedSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: Nonlinear1DProblem<F>,
{
    const NDIM: usize = NP::NDIM;
}

impl<'a, F, NP> NonlinearSolver<F>
    for NewtonBisectionBracketedSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: Nonlinear1DProblem<F>,
{
    fn setup_options(&mut self, max_iter: usize, tolerance: F, output_level: Option<i32>) {
        self.converged = false;
        self.function_evals = 0;
        self.max_iterations = max_iter;
        self.solution_tolerance = tolerance;
        self.x_tolerance = tolerance * F::from(1e-4).unwrap();

        self.logging_level = if let Some(output) = output_level {
            output
        } else {
            0
        };
    }
    fn set_logging_level(&mut self, output_level: Option<i32>) {
        self.logging_level = if let Some(output) = output_level {
            output
        } else {
            0
        };
    }

    fn solve(&mut self) -> Result<(), crate::helix_error::SolverError> {

        self.converged = false;

        let mut func = F::zero();
        let mut jacobian = F::zero();

        if !Nonlinear1DSolver::compute_residual_jacobian(self, &mut func, &mut  jacobian) {
            return Err(crate::helix_error::SolverError::InitialEvalFailure);
        }

        if func.abs() < self.solution_tolerance {
            self.l2_error = func.abs();
            self.converged = true;
            return Ok(());
        }

        let mut func_lower = F::zero();
        let mut func_upper = F::zero();
        {
            if self.crj.compute_resid_jacobian(&self.x_lower, &mut func_lower, &mut None) {
                return Err(crate::helix_error::SolverError::InitialEvalFailure);
            }
            if func_lower.abs() < self.solution_tolerance {
                self.x = self.x_lower;
                self.l2_error = func_lower.abs();
                self.converged = true;
                if self.logging_level > 0 {
                    info!("Converged with f(x): {:?} at x: {:?}", &func_lower, &self.x);
                }
                return Ok(());
            }
            if self.crj.compute_resid_jacobian(&self.x_upper, &mut func_upper, &mut None) {
                return Err(crate::helix_error::SolverError::InitialEvalFailure);
            }
            if func_upper.abs() < self.solution_tolerance {
                self.x = self.x_upper;
                self.l2_error = func_upper.abs();
                self.converged = true;
                if self.logging_level > 0 {
                    info!("Converged with f(x): {:?} at x: {:?}", &func_upper, &self.x);
                }
                return Ok(());
            }
            self.function_evals += 2;
        }

        if (func_lower * func_upper) > F::zero() {
            if !self.bracket {
                if !self.calculate_bounds(&mut func_lower, &mut func_upper) {
                    return Err(crate::helix_error::SolverError::AlgorithmFailure)
                }
            }
            else {
                return Err(crate::helix_error::SolverError::AlgorithmFailure)
            }
        }
        // We want x_lower to correspond to the point at which f(x_lower) < 0
        if func_lower > F::zero() {
            core::mem::swap(&mut self.x_lower, &mut self.x_upper);
            core::mem::swap(&mut func_lower, &mut func_upper);
        }

        let mut delta_x_old = (self.x_upper - self.x_lower).abs();
        let mut delta_x = delta_x_old;

        if (func < F::zero()) && (func > func_lower) {
            self.x_lower = self.x;
        }
        else if (func > F::zero()) && (func < func_upper) {
            self.x_upper = self.x;
        }
        //     main loop over a given number of iterations. checks whether
        //     extrapolated value using the gradient for newton iteration 
        //     is beyond the bounds and either uses the Newton estimate of
        //     bisection depending on outcome. convergence is checked on 
        //     the value of the function, closeness of variable to either
        //     limit and change of variable over an iteration.

        for i in 0..self.max_iterations {
            let conditional1 = ((self.x - self.x_upper) * jacobian - func) * ((self.x - self.x_lower) * jacobian - func) >= F::zero();
            let conditional2 = (F::from(2.0).unwrap() * func).abs() > (delta_x_old * jacobian).abs();
            if conditional1 || conditional2 {
                delta_x_old = delta_x;
                delta_x = (self.x_upper - self.x_lower) * F::from(0.5).unwrap();
                self.x = self.x_lower + delta_x;
            }
            else {
                delta_x_old = delta_x;
                delta_x = -func / jacobian;
                self.x += delta_x;
            }
            if self.logging_level > 0 {
                info!("NewtonBB doing bisection with dx: {:?}", &delta_x);
            }

            if delta_x.abs() < self.x_tolerance && i > 10 {
                //
                // could additionally check (fabs(x) > _tolx) && (fabs(dx) / fabs(x) < _tol) for convergence
                // but that may in some cases be too sloppy
                //
                if self.logging_level > 0 {
                    info!("Converged by bracketing with dx: {:?}, f(x): {:?}, x: {:?}", &delta_x, &func, self.x);
                }
                self.l2_error = func.abs();
                self.converged = true;                
                return Ok(());
            }
            if !Nonlinear1DSolver::compute_residual_jacobian(self, &mut func, &mut  jacobian) {
                return Err(crate::helix_error::SolverError::EvalFailure);
            }
            if self.logging_level > 0 {
                info!("NewtonBB evaluation with x, f, J : {:?}, {:?}, {:?}", &self.x, &func, &jacobian);
            }
            if func.abs() < self.solution_tolerance {
                if self.logging_level > 0 {
                    info!("Converged with f(x): {:?} at x: {:?}", &func, &self.x);
                }
                self.l2_error = func.abs();
                self.converged = true;                
                return Ok(());                
            }
            if func < F::zero() {
                self.x_lower = self.x;
             }
             else {
                self.x_upper = self.x;
             }
        }
        return Err(crate::helix_error::SolverError::UnconvergedMaxIter);
    }
    fn get_num_fcn_evals(&self) -> usize {
        self.function_evals
    }
    fn get_num_jacobian_evals(&self) -> usize {
        self.jacobian_evals
    }
    fn get_solver_rho(&self) -> F {
        F::one()
    }
    fn get_solver_delta(&self) -> F {
        F::one()
    }
    fn get_l2_error(&self) -> F {
        self.l2_error
    }
}

impl<'a, F, NP> Nonlinear1DSolver<F>
    for NewtonBisectionBracketedSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: Nonlinear1DProblem<F>,
{
    fn compute_residual_jacobian(
        &mut self,
        fcn_eval: &mut F,
        jacobian: &mut F,
    ) -> bool {

        self.function_evals += 1;
        self.jacobian_evals += 1;

        self.crj
            .compute_resid_jacobian(&self.x, fcn_eval, &mut Some(jacobian))
    }

    fn compute_residual(
        &mut self,
        fcn_eval: &mut F,
    ) -> bool {
        self.function_evals += 1;
        self.crj
            .compute_resid_jacobian(&self.x, fcn_eval, &mut None)
    }
}