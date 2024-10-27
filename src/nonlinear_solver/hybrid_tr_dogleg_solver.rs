#![allow(dead_code)]
#![allow(unused_variables)]

use crate::linear_algebra::{
    dot_prod, householder_qr, make_givens, mat_t_vec_mult, norm, qr_solve,
    upper_tri_mat_t_vec_mult, upper_tri_mat_vec_mult,
};
use crate::nonlinear_solver::*;

use core::result::Result;
use log::info;

/// This nonlinear solver makes use of a model trust-region method that makes use of a dogleg solver
/// for the sub-problem of the nonlinear problem. It reduces down to taking a full newton raphson step
/// when a given step is near the solution.
// A hybrid trust region type solver, dogleg approximation
// for dense general Jacobian matrix that makes use of a rank-1 update of the jacobian
// using QR factorization
// Method is inspired by SNLS current trust region dogleg solver, Powell's original hybrid method for
// nonlinear equations, and MINPACK's modified version of it.
// Powell's original hybrid method can be found at:
// M. J. D. Powell, "A hybrid method for nonlinear equations", in Numerical methods for nonlinear algebraic equations,
// Philip Rabinowitz, editor, chapter 6, pages 87-114, Gordon and Breach Science Publishers, New York, 1970.
// MINPACK's user guide is found at https://doi.org/10.2172/6997568
pub struct HybridTRDglSolver<'a, F, NP>
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
    /// The number of consecutive successive iterations
    num_consecutive_s_iterations: usize,
    /// The number of consecutive failed iterations
    num_consecutive_f_iterations: usize,
    /// The number of slow iterations high
    num_slow_1_iterations: usize,
    /// The number of slow iterations low
    num_slow_2_iterations: usize,
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

impl<'a, F, NP> HybridTRDglSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM - 1]: Sized,
    [(); NP::NDIM]: Sized,
    [(); NP::NDIM + 1]: Sized,
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
    ) -> HybridTRDglSolver<'a, F, NP> {
        HybridTRDglSolver::<'a, F, NP> {
            x: [F::zero(); NP::NDIM],
            delta_control,
            function_evals: 0,
            jacobian_evals: 0,
            num_iterations: 0,
            max_iterations: NP::NDIM * 1000,
            num_consecutive_s_iterations: 0,
            num_consecutive_f_iterations: 0,
            num_slow_1_iterations: 0,
            num_slow_2_iterations: 0,
            solution_tolerance: F::from(1e-12).unwrap(),
            l2_error: -F::one(),
            delta: F::from(1e8).unwrap(),
            rho_last: F::zero(),
            crj,
            converged: false,
            logging_level: 0,
        }
    }

    /// Computes the newton step for a given iteration
    fn compute_newton_step(
        &self,
        func: &[F],
        r_matrix: &[[F; NP::NDIM]],
        x: &mut [F],
    ) -> Result<(), crate::helix_error::SolverError>
    where
        [(); NP::NDIM + 1]: Sized,
    {
        assert!(x.len() >= NP::NDIM);
        qr_solve::<{ NP::NDIM }, F>(r_matrix, func, x)?;
        for i in 0..NP::NDIM {
            x[i] *= -F::one();
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

    fn solve_initialization(
        &mut self,
        residual: &mut [F],
    ) -> Result<(), crate::helix_error::SolverError> {
        assert!(residual.len() >= NP::NDIM);

        self.converged = false;
        self.function_evals = 0;
        self.jacobian_evals = 0;

        let resid_jacob_success = NonlinearNDSolver::compute_residual(self, residual);

        if !resid_jacob_success {
            return Err(crate::helix_error::SolverError::InitialEvalFailure);
        }

        self.l2_error = norm::<{ NP::NDIM }, F>(residual);

        if self.logging_level > 0 {
            info!("Initial residual = {:?}", self.l2_error);
        }

        self.delta = self.delta_control.get_delta_initial();

        if self.logging_level > 0 {
            info!("Initial delta = {:?}", self.delta);
        }

        // initialize iteration counter and monitors
        self.num_iterations = 1;
        self.num_consecutive_f_iterations = 0;
        self.num_consecutive_s_iterations = 0;
        self.num_slow_1_iterations = 0;
        self.num_slow_2_iterations = 0;

        Ok(())
    }

    fn solve_step(
        &mut self,
        residual: &mut [F],
        jacobian: &mut [[F; NP::NDIM]],
        q_matrix: &mut [[F; NP::NDIM]],
        qtf: &mut [F],
        grad: &mut [F],
        newton_raphson_step: &mut [F],
        delta_x: &mut [F],
    ) -> Result<(), crate::helix_error::SolverError> {
        assert!(residual.len() >= NP::NDIM);
        assert!(jacobian.len() >= NP::NDIM);
        assert!(q_matrix.len() >= NP::NDIM);
        assert!(qtf.len() >= NP::NDIM);
        assert!(grad.len() >= NP::NDIM);
        assert!(newton_raphson_step.len() >= NP::NDIM);
        assert!(delta_x.len() >= NP::NDIM);

        let mut jacob_eval = true;
        let resid_jacob_success =
            NonlinearNDSolver::compute_residual_jacobian(self, residual, jacobian);

        // If this fails our solver is in trouble and needs to die.
        if !resid_jacob_success {
            return Err(crate::helix_error::SolverError::EvalFailure);
        }
        // Jacobian is our R matrix and Q
        // could re-use nrstep, grad, and delx given if we're already here than we need to reset our solver
        // so these arrays can be used as scratch arrays.
        householder_qr::<{ NP::NDIM }, F>(jacobian, q_matrix, grad, newton_raphson_step, delta_x);
        // Nothing crazy here as qtf = Q^T * residual
        mat_t_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(q_matrix, residual, qtf);
        // we're essentially starting over here so we can reset these values
        let mut reject_previous = false;
        let mut jacob_grad_2 = F::zero();
        // self.l2_error is set initially in solve_initialization
        // and later on in delta_control.update_delta, so it's always set
        let mut l2_error_0 = self.l2_error;

        loop {
            // This is done outside this step so that these operations can be done with varying solve
            // techniques such as LU/QR or etc...
            if !reject_previous {
                // So the LU solve does things in-place which causes issues when calculating the grad term...
                // So, we need to pull this out and perform this operation first
                // R^T * Q^T * f
                upper_tri_mat_t_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(jacobian, qtf, grad);
                {
                    let mut temp = [F::zero(); NP::NDIM];
                    upper_tri_mat_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(
                        jacobian, grad, &mut temp,
                    );
                    jacob_grad_2 = dot_prod::<{ NP::NDIM }, F>(&temp, &temp);
                }
                // R x = Q^T f solve
                // If R is signular we fail out of the solve with an error on the CPU
                // On the GPU, the fail just prints out and  doesn't abort anything so
                // we return this signal notifying us of the failure which can then be passed
                // onto the other libraries / application codes using SNLS.
                self.compute_newton_step(qtf, jacobian, newton_raphson_step)?;
            }

            let mut use_newton_raphson = false;
            // If the step was rejected nrStep will be the same value as previously, and so we can just recalculate nr_norm here.
            let newton_raphson_l2_norm = norm::<{ NP::NDIM }, F>(newton_raphson_step);

            let mut predicted_residual = F::one();
            // computes the updated delta x, predicated residual error, and whether or not NR method was used.
            dogleg::<{ NP::NDIM }, F>(
                self.delta,
                l2_error_0,
                newton_raphson_l2_norm,
                jacob_grad_2,
                grad,
                newton_raphson_step,
                delta_x,
                &mut self.x,
                &mut predicted_residual,
                &mut use_newton_raphson,
            );
            reject_previous = false;

            let resid_jacob_success = NonlinearNDSolver::compute_residual(self, residual);

            let converged = self.delta_control.update::<{ NP::NDIM }>(
                residual,
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

            if reject_previous {
                if self.logging_level > 0 {
                    info!("Rejecting solution");
                }

                self.l2_error = l2_error_0;
                self.reject(delta_x);
            }

            // Look at a relative reduction in residual to see if convergence is slow
            let actual_reduction = F::one() - self.l2_error / l2_error_0;
            // Delta has been updated from a bounds already
            // Check to see if we need to recalculate jacobian
            self.num_consecutive_f_iterations = if self.rho_last < F::from(0.1).unwrap() {
                self.num_consecutive_f_iterations + 1
            } else {
                0
            };
            if self.num_consecutive_f_iterations == 2 {
                return Ok(());
            }

            // Determine the progress of the iteration
            self.num_slow_1_iterations = if actual_reduction >= F::from(0.001).unwrap() {
                0
            } else {
                self.num_slow_1_iterations + 1
            };
            self.num_slow_2_iterations = if jacob_eval && (actual_reduction < F::from(0.1).unwrap())
            {
                self.num_slow_2_iterations + 1
            } else {
                0
            };

            // Tests for termination and stringent tolerances
            if self.num_slow_2_iterations == 5 {
                return Err(crate::helix_error::SolverError::SlowJacobian);
            }
            if self.num_slow_1_iterations == 10 {
                return Err(crate::helix_error::SolverError::SlowConvergence);
            }

            // Only calculate this if solution wasn't rejected
            if !reject_previous {
                // Here we can use delx, nrStep, and grad as working arrays as we're just going
                // to rewrite them in a second...
                // nrStep = (R * delx + Q^T * f_i)
                upper_tri_mat_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(
                    jacobian,
                    delta_x,
                    newton_raphson_step,
                );
                for i in 0..NP::NDIM {
                    // work_arr3 = R \Delta x + Q^T F
                    // where F here is our residual vector
                    newton_raphson_step[i] += qtf[i];
                }
                // calculate the rank one modification to the jacobian
                // and update qtf if necessary
                {
                    // delx = delx / ||delx||_L2
                    let inv_delta_x_norm = F::one() / norm::<{ NP::NDIM }, F>(delta_x);
                    for i in 0..NP::NDIM {
                        delta_x[i] *= inv_delta_x_norm;
                    }

                    mat_t_vec_mult::<{ NP::NDIM }, { NP::NDIM }, F>(&q_matrix, residual, grad);

                    // Update qtf value first and then we can update the gradient term
                    // grad = (Q^T * f_{i+1} - Q^T * f_i - R * delx)
                    for i in 0..NP::NDIM {
                        qtf[i] = grad[i];
                        grad[i] = (grad[i] - newton_raphson_step[i]) * inv_delta_x_norm;
                    }
                }

                // compute the qr factorization of the updated jacobian
                self.rank1_update(delta_x, grad, jacobian, q_matrix, qtf, newton_raphson_step)?;
            }

            l2_error_0 = self.l2_error;
            jacob_eval = false;

            self.num_iterations += 1;
            if self.num_iterations > self.max_iterations {
                break;
            }
        }

        Ok(())
    }

    // This performs a Broyden Rank-1 style update for Q, R and Q^T f
    // This version has origins in this paper:
    // Gill, Philip E., et al. "Methods for modifying matrix factorizations." Mathematics of computation 28.126 (1974): 505-535.
    // However, you can generally find it described in more approachable manners elsewhere on the internet
    //
    // The Broyden update method is described in:
    // Broyden, Charles G. "A class of methods for solving nonlinear simultaneous equations." Mathematics of computation 19.92 (1965): 577-593.
    // Additional resources that might  be of interest are:
    // Chapter 8 of https://doi.org/10.1137/1.9781611971200.ch8
    // or the pseudo-algorithms / code for how to update things in
    // Appendix A of https://doi.org/10.1137/1.9781611971200.appa
    fn rank1_update(
        &mut self,
        delta_x_normalized: &[F],    // delta x / || delta_x||_L2
        delta_residual_vector: &[F], // (Q^T * f_{i+1} - Q^T * f_i - R * \Delta x)
        r_matrix: &mut [[F; NP::NDIM]],
        q_matrix: &mut [[F; NP::NDIM]],
        qtf: &mut [F],
        residual_vector: &mut [F], // (R * \Delta x + Q^T * f_i)
    ) -> Result<(), crate::helix_error::SolverError> {
        assert!(delta_x_normalized.len() >= NP::NDIM);
        assert!(delta_residual_vector.len() >= NP::NDIM);
        assert!(r_matrix.len() >= NP::NDIM);
        assert!(q_matrix.len() >= NP::NDIM);
        assert!(qtf.len() >= NP::NDIM);
        assert!(residual_vector.len() >= NP::NDIM);

        let ndim1 = NP::NDIM - 1;
        residual_vector[ndim1] = r_matrix[ndim1][ndim1];
        let mut delta_residual_vector_n1 = delta_residual_vector[ndim1];
        let mut givens = [F::zero(); 2];

        // Rotate the vector (Q^T * f_{i+1} - Q^T * f_i - R * \Delta x) into a multiple of the n-th unit vector in
        // such a way that a spike is introduced into (R * \Delta x + Q^T * f_i)
        for i in (0..(NP::NDIM - 1)).rev() {
            residual_vector[i] = F::zero();
            if delta_residual_vector_n1 != F::zero() {
                // Determine a givens rotation which eliminates the information
                // necessary to recover the givens rotation
                make_givens(
                    -delta_residual_vector_n1,
                    delta_residual_vector[i],
                    &mut givens,
                );
                delta_residual_vector_n1 =
                    givens[1] * delta_residual_vector[i] + givens[0] * delta_residual_vector_n1;
                // Apply the transformation to R and extend the spike in (R * \Delta x + Q^T * f_i)
                for j in i..NP::NDIM {
                    let rmat_val = givens[0] * r_matrix[i][j] - givens[1] * residual_vector[j];
                    residual_vector[j] =
                        givens[1] * r_matrix[i][j] + givens[0] * residual_vector[j];
                    r_matrix[i][j] = rmat_val;
                }
            } else {
                givens[0] = F::one();
                givens[1] = F::zero();
            }

            // 1st updates of Q and Q^T f
            for j in 0..NP::NDIM {
                let qmat_val = givens[0] * q_matrix[j][i] - givens[1] * q_matrix[j][ndim1];
                q_matrix[j][ndim1] = givens[1] * q_matrix[j][i] + givens[0] * q_matrix[j][ndim1];
                q_matrix[j][i] = qmat_val;
            }

            {
                let qtf_val = givens[0] * qtf[i] - givens[1] * qtf[ndim1];
                qtf[ndim1] = givens[1] * qtf[i] + givens[0] * qtf[ndim1];
                qtf[i] = qtf_val;
            }
        }

        // Add the spike from the Rank-1 update to (R * \Delta x + Q^T * f_i)
        for i in 0..NP::NDIM {
            residual_vector[i] += delta_residual_vector_n1 * delta_x_normalized[i];
        }

        // Eliminate the spike
        for i in 0..(NP::NDIM - 1) {
            if residual_vector[i] != F::zero() {
                // Determine a givens rotation which eliminates the i-th element of the spike
                make_givens(-r_matrix[i][i], residual_vector[i], &mut givens);
                // Apply the transformation to R and reduce the spike in (R * \Delta x + Q^T * f_i)
                for j in i..NP::NDIM {
                    let rmat_val = givens[0] * r_matrix[i][j] + givens[1] * residual_vector[j];
                    residual_vector[j] =
                        -givens[1] * r_matrix[i][j] + givens[0] * residual_vector[j];
                    r_matrix[i][j] = rmat_val;
                }
            } else {
                givens[0] = F::one();
                givens[1] = F::zero();
            }
            // Test for zero diagonal in output
            if r_matrix[i][i] == F::zero() {
                return Err(crate::helix_error::SolverError::AlgorithmFailure);
            }
            // 2nd update of Q and Q^T F
            for j in 0..NP::NDIM {
                let qmat_val = givens[0] * q_matrix[j][i] + givens[1] * q_matrix[j][ndim1];
                q_matrix[j][ndim1] = -givens[1] * q_matrix[j][i] + givens[0] * q_matrix[j][ndim1];
                q_matrix[j][i] = qmat_val;
            }

            {
                let qtf_val = givens[0] * qtf[i] + givens[1] * qtf[ndim1];
                qtf[ndim1] = -givens[1] * qtf[i] + givens[0] * qtf[ndim1];
                qtf[i] = qtf_val;
            }
        }

        // Move (R * \Delta x + Q^T * f_i) back into the last column of the output R
        r_matrix[ndim1][ndim1] = residual_vector[ndim1];

        Ok(())
    }
}

impl<'a, F, NP> NonlinearSystemSize for HybridTRDglSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM]: Sized,
{
    const NDIM: usize = NP::NDIM;
}

impl<'a, F, NP> NonlinearSolver<F> for HybridTRDglSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM - 1]: Sized,
    [(); NP::NDIM]: Sized,
    [(); NP::NDIM + 1]: Sized,
{
    fn setup_options(&mut self, max_iter: usize, tolerance: F, output_level: Option<i32>) {
        self.converged = false;
        self.function_evals = 0;
        self.max_iterations = max_iter;
        self.solution_tolerance = tolerance;

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
        let mut residual = [F::zero(); NP::NDIM];
        let mut jacobian = [[F::zero(); NP::NDIM]; NP::NDIM];

        // Working arrays
        let mut q_matrix = [[F::zero(); NP::NDIM]; NP::NDIM];
        let mut work_array_1 = [F::zero(); NP::NDIM];
        let mut work_array_2 = [F::zero(); NP::NDIM];
        let mut work_array_3 = [F::zero(); NP::NDIM];
        let mut qtf = [F::zero(); NP::NDIM];

        // Do initial solver checks
        self.solve_initialization(&mut residual)?;
        // Run our solver until it converges or fails
        while !self.converged
            && (self.num_iterations < self.max_iterations)
            && (self.function_evals < self.max_iterations)
        {
            self.solve_step(
                &mut residual,
                &mut jacobian,
                &mut q_matrix,
                &mut qtf,
                &mut work_array_1,
                &mut work_array_2,
                &mut work_array_3,
            )?;
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

impl<'a, F, NP> NonlinearNDSolver<F> for HybridTRDglSolver<'a, F, NP>
where
    F: crate::FloatType,
    NP: NonlinearNDProblem<F>,
    [(); NP::NDIM - 1]: Sized,
    [(); NP::NDIM]: Sized,
    [(); NP::NDIM + 1]: Sized,
{
    fn compute_residual_jacobian<const NDIM: usize>(
        &mut self,
        fcn_eval: &mut [F],
        jacobian: &mut [[F; NDIM]],
    ) -> bool {
        self.function_evals += 1;
        self.jacobian_evals += 1;

        assert!(
            NP::NDIM == NDIM,
            "Self::NDIM/NP_NDIM and const NDIMs are not equal..."
        );
        let jac = crate::array2d_to_array1d_mut(jacobian);
        self.crj
            .compute_resid_jacobian(&self.x, fcn_eval, Some(jac))
    }

    fn compute_residual(&mut self, fcn_eval: &mut [F]) -> bool {
        self.function_evals += 1;
        self.crj.compute_resid_jacobian(&self.x, fcn_eval, None)
    }
}
