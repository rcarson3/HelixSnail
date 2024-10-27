use core::result::Result;
use libnum::Float;
use log::info;

use crate::linear_algebra::norm;

/// The DeltaControl trait is used to define what an acceptable step size is in our solution step size
pub trait DeltaControl<F>
where
    F: crate::FloatType,
{
    /// Returns the initial acceptable step size of the solution
    fn get_delta_initial(&self) -> F;
    /// Decreases the acceptable step size
    ///
    /// # Arguments
    /// * `delta` - the acceptable step size
    /// * `norm_full` - the l2 norm of a full step size of the solution
    /// * `took_full` - whether or not a full solution step size was taken
    fn decrease_delta(&self, norm_full: F, took_full: bool, delta: &mut F) -> bool;
    /// Increases the acceptable step size
    ///
    /// # Arguments
    /// * `delta` - the acceptable step size
    fn increase_delta(&self, delta: &mut F);
    /// Updates the acceptable step size of the nonlinear system
    ///
    /// # Arguments
    /// * `delta` - the acceptable step size
    /// * `reject` - whether or not the current solution should be rejected for a given iteration
    /// * `rho` - a normalized ratio between the actual l2 error of the residual and the predicted l2 error of the residual
    /// * `res` - the current iteration l2 norm of the residual
    /// * `res0` - the previous iteration l2 norm of the residual
    /// * `pred_resid` - the predicted l2 norm of the residual
    /// * `norm_full` - the l2 norm of a full step size of the solution
    /// * `took_full` - whether or not a full solution step size was taken
    #[allow(clippy::too_many_arguments)]
    fn update_delta(
        &self,
        res: F,
        res0: F,
        pred_resid: F,
        took_full: bool,
        norm_full: F,
        delta: &mut F,
        reject: &mut bool,
        rho: &mut F,
    ) -> bool;
}

/// This defines the acceptable step size for the solution step based on a trust-region type method
///
/// Additional resources that might be of interest are:
/// Chapter 6 of <https://doi.org/10.1137/1.9781611971200.ch6>
/// or the pseudo-algorithms / code for how to update things in
/// Appendix A of <https://doi.org/10.1137/1.9781611971200.appa>
/// Algorithm A6.4.5 contains variations of the below formulation of things
pub struct TrustRegionDeltaControl<F>
where
    F: crate::FloatType,
{
    pub xi_lg: F,
    pub xi_ug: F,
    pub xi_lo: F,
    pub xi_uo: F,
    pub xi_incr_delta: F,
    pub xi_decr_delta: F,
    pub xi_forced_incr_delta: F,
    /// Initial step size for nonlinear solver
    pub delta_init: F,
    /// Minimum step size for nonlinear solver
    pub delta_min: F,
    /// Maximum step size for nonlinear solver
    pub delta_max: F,
    /// Option to reject solutions that increase residual
    pub reject_resid_increase: bool,
}

impl<F> Default for TrustRegionDeltaControl<F>
where
    F: crate::FloatType,
{
    /// Sane default values for the delta control
    /// One can generally play around with values of xi_* to help their system
    /// potentially converge when dealing with a nasty problem.
    ///
    /// TrustRegionDeltaControl::<F> {
    ///     xi_lg: F::from(0.75).unwrap(),
    ///     xi_ug: F::from(1.4).unwrap(),
    ///     xi_lo: F::from(0.35).unwrap(),
    ///     xi_uo: F::from(5.0).unwrap(),
    ///     xi_incr_delta: F::from(1.5).unwrap(),
    ///     xi_decr_delta: F::from(0.25).unwrap(),
    ///     xi_forced_incr_delta: F::from(1.2).unwrap(),
    ///     delta_init: F::from(1.0).unwrap(),
    ///     delta_min: F::from(1e-12).unwrap(),
    ///     delta_max: F::from(1e4).unwrap(),
    ///     reject_resid_increase: true,
    /// }
    fn default() -> TrustRegionDeltaControl<F> {
        TrustRegionDeltaControl::<F> {
            xi_lg: F::from(0.75).unwrap(),
            xi_ug: F::from(1.4).unwrap(),
            xi_lo: F::from(0.35).unwrap(),
            xi_uo: F::from(5.0).unwrap(),
            xi_incr_delta: F::from(1.5).unwrap(),
            xi_decr_delta: F::from(0.25).unwrap(),
            xi_forced_incr_delta: F::from(1.2).unwrap(),
            delta_init: F::from(1.0).unwrap(),
            delta_min: F::from(1e-12).unwrap(),
            delta_max: F::from(1e4).unwrap(),
            reject_resid_increase: true,
        }
    }
}

impl<F> TrustRegionDeltaControl<F>
where
    F: crate::FloatType,
{
    /// Simple check to ensure that the parameters being used are consistent with one another and usable
    #[allow(dead_code)]
    fn check_params(&self) -> bool {
        !((self.delta_min <= F::zero())
            || (self.delta_max <= self.delta_min)
            || (self.xi_lg <= self.xi_lo)
            || (self.xi_ug >= self.xi_uo)
            || (self.xi_incr_delta <= F::one())
            || ((self.xi_decr_delta >= F::one()) || (self.xi_decr_delta <= F::zero()))
            || (self.xi_forced_incr_delta <= F::zero()))
    }

    /// Updates the acceptable step size of the nonlinear system and also examines
    /// to see if our solution has converged, failed, or our current solution needs
    /// to be rejected and we need to try again with a smaller delta.
    ///
    /// # Arguments
    /// * `residual` - the acceptable step size
    /// * `l2_error_0` - the previous iteration l2 norm of the residual
    /// * `predicted_residual` - the predicted l2 norm of the residual
    /// * `newton_raphson_norm` - the l2 norm of a full step size of the solution
    /// * `tolerance` - the error tolerance of our solution
    /// * `use_newton_raphson`  - whether or not a full solution step size was taken
    /// * `resid_jacob_success` - whether or not the compute residual / jacobian function failed
    /// * `logging_level` - the logging level
    /// * `delta` - the acceptable step size
    /// * `rho_last` - a normalized ratio between the actual l2 error of the residual and the predicted l2 error of the residual
    /// * `l2_error` - the current iteration l2 norm of the residual
    /// * `reject_previous` - whether or not the current solution should be rejected for a given iteration
    pub fn update<const NDIM: usize>(
        &self,
        residual: &[F],
        l2_error_0: F,
        predicted_residual: F,
        newton_raphson_norm: F,
        tolerance: F,
        use_newton_raphson: bool,
        resid_jacob_success: bool,
        logging_level: i32,
        delta: &mut F,
        rho_last: &mut F,
        l2_error: &mut F,
        reject_previous: &mut bool,
    ) -> Result<bool, crate::helix_error::SolverError> {
        if !resid_jacob_success {
            let delta_success = self.decrease_delta(newton_raphson_norm, use_newton_raphson, delta);
            if !delta_success {
                return Err(crate::helix_error::SolverError::DeltaFailure);
            }
            *reject_previous = false;
        } else {
            *l2_error = norm::<{ NDIM }, F>(residual);
            if logging_level > 0 {
                info!("L2_error equals {:?}", *l2_error);
            }
        }

        if *l2_error < tolerance {
            if logging_level > 0 {
                info!("Solution converged");
            }
            return Ok(true);
        }

        {
            let delta_success = self.update_delta(
                *l2_error,
                l2_error_0,
                predicted_residual,
                use_newton_raphson,
                newton_raphson_norm,
                delta,
                reject_previous,
                rho_last,
            );
            if !delta_success {
                return Err(crate::helix_error::SolverError::DeltaFailure);
            }
        }
        Ok(false)
    }
}

impl<F> DeltaControl<F> for TrustRegionDeltaControl<F>
where
    F: crate::FloatType,
{
    fn get_delta_initial(&self) -> F {
        self.delta_init
    }
    fn decrease_delta(&self, norm_full: F, took_full: bool, delta: &mut F) -> bool {
        if took_full {
            *delta = Float::sqrt((*delta) * norm_full * self.xi_decr_delta * self.xi_decr_delta);
        } else {
            *delta *= self.xi_decr_delta;
        }
        if *delta < self.delta_min {
            *delta = self.delta_min;
            false
        } else {
            true
        }
    }
    fn increase_delta(&self, delta: &mut F) {
        *delta *= self.xi_incr_delta;
        if *delta > self.delta_max {
            *delta = self.delta_max;
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn update_delta(
        &self,
        res: F,
        res0: F,
        pred_resid: F,
        took_full: bool,
        norm_full: F,
        delta: &mut F,
        reject: &mut bool,
        rho: &mut F,
    ) -> bool {
        let actual_change = res - res0;
        let pred_change = pred_resid - res0;
        let mut success = true;

        if pred_change == F::zero() {
            if *delta >= self.delta_max {
                return false;
            }
            *delta = Float::min(*delta * self.xi_forced_incr_delta, self.delta_max);
        } else {
            *rho = actual_change / pred_change;
            if (*rho > self.xi_lg) && (actual_change < F::zero()) && (*rho < self.xi_ug) {
                if !took_full {
                    self.increase_delta(delta);
                }
            } else if (*rho < self.xi_lo) || (*rho > self.xi_uo) {
                success = self.decrease_delta(norm_full, took_full, delta);
            }
        }
        *reject = false;
        if (actual_change > F::zero()) && (self.reject_resid_increase) {
            *reject = true;
        }
        success
    }
}
