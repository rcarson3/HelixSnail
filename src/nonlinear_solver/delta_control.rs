use libnum::{Float, NumAssignOps, NumOps, One, Zero};

/// The DeltaControl trait is used to define what an acceptable step size is in our solution step size
pub trait DeltaControl<F>
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    /// Returns the initial acceptable step size of the solution
    fn get_delta_initial(&self) -> F;
    /// Decreases the acceptable step size
    /// delta: the acceptable step size
    /// norm_full: the l2 norm of a full step size of the solution
    /// took_full: whether or not a full solution step size was taken
    fn decrease_delta(&self, delta: &mut F, norm_full: F, took_full: bool) -> bool;
    /// Increases the acceptable step size
    /// delta: the acceptable step size
    fn increase_delta(&self, delta: &mut F);
    /// Updates the acceptable step size of the nonlinear system
    /// delta: the acceptable step size
    /// reject: whether or not the current solution should be rejected for a given iteration
    /// rho: a normalized ratio between the actual l2 error of the residual and the predicted l2 error of the residual
    /// res: the current iteration l2 norm of the residual
    /// res0: the previous iteration l2 norm of the residual
    /// pred_resid: the predicted l2 norm of the residual
    /// norm_full: the l2 norm of a full step size of the solution
    /// took_full: whether or not a full solution step size was taken
    #[allow(clippy::too_many_arguments)]
    fn update_delta(
        &self,
        delta: &mut F,
        reject: &mut bool,
        rho: &mut F,
        res: F,
        res0: F,
        pred_resid: F,
        took_full: bool,
        norm_full: F,
    ) -> bool;
}

/// This defines the acceptable step size for the solution step based on a trust-region type method
/// Additional resources that might be of interest are:
/// Chapter 6 of https://doi.org/10.1137/1.9781611971200.ch6
/// or the pseudo-algorithms / code for how to update things in
/// Appendix A of https://doi.org/10.1137/1.9781611971200.appa
/// Algorithm A6.4.5 contains variations of the below formulation of things
pub struct TrustRegionDeltaControl<F>
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    pub xi_lg: F,
    pub xi_ug: F,
    pub xi_lo: F,
    pub xi_uo: F,
    pub xi_incr_delta: F,
    pub xi_decr_delta: F,
    pub xi_forced_incr_delta: F,
    pub delta_init: F,
    pub delta_min: F,
    pub delta_max: F,
    pub reject_resid_increase: bool,
}

impl<F> TrustRegionDeltaControl<F>
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
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
}

impl<F> DeltaControl<F> for TrustRegionDeltaControl<F>
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    fn get_delta_initial(&self) -> F {
        self.delta_init
    }
    fn decrease_delta(&self, delta: &mut F, norm_full: F, took_full: bool) -> bool {
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
        delta: &mut F,
        reject: &mut bool,
        rho: &mut F,
        res: F,
        res0: F,
        pred_resid: F,
        took_full: bool,
        norm_full: F,
    ) -> bool {
        let actual_change = res - res0;
        let pred_change = pred_resid - res0;
        let mut success = false;

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
            } else if (*rho < self.xi_lo) && (*rho > self.xi_uo) {
                success = self.decrease_delta(delta, norm_full, took_full);
            }
        }
        *reject = false;

        if (actual_change > F::zero()) && (self.reject_resid_increase) {
            *reject = true;
        }
        success
    }
}
