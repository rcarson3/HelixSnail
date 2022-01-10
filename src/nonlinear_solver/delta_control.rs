// Make use of the num-traits::Float to enable us to easily swap between std and no_std float operations
// We could also maybe eventually look into even making the solvers generic, so users could use
// either f32 or f64 types.
use libnum::{Float, Zero, One, NumAssignOps, NumOps};
pub trait DeltaControl<F>
where F: Float + Zero + One + NumAssignOps + NumOps
{
    fn get_delta_initial(&self) -> F;
    fn decrease_delta(&self, delta: &mut F, norm_full: F, took_full: bool) -> bool;
    fn increase_delta(&self, delta: &mut F);
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

pub struct TrustRegionDeltaControl<F>
where F: Float + Zero + One + NumAssignOps + NumOps
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
where F: Float + Zero + One + NumAssignOps + NumOps
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
where F: Float + Zero + One + NumAssignOps + NumOps
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
