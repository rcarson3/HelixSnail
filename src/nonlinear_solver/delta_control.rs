use libm;
pub trait DeltaControl {
    fn get_delta_initial(&self) -> f64;
    fn decrease_delta(&self, delta: &mut f64, norm_full: f64, took_full: bool) -> bool;
    fn increase_delta(&self, delta: &mut f64);
    #[allow(clippy::too_many_arguments)]
    fn update_delta(
        &self,
        delta: &mut f64,
        reject: &mut bool,
        rho: &mut f64,
        res: f64,
        res0: f64,
        pred_resid: f64,
        took_full: bool,
        norm_full: f64,
    ) -> bool;
}

pub struct TrustRegionDeltaControl {
    pub xi_lg: f64,
    pub xi_ug: f64,
    pub xi_lo: f64,
    pub xi_uo: f64,
    pub xi_incr_delta: f64,
    pub xi_decr_delta: f64,
    pub xi_forced_incr_delta: f64,
    pub delta_init: f64,
    pub delta_min: f64,
    pub delta_max: f64,
    pub reject_resid_increase: bool,
}

impl TrustRegionDeltaControl {
    #[allow(dead_code)]
    fn check_params(&self) -> bool {
        !((self.delta_min <= 0.0_f64)
            || (self.delta_max <= self.delta_min)
            || (self.xi_lg <= self.xi_lo)
            || (self.xi_ug >= self.xi_uo)
            || (self.xi_incr_delta <= 1.0_f64)
            || ((self.xi_decr_delta >= 1.0_f64) || (self.xi_decr_delta <= 0.0_f64))
            || (self.xi_forced_incr_delta <= 1.0_f64))
    }
}

impl DeltaControl for TrustRegionDeltaControl {
    fn get_delta_initial(&self) -> f64 {
        self.delta_init
    }
    fn decrease_delta(&self, delta: &mut f64, norm_full: f64, took_full: bool) -> bool {
        if took_full {
            *delta = libm::sqrt((*delta) * norm_full * self.xi_decr_delta * self.xi_decr_delta);
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
    fn increase_delta(&self, delta: &mut f64) {
        *delta *= self.xi_incr_delta;
        if *delta > self.delta_max {
            *delta = self.delta_max;
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn update_delta(
        &self,
        delta: &mut f64,
        reject: &mut bool,
        rho: &mut f64,
        res: f64,
        res0: f64,
        pred_resid: f64,
        took_full: bool,
        norm_full: f64,
    ) -> bool {
        let actual_change = res - res0;
        let pred_change = pred_resid - res0;
        let mut success = false;

        if pred_change == 0.0 {
            if *delta >= self.delta_max {
                return false;
            }
            *delta = libm::fmin(*delta * self.xi_forced_incr_delta, self.delta_max);
        } else {
            *rho = actual_change / pred_change;
            if (*rho > self.xi_lg) && (actual_change < 0.0_f64) && (*rho < self.xi_ug) {
                if !took_full {
                    self.increase_delta(delta);
                }
            } else if (*rho < self.xi_lo) && (*rho > self.xi_uo) {
                success = self.decrease_delta(delta, norm_full, took_full);
            }
        }
        *reject = false;

        if (actual_change > 0.0_f64) && (self.reject_resid_increase) {
            *reject = true;
        }
        success
    }
}
