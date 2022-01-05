pub trait DeltaControl {
    fn get_delta_initial(&self) -> f64;
    fn decrease_delta(&self, delta: &mut f64, norm_full: f64, took_full: bool) -> bool;
    fn increase_delta(&self, delta: &mut f64) -> bool;
    #[allow(clippy::too_many_arguments)]
    fn update_delta(&self, delta: &mut f64, reject: &mut bool, rho: &mut f64, res: f64, res0: f64, pred_resid: f64, took_full: bool) -> bool;
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
}

impl TrustRegionDeltaControl {
    #[allow(dead_code)]
    fn check_params(&self) -> bool { 
        !(( self.delta_min <= 0.0_f64 ) ||
        ( self.delta_max <= self.delta_min ) ||
        ( self.xi_lg <= self.xi_lo ) ||
        ( self.xi_ug >= self.xi_uo ) ||
        ( self.xi_incr_delta <= 1.0_f64 ) ||
        ( ( self.xi_decr_delta >= 1.0_f64 ) || ( self.xi_decr_delta <= 0.0_f64 ) ) ||
        ( self.xi_forced_incr_delta <= 1.0_f64 ))
    }
}