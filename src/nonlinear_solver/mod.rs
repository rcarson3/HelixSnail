
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

/// Nonlinear Solver trait which contains functions that should be shared between solvers
pub trait NonlinearSolver{
    const NDIM: usize;
    fn setup_solver(&mut self, max_iter: u32, tolerance: f64, delta_control: &TrustRegionDeltaControl, output_level: Option<i32>);
    fn set_logging_level(output_level: Option<u32>);
    fn solve(&mut self) -> NonlinearSolverStatus;
    fn get_num_fcn_evals(&self) -> u32;
    fn get_solver_rho(&self) -> f64;
    fn get_solver_delta(&self) -> f64;
    fn get_l2_error(&self) -> f64;
    fn compute_residual_jacobian(fcn_eval: &mut [f64], jacobian: &mut [f64]) -> bool;
}

pub trait NonlinearProblem {
    fn compute_resid_jacobian(&mut self, fcn_eval: &mut [f64], jacobian: &mut [f64], x: &[f64]) -> bool;
    const NDIM: usize;
}

pub struct TrustRegionDoglegSolver<NP: NonlinearProblem> 
where [(); NP::NDIM]:
{
    pub m_crj: NP,
    pub x: [f64; NP::NDIM],
}