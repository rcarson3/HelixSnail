
use super::*;
pub struct TrustRegionDoglegSolver<NP: NonlinearProblem> 
where [(); NP::NDIM]:
{
    pub m_crj: NP,
    pub x: [f64; NP::NDIM],
}