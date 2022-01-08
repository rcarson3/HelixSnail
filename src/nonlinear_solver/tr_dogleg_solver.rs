use super::*;
pub struct TrustRegionDoglegSolver<F, NP: NonlinearProblem<F>>
where
    F: Float + Zero + One + NumAssignOps,
    [(); NP::NDIM]:,
{
    pub m_crj: NP,
    pub x: [F; NP::NDIM],
}
