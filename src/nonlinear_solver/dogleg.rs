use super::*;

fn dogleg<const NDIM: usize, F>(delta: F,
                           res_0: F,
                           nr_norm: F,
                           jacobian2_gradient: F,
                           gradient: &[F],
                           newton_raphson_step: &[F],
                           delta_x: & mut [F],
                           x: & mut [F],
                           predicted_residual: & mut F,
                           use_newton_raphson: & mut bool)
where F: Float + Zero + One + NumAssignOps,
{
    assert!(gradient.len() >= NDIM);
    assert!(newton_raphson_step.len() >= NDIM);
    assert!(delta_x.len() >= NDIM);
    assert!(x.len() >= NDIM);
}