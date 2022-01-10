use super::*;
use crate::linear_algebra::dot_prod;

fn dogleg<const NDIM: usize, F>(
    delta: F,
    residual_0: F,
    newton_raphson_l2_norm: F,
    jacobian2_gradient: F,
    gradient: &[F],
    newton_raphson_step: &[F],
    delta_x: &mut [F],
    x: &mut [F],
    predicted_residual: &mut F,
    use_newton_raphson: &mut bool,
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
    f64: Into<F>,
{
    assert!(gradient.len() >= NDIM);
    assert!(newton_raphson_step.len() >= NDIM);
    assert!(delta_x.len() >= NDIM);
    assert!(x.len() >= NDIM);

    // No need to do any other calculations if this condition is true
    if newton_raphson_l2_norm <= delta {
        // use Newton step
        *use_newton_raphson = true;

        for i in 0..NDIM {
            delta_x[i] = newton_raphson_step[i];
        }
        *predicted_residual = F::zero();
    }
    // Find Cauchy point
    else {
        let sqr_grad_l2_norm: F = dot_prod::<NDIM, F>(gradient, gradient);
        let grad_l2_norm: F = F::sqrt(sqr_grad_l2_norm);

        let alpha: F = if jacobian2_gradient > F::zero() {
            sqr_grad_l2_norm / jacobian2_gradient
        } else {
            F::one()
        };

        let grad_l2_norm_inv: F = if grad_l2_norm > F::zero() {
            F::one() / grad_l2_norm
        } else {
            F::one()
        };

        let norm_s_sd_opt: F = alpha * grad_l2_norm;

        // step along the dogleg path
        if norm_s_sd_opt >= delta {
            // use step along steapest descent direction
            {
                let factor: F = -delta * grad_l2_norm_inv;
                for i in 0..NDIM {
                    delta_x[i] = factor * gradient[i];
                }
            }

            {
                let val: F = -(delta * grad_l2_norm)
                    + 0.5.into()
                        * delta
                        * delta
                        * jacobian2_gradient
                        * (grad_l2_norm_inv * grad_l2_norm_inv);
                *predicted_residual = F::sqrt(F::max(
                    2.0.into() * val + residual_0 * residual_0,
                    F::zero(),
                ));
            }
        } else {
            let mut beta: F;
            // Scoping this set of calculations
            {
                let mut qb: F = F::zero();
                let mut qa: F = F::zero();
                for i in 0..NDIM {
                    let temp: F = newton_raphson_step[i] + alpha * gradient[i];
                    qa += temp * temp;
                    qb += temp * gradient[i];
                }
                // Previously qb = (-p^t g / ||g||) * alpha * ||g|| * 2.0
                // However, we can see that this simplifies a bit and also with the beta term
                // down below we there's a common factor of 2.0 that we can eliminate from everything
                qb *= alpha;
                // qc and beta depend on delta
                //
                let qc: F = norm_s_sd_opt * norm_s_sd_opt - delta * delta;
                beta = (qb + F::sqrt(qb * qb - qa * qc)) / qa;
            }
            beta = F::max(F::zero(), F::min(F::one(), beta));

            // delta_x[iX] = alpha*gradient[iX] + beta*p[iX] = beta*newton_raphson_step[iX] - (1.0-beta)*alpha*gradient[iX]
            //
            {
                let omb: F = F::one() - beta;
                let omba: F = omb * alpha;
                for i in 0..NDIM {
                    delta_x[i] = beta * newton_raphson_step[i] - omba * gradient[i];
                }

                let res_cauchy: F = if jacobian2_gradient > F::zero() {
                    F::sqrt(F::max(
                        residual_0 * residual_0 - alpha * sqr_grad_l2_norm,
                        F::zero(),
                    ))
                } else {
                    residual_0
                };

                *predicted_residual = omb * res_cauchy;
            }
        } // if norm_s_sd_opt >= delta
    } // use_newton_raphson

    // update x here to keep in line with batch version
    for i in 0..NDIM {
        x[i] += delta_x[i];
    }
}
