#![allow(dead_code)]

use crate::linear_algebra::dot_prod;

/// The dogleg is an approach to solving the update step of a nonlinear system of equations.
///
/// It is originally described in:
/// M.J.D. Powell, “A new algorithm for unconstrained optimization”, in:
/// J.B. Rosen, O.L. Mangasarian, and K. Ritter, eds., Nonlinear programming (Academic Press, New York, 1970).
///
/// # Arguments
///
/// * `delta` - the restriction placed on the allowable step size based on something like the trust-region
/// * `residual_0` - the previous time step solution residual
/// * `newton_raphson_l2_norm` - the l2 norm of the newton raphson solution of the nonlinear equation for a given step
/// * `jacobian2_gradient` - the square of the l2 norm of the J * J^T * residual product
/// * `gradient` - the J^T * residual product
/// * `newton_raphson_step` - the delta x if a full newton step is taken
/// * `delta_x` - the solution step based on the dogleg problem
/// * `x` - the updated solution at end of the dogleg problem
/// * `predicted_residual` - the predicted l2 norm of the updated residual calculated using the updated x
/// * `use_newton_raphson` - a simple flag just saying whether or not or dogleg method took the full NR step
#[allow(clippy::too_many_arguments)]
pub fn dogleg<const NDIM: usize, F>(
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
    F: crate::FloatType,
{
    assert!(gradient.len() >= NDIM);
    assert!(newton_raphson_step.len() >= NDIM);
    assert!(delta_x.len() >= NDIM);
    assert!(x.len() >= NDIM);

    // No need to do any other calculations if this condition is true
    if newton_raphson_l2_norm <= delta {
        // use Newton step
        *use_newton_raphson = true;
        delta_x[..NDIM].clone_from_slice(&newton_raphson_step[..NDIM]);
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
            // use step along steepest descent direction
            {
                let factor: F = -delta * grad_l2_norm_inv;
                for i in 0..NDIM {
                    delta_x[i] = factor * gradient[i];
                }
            }

            {
                let val: F = -(delta * grad_l2_norm)
                    + F::from(0.5).unwrap()
                        * delta
                        * delta
                        * jacobian2_gradient
                        * (grad_l2_norm_inv * grad_l2_norm_inv);
                *predicted_residual = F::sqrt(F::max(
                    F::from(2.0).unwrap() * val + residual_0 * residual_0,
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
