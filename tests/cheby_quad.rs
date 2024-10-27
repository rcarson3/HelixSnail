#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
extern crate env_logger;
extern crate helix_snail;
extern crate num_traits as libnum;

use core::marker::PhantomData;
use helix_snail::nonlinear_solver::*;
use log::{warn, info};
// Making use of past here makes it slightly nicer to write the necessary test macro
use paste::paste;

const LOGGING_LEVEL: i32 = 1;

// This problem is described originally in 
// Fletcher, R. "Function minimization without evaluating derivatives - a review." The Computer Journal 8.1 (1965): 33-41.
// doi: https://doi.org/10.1093/comjnl/8.1.33
// It's original description in the Fletcher paper is a function that "represents those found in practice".
struct ChebyQuad<F, const NDIMI: usize>
where
    F: helix_snail::FloatType,
{
    phantom_data: PhantomData<F>,
}

impl<F, const NDIMI: usize> NonlinearSystemSize for ChebyQuad<F, NDIMI>
where
    F: helix_snail::FloatType,
{
    const NDIM: usize = NDIMI;
}

impl<F, const NDIMI: usize> NonlinearNDProblem<F> for ChebyQuad<F, NDIMI>
where
    F: helix_snail::FloatType,
    [(); Self::NDIM]:
{
    fn compute_resid_jacobian(
        &mut self,
        x: &[F],
        fcn_eval: &mut [F],
        opt_jacobian: Option<&mut [F]>,
    ) -> bool {
        assert!(fcn_eval.len() >= Self::NDIM);
        assert!(x.len() >= Self::NDIM);

        let mut temp1 = F::zero();
        let mut temp2 = F::zero();
        let mut temp = F::zero();
        let mut ti = F::zero();
        let mut d1 = F::zero();
        let mut tk = F::zero();
        let mut temp3 = F::zero();
        let mut temp4 = F::zero();
    
        for i in 0..Self::NDIM {
            fcn_eval[i] = F::zero();
        }

        for i in 0..Self::NDIM {
            temp1 = F::one();
            temp2 = F::from(2.0).unwrap() * x[i] - F::one();
            temp = F::from(2.0).unwrap() * temp2;
            for j in 0..Self::NDIM {
                fcn_eval[j] += temp2;
                ti = temp * temp2 - temp1;
                temp1 = temp2;
                temp2 = ti;
            }
        }

        tk = F::one() / F::from(Self::NDIM).unwrap();

        let mut iev = false;

        for i in 0..Self::NDIM {
            fcn_eval[i] *= tk;
            if iev {
                d1 = F::from(i).unwrap() + F::one();
                fcn_eval[i] += F::one() / (d1 * d1 - F::one()); 
            }
            iev = !iev;

        }

        if let Some(jac) = opt_jacobian {
            assert!(jac.len() >= Self::NDIM * Self::NDIM, "length {:?}", jac.len());
            let jacobian = helix_snail::array1d_to_array2d_mut::<{Self::NDIM}, F>(jac);
            for i in 0..Self::NDIM {
                temp1 = F::one();
                temp2 = F::from(2.0).unwrap() * x[i] - F::one();
                temp = F::from(2.0).unwrap() * temp2;
                temp3 = F::zero();
                temp4 = F::from(2.0).unwrap();
                for j in 0..Self::NDIM {
                    jacobian[i][j] = tk * temp4;
                    ti = F::from(4.0).unwrap() * temp2 + temp * temp4 - temp3;
                    temp3 = temp4;
                    temp4 = ti;
                    ti = temp * temp2 - temp1;
                    temp1 = temp2;
                    temp2 = ti;
                }
            }
        }
        true
    }
}

/// Test macro for the trust region method that uses a dogleg solver for the subspace
/// for the broyden test problem.
/// Inputs for this are the extended name we want to go with the initial name broyden_tr_dogleg_$name_$type
/// $type is either f32 or f64 for the solver
/// $lambda is the lambda we want the Broyden class to use and be associated with
/// $tolerance is the tolerance for the solver
/// Note: we make use of the paste macro in order to be able to actually append names to the function
/// as this makes our naming convention for things simpler...
macro_rules! cheby_quad_tests {
    ($(($name:ident, $type:ident, $ndim:literal, $tolerance:expr, $pass_fail:expr),)*) => {
        $(
            paste! {
                #[test]
                fn [< cheby_quad_ $name _ $type >]() {
                    let _ = env_logger::builder().is_test(true).try_init();

                    let mut cheby_quad = ChebyQuad::<$type, $ndim> {
                        phantom_data: PhantomData::<$type>,
                    };

                    let dc = TrustRegionDeltaControl::<$type> {
                        delta_init: 1.0,
                        xi_decr_delta: 0.75,
                        ..Default::default()
                    };
                    {
                        let mut solver = HybridTRDglSolver::<$type, ChebyQuad::<$type,$ndim>>::new(&dc, &mut cheby_quad);
                        let ndim: u16 = $ndim;
                        let ndim_range: usize = $ndim;
                        let h = $type::from(1.0) / ($type::from(ndim) + $type::from(1.0));
                        for i in 0..ndim_range {
                            let tmp: u16 = i.try_into().unwrap();
                            solver.x[i] = ($type::from(tmp) + $type::from(1.0)) * h;
                        }

                        solver.set_logging_level(Some(LOGGING_LEVEL));
                        solver.setup_options(ChebyQuad::<$type,{$ndim}>::NDIM * 30, $tolerance, Some(LOGGING_LEVEL));

                        let err = solver.solve();

                        info!("solver fcn iterations: {:?}", solver.get_num_fcn_evals());
                        info!("solver jacob iterations: {:?}", solver.get_num_jacobian_evals());

                        let status = match err {
                            Ok(()) => true,
                            Err(e) => {
                                warn!("Solution did not converge with following error {:?}", e);
                                false
                            }
                        };

                        assert!(
                            status == $pass_fail,
                            "Solution did not converge"
                        );
                    }
                }
            }
        )*
    }
}

// The ndim = 3 test will fail due to how our trust region controls our step size
// It's possible that this could be solved if we tuned the trust region parameters just right.
// The other tests should pass
cheby_quad_tests! {
    (n3_fail, f64, 3, 1e-12, false),
    (n3_fail, f32, 3, 1e-6, false),
    (n5, f64, 5, 1e-12, true),
    (n5, f32, 5, 1e-6, true),
}