#![allow(dead_code)]

/// Contains basic linear algebra functions like dot products, norms, matrix vector products, and matrix matrix products.
pub(crate) mod math;
/// Contains a few dense linear solvers such as LU and eventually a QR-based solver
pub(crate) mod linear_solvers;

pub(crate) use self::math::*;
#[allow(unused_imports)]
pub(crate) use self::linear_solvers::*;