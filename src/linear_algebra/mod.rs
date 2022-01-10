use libnum::{Float, NumAssignOps, NumOps, One, Zero};

/// Dot product of two vectors
/// v1 and v2 both have lengths NDIM
pub(crate) fn dot_prod<const NDIM: usize, F>(v1: &[F], v2: &[F]) -> F
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(v1.len() >= NDIM);
    assert!(v2.len() >= NDIM);

    let mut dot_prod: F = F::zero();
    for i in 0..NDIM {
        dot_prod += v1[i] * v2[i];
    }

    dot_prod
}
