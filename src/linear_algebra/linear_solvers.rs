use libnum::{Float, NumAssignOps, NumOps, One, Zero};
use log::error;

#[derive(Clone, PartialEq)]
pub(crate) enum SolverError {
    SmallPivot,
    NoError,
}

pub(crate) fn lup_decompose<const NDIM: usize, F>(
    matrix: &mut [F],
    pivot: &mut [usize],
    tolerance: F,
) -> SolverError
where
    F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug + core::convert::From<f64>,
{
    assert!(pivot.len() >= NDIM + 1);
    assert!(matrix.len() >= NDIM * NDIM);

    for i in 0..=NDIM {
        pivot[i] = i;
    }

    for i in 0..NDIM {
        let mut imax = i;
        let mut max_a = F::zero();
        for j in i..NDIM {
            let abs_a = F::abs(matrix[j * NDIM + i]);
            if abs_a > max_a {
                max_a = abs_a;
                imax = j;
            }
        }

        if max_a < tolerance {
            error!(
                "Pivot too small (pivot: {:?} < tolerance: {:?})",
                max_a, tolerance
            );
            return SolverError::SmallPivot;
        }

        if imax != i {
            let tmp = pivot[i];
            // Pivot contains what original row is in the current pivot[index] row
            pivot[i] = pivot[imax];
            pivot[imax] = tmp;
            // Swap the rows
            for j in 0..NDIM {
                let tmp = matrix[i * NDIM + j];
                matrix[i * NDIM + j] = matrix[imax * NDIM + j];
                matrix[imax * NDIM + j] = tmp;
            }

            pivot[NDIM] += 1;
        }

        for j in (i + 1)..NDIM {
            // matrix_ji /= matrix_ii
            matrix[j * NDIM + i] /= matrix[i * NDIM + i];
            for k in (i + 1)..NDIM {
                // matrix_jk -= matrix_ji * matrix_ik
                matrix[j * NDIM + k] -= matrix[j * NDIM + i] * matrix[i * NDIM + k];
            }
        }
    }

    SolverError::NoError
}

pub(crate) fn lup_solve<const NDIM: usize, F>(
    solution: &mut [F],
    matrix: &[F],
    rhs: &[F],
    pivot: &[usize],
) where
    F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug + core::convert::From<f64>,
{
    assert!(pivot.len() >= NDIM + 1);
    assert!(solution.len() >= NDIM);
    assert!(rhs.len() >= NDIM);
    assert!(matrix.len() >= NDIM * NDIM);

    for i in 0..NDIM {
        solution[i] = rhs[pivot[i]];
        for k in 0..i {
            // solution_i -= matrix_ik * x_k
            solution[i] -= matrix[i * NDIM + k] * solution[k];
        }
    }

    for i in (0..NDIM).rev() {
        for k in (i + 1)..NDIM {
            solution[i] -= matrix[i * NDIM + k] * solution[k];
        }
        solution[i] = solution[i] / matrix[i * NDIM + i];
    }
}

pub(crate) fn lup_solver<const NDIM: usize, F>(
    matrix: &mut [F],
    solution: &mut [F],
    rhs: &[F],
) -> SolverError
where
    F: Float + Zero + One + NumAssignOps + NumOps + core::fmt::Debug + core::convert::From<f64>,
    [(); NDIM + 1]:,
{
    assert!(solution.len() >= NDIM);
    assert!(rhs.len() >= NDIM);
    assert!(matrix.len() >= NDIM * NDIM);

    let mut pivot = [0; NDIM + 1];
    let tolerance = 1e-50.into();

    let error = lup_decompose::<{ NDIM }, F>(matrix, &mut pivot, tolerance);

    if error != SolverError::SmallPivot {
        // Don't worry about pivoting matrix back to original form as we don't use LU in rest of code
        lup_solve::<{ NDIM }, F>(solution, matrix, rhs, &pivot);
    }

    return error;
}
