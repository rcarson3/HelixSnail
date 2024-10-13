use core::result::Result;
use log::error;

/// Decomposes our matrix into a partially pivoted LU decomposition, we supply a tolerance
/// for which the pivot is allowed to fail due to a row being nominally all zeros.
///
/// # Arguments:
/// * `matrix` - the matrix that is going to have the in-place partially pivoted LU decomposition occur and has a size of NDIM * NDIM
/// * `pivot` - the array which contains the indices corresponding to which row now corresponds to the i-th
///    row in the current `matrix`.
/// * `tolerance` - the tolerance in which we say a given value is zero
///
/// # Outputs:
/// A solver error which tells us why the linear solver failed
pub fn lup_decompose<const NDIM: usize, F>(
    tolerance: F,
    matrix: &mut [[F; NDIM]],
    pivot: &mut [usize],
) -> Result<(), crate::helix_error::SolverError>
where
    F: crate::FloatType,
{
    assert!(pivot.len() > NDIM);
    assert!(matrix.len() >= NDIM);

    for (i, item) in pivot.iter_mut().enumerate().take(NDIM + 1) {
        *item = i;
    }

    for i in 0..NDIM {
        let mut imax = i;
        let mut max_a = F::zero();
        for j in i..NDIM {
            let abs_a = F::abs(matrix[j][i]);
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
            return Err(crate::helix_error::SolverError::SmallPivot);
        }

        if imax != i {
            // Pivot contains what original row is in the current pivot[index] row
            pivot.swap(i, imax);
            // Swap the rows
            matrix.swap(i, imax);
            // for j in 0..NDIM {
            //     matrix.swap(i * NDIM + j, imax * NDIM + j)
            // }

            pivot[NDIM] += 1;
        }

        for j in (i + 1)..NDIM {
            // matrix_ji /= matrix_ii
            matrix[j][i] /= matrix[i][i];
            for k in (i + 1)..NDIM {
                // matrix_jk -= matrix_ji * matrix_ik
                matrix[j][k] -= matrix[j][i] * matrix[i][k];
            }
        }
    }

    Ok(())
}

/// This performs the solve of the system given a factorized A matrix in the form of P L U
/// It performs forward and back substitution in order to solve for the system
///
/// # Arguments:
/// * `solution` - the solution vector we're solving for which has a size of NDIM
/// * `matrix` - the partially pivoted LU decomposed A matrix which has a size of NDIM * NDIM
/// * `rhs` - the RHS of the system of equations we're solving for which has a size of NDIM
/// * `pivot` - the array which contains the indices corresponding to which row now corresponds to the i-th
///    row in the current `matrix`.
pub fn lup_solve<const NDIM: usize, F>(
    matrix: &[[F; NDIM]],
    rhs: &[F],
    pivot: &[usize],
    solution: &mut [F],
) where
    F: crate::FloatType,
{
    assert!(pivot.len() > NDIM);
    assert!(solution.len() >= NDIM);
    assert!(rhs.len() >= NDIM);
    assert!(matrix.len() >= NDIM);

    for i in 0..NDIM {
        solution[i] = rhs[pivot[i]];
        for k in 0..i {
            // solution_i -= matrix_ik * x_k
            solution[i] -= matrix[i][k] * solution[k];
        }
    }

    for i in (0..NDIM).rev() {
        for k in (i + 1)..NDIM {
            solution[i] -= matrix[i][k] * solution[k];
        }
        solution[i] /= matrix[i][i];
    }
}

/// A solver based on a partial pivot LU decomposition
/// For this solve, the decomposition is done in place on the matrix which is acceptable given
/// that we never use matrix afterwards.
/// We are solving for Ax = b where A = P L U
///
/// # Arguments:
/// * `matrix` - the A matrix up above which has a size of NDIM * NDIM
/// * `solution` - the x vector up above which has a size of NDIM
/// * `rhs` - the b vector up above which has a size of NDIM
pub fn lup_solver<const NDIM: usize, F>(
    rhs: &[F],
    matrix: &mut [[F; NDIM]],
    solution: &mut [F],
) -> Result<(), crate::helix_error::SolverError>
where
    F: crate::FloatType,
    [F; NDIM + 1]: Sized,
{
    assert!(solution.len() >= NDIM);
    assert!(rhs.len() >= NDIM);
    assert!(matrix.len() >= NDIM);

    let mut pivot = [0; NDIM + 1];
    let tolerance: F = F::from(1e-35).unwrap();

    lup_decompose::<{ NDIM }, F>(tolerance, matrix, &mut pivot)?;

    // Don't worry about pivoting matrix back to original form as we don't use LU in rest of code
    lup_solve::<{ NDIM }, F>(matrix, rhs, &pivot, solution);

    Ok(())
}
