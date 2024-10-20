// use log::error;
use core::result::Result;
use crate::linear_algebra::math::*;

// Need to add an actual QR solver in here at some point. It currently contains just QR factorization functions

// Algorithm taken from Matrix Computation 4th ed. by GH Golub and CF Van Loan
// pg. 236 alg. 5.1.1
pub fn householder_vector<const NDIM: usize, F>(
    x: &[F],
    nu: &mut [F],
) -> F
where
    F: crate::FloatType,
    [(); NDIM - 1]: Sized,
{
    assert!(x.len() >= NDIM);
    assert!(nu.len() >= NDIM);
    let sigma: F;
    let beta: F;

    sigma = dot_prod::<{NDIM - 1}, F>(&x[1..], &x[1..]);
    nu[0] = F::one();
    for i in 1..NDIM {
        nu[i] = x[i];
    }

    if (sigma == F::zero()) && (x[0] != F::zero()) {
        beta = F::zero();
    }
    else {
        let mu = F::sqrt(x[0] * x[0] + sigma);
        if x[0] >= F::zero() {
            nu[0] = x[0] - mu;
        }
        else {
            nu[0] = -sigma / (x[0] + mu);
        }
        // Sometimes nu[0] is pratically zero which can cause issues when inverting it.
        // Therefore, we want to multiply it by some small number.
        // Here we choose double machine precision. However, we might need to improve this
        // in the future and copy the scaling scheme in LAPACK
        if F::abs(nu[0]) < F::epsilon() {
            nu[0] = F::epsilon();
        }
        beta = (F::from(2.0).unwrap() * nu[0] * nu[0]) / (sigma + nu[0] * nu[0]);
        let inu0: F = F::one() / nu[0];
        nu[0] = F::one();
        for i in 1..NDIM {
            nu[i] *= inu0;
        }
    }
    beta
}

// Algorithm taken from Matrix Computation 4th ed. by GH Golub and CF Van Loan
// pg. 238 alg 5.1.5 with the slight modification that we use the beta values
// saved off earlier.
pub fn householder_q_matrix<const NDIM: usize, F>(
    matrix_factor: &[[F; NDIM]],
    beta_vector: &[F],
    q_matrix: &mut [[F; NDIM]]
)
where
    F: crate::FloatType,
    [(); NDIM]: Sized,
{
    assert!(matrix_factor.len() >= NDIM);
    assert!(beta_vector.len() >= NDIM);
    assert!(q_matrix.len() >= NDIM);
    // We need two working arrays for the
    // nu^T Q product and nu.
    let mut nu = [F::zero(); NDIM];
    let mut nu_q_t = [F::zero(); NDIM];

    // Initialize Q_matrix to be an identity matrix
    for i in 0..NDIM {
        for j in 0..NDIM {
            q_matrix[i][j] = F::zero();
        }
        q_matrix[i][i] = F::one();
    }

    for i in (0..(NDIM - 1)).rev() {
        // Initialize nu and nu_Q_T
        for j in 0..(NDIM - i) {
            nu[j] = matrix_factor[j + i][i];
            nu_q_t[j] = F::zero();
        }
        nu[0] = F::one();
        let beta = beta_vector[i];

        // nu.T * Q[i:m, i:m]
        for j in 0..(NDIM - i) {
            for k in 0..(NDIM - i) {
                nu_q_t[k] += nu[j] * q_matrix[i + j][i + k]; 
            }
        }

        // Now Q[i:m, i:m]  -= (betav[i] * nu) \otimes nuTQ
        // Q_jk -= betav[i] * nu_j * nuTQ_k
        for j in 0..(NDIM - i) {
            for k in 0..(NDIM - i) {
                q_matrix[i + j][i + k] -= beta * nu[j] * nu_q_t[k]; 
            }
        }
    }
}

// Algorithm taken from Matrix Computation 4th ed. by GH Golub and CF Van Loan
// pg. 249
// The below are the general requirements for the arrays
// Here we return the householder R matrix in the matrix_factor matrix
// and Q within the supplied Q matrix. Q should have dimensions
// mxm and R should have dimensions of mxn where m >= n
// However for our nonlinear solvers we always have m == n
// So, we're just going to specialize things to simplify the logic a bit
pub fn householder_qr<const NDIM: usize, F>(
    matrix_factor: &mut [[F; NDIM]],
    q_matrix: &mut [[F; NDIM]],
    work_array1: &mut [F],
    work_array2: &mut [F],
    work_array3: &mut [F],
)
where
    F: crate::FloatType,
    [(); NDIM]: Sized,
    [(); NDIM - 1]: Sized,
{
    assert!(matrix_factor.len() >= NDIM);
    assert!(q_matrix.len() >= NDIM);
    assert!(work_array1.len() >= NDIM);
    assert!(work_array2.len() >= (NDIM - 1));
    assert!(work_array3.len() >= NDIM);

    for i in 0..NDIM {
        // The householder vector / work_array3 is below:
        for j in 0..(NDIM - i) {
            work_array3[j] = matrix_factor[i + j][i];
        }
        // work_arrary_1 is the beta values
        // work_array_2 are the nu vector which have length m - i
        work_array1[i] = householder_vector::<NDIM, F>(&work_array3, work_array2);
        // work_array_3 will now contain the product nu.T * matrix[i:m, i:n]
        // It has dimensions of 1 x n
        for j in 0..(NDIM - i) {
            work_array3[j] = F::zero();
        }
        for j in 0..(NDIM - i) {
            for k in 0..(NDIM - i) {
                work_array3[k] += work_array2[j] * matrix_factor[i + j][i + k];
            }
        }

        // Now matrix[i:m, i:n]  -= (beta[i] * nu) \otimes work_array_3
        // matrix[j][k] -= beta[i] * nu[j] * work_array_3[k]
        for j in 0..(NDIM - i) {
            for k in 0..(NDIM - i) {
                matrix_factor[i + j][i + k] -= work_array1[i] * work_array2[j] * work_array3[k]; 
            }
        }
        // Save off our nu[1:(m - i)] values to the lower triangle parts of the matrix
        // This won't run for the last element in the array as nu is a single value and 
        // has the trivial 1 value.
        if i < NDIM {
            for j in 1..(NDIM - i) {
                matrix_factor[i + j][i] = work_array2[j];
            }
        }
    }

    // Now back out the Q array. Although, we could maybe do this inline with the above
    // if we thought about this a bit more.
    householder_q_matrix(matrix_factor, &work_array1, q_matrix);
}

pub fn qr_solve<const NDIM: usize, F>(
    r_matrix: &[[F; NDIM]],
    rhs: &[F],
    solution: &mut [F],
) -> Result<(), crate::helix_error::SolverError>
where
    F: crate::FloatType,
    [(); NDIM]: Sized,
{
    assert!(solution.len() >= NDIM);
    assert!(rhs.len() >= NDIM);
    assert!(r_matrix.len() >= NDIM);

    let tolerance: F = F::from(1e-35).unwrap();

    for i in (0..(NDIM - 1)).rev() {
        let rmat_val = r_matrix[i][i];
        if F::abs(rmat_val) < tolerance {
            return Err(crate::helix_error::SolverError::AlgorithmFailure);
        }

        let mut sum = F::zero();
        for j in (i + 1)..NDIM {
            sum += r_matrix[i][j] * solution[j];
        }
        solution[i] = (rhs[i] - sum) / rmat_val;
    }
    Ok(())
}

pub fn make_givens<F: crate::FloatType>(p: F, q: F, givens: &mut [F])
{
    assert!(givens.len() >= 2);
    if q == F::zero() {
        givens[0] = if p < F::zero() { -F::one() } else { F::one()};
        givens[1] = F::zero();
    }
    else if p == F::zero() {
        givens[1] = if q < F::zero() { F::one() } else { -F::one()};
        givens[0] = F::zero();
    }
    else if F::abs(p) > F::abs(q) {
        let t = q / p;
        let factor = if p < F::zero() { -F::one() } else { F::one()};
        let u = factor * F::sqrt(F::one() + t * t);
        givens[0] = F::one() / u;
        givens[1] = -t * givens[0];
    }
    else {
        let t = p / q;
        let factor = if q < F::zero() { -F::one() } else { F::one()};
        let u = factor * F::sqrt(F::one() + t * t);
        givens[1] = -F::one() / u;
        givens[0] = -t * givens[1];
    }    
}
