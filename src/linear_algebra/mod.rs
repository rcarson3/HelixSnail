use libnum::{Float, NumAssignOps, NumOps, One, Zero};

/// Dot product of two vectors
/// vec1 and vec2 both have lengths NDIM
pub(crate) fn dot_prod<const NDIM: usize, F>(vec1: &[F], vec2: &[F]) -> F
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(vec1.len() >= NDIM);
    assert!(vec2.len() >= NDIM);

    let mut dot_prod: F = F::zero();
    for i in 0..NDIM {
        dot_prod += vec1[i] * vec2[i];
    }

    dot_prod
}

// Might want a stable norm eventually
/// Takes the L2 norm of a vector
/// where vec has length NDIM
pub(crate) fn norm<const NDIM: usize, F>(vec: &[F]) -> F
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(vec.len() >= NDIM);
    F::sqrt(dot_prod::<NDIM, F>(vec, vec))
}

/// Takes the norm of the columns of a matrix
/// where the matrix has dimensions NDIM x MDIM
/// The values are stored in norm_vec which is of length m
pub(crate) fn norm_column<const NDIM: usize, const MDIM: usize, F>(matrix: &[F], norm_vec: &mut [F])
where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(norm_vec.len() >= MDIM);
    // Initialize this to have the squared values of the first row
    for i_m in 0..MDIM {
        norm_vec[i_m] = matrix[i_m] * matrix[i_m];
    }

    // Accumulate the results across all remaining rows
    for i_n in 1..NDIM {
        for j_m in 0..MDIM {
            norm_vec[j_m] += matrix[i_n * MDIM + j_m] * matrix[i_n * MDIM + j_m];
        }
    }

    // Calculate the norm for each column
    for i_m in 0..MDIM {
        norm_vec[i_m] = F::sqrt(norm_vec[i_m]);
    }
}

/// Outer product of two vectors
/// over writes value in supplied matrix
/// matrix = vec1 \otimes vec2
/// vec1 has length NDIM
/// vec2 has length MDIM
/// matrix has dimensions NDIM x MDIM
pub(crate) fn outer_prod<const NDIM: usize, const MDIM: usize, F>(
    vec1: &[F],
    vec2: &[F],
    matrix: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(vec1.len() >= MDIM);
    assert!(vec2.len() >= NDIM);

    for i_n in 0..NDIM {
        for j_m in 0..MDIM {
            matrix[i_n * MDIM + j_m] = vec1[i_n] * vec2[j_m];
        }
    }
}

/// Adds a scaled outer product to supplied matrix
/// matrix += \scale * vec1 \otimes vec2
/// vec1 has length NDIM
/// vec2 has length MDIM
/// scale is an Option type and if supplied None defaults to a value of 1.0
/// matrix has dimensions NDIM x MDIM
pub(crate) fn outer_prod_add_scale<const NDIM: usize, const MDIM: usize, F>(
    vec1: &[F],
    vec2: &[F],
    scale: Option<F>,
    matrix: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(vec1.len() >= MDIM);
    assert!(vec2.len() >= NDIM);

    let alpha = if let Some(x) = scale { x } else { F::one() };

    for i_n in 0..NDIM {
        for j_m in 0..MDIM {
            matrix[i_n * MDIM + j_m] += alpha * vec1[i_n] * vec2[j_m];
        }
    }
}

/// Matrix vector product
/// matrix has dimensions NDIM x MDIM
/// vec has dimensions MDIM
/// prod has dimensions NDIM
pub(crate) fn mat_vec_mult<const NDIM: usize, const MDIM: usize, F>(
    matrix: &[F],
    vec: &[F],
    prod: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(vec.len() >= MDIM);
    assert!(prod.len() >= NDIM);
    for i_n in 0..NDIM {
        prod[i_n] = F::zero();
        for j_m in 0..MDIM {
            prod[i_n] += matrix[i_n * MDIM + j_m] * vec[j_m];
        }
    }
}

/// Matrix transpose vector product
/// matrix has dimensions NDIM x MDIM
/// vec has dimensions NDIM
/// prod has dimensions MDIM
pub(crate) fn mat_t_vec_mult<const NDIM: usize, const MDIM: usize, F>(
    matrix: &[F],
    vec: &[F],
    prod: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(vec.len() >= NDIM);
    assert!(prod.len() >= MDIM);

    for i_m in 0..MDIM {
        prod[i_m] = F::zero();
        for j_n in 0..NDIM {
            prod[i_m] += matrix[j_n * NDIM + i_m] * vec[j_n];
        }
    }
}

/// Upper triangle matrix vector product
/// matrix is an upper triangle matrix
/// (values below the diagonal are assumed zero)
/// matrix has dimensions NDIM x MDIM
/// vec has dimensions MDIM
/// prod has dimensions NDIM
/// NDIM <= MDIM
pub(crate) fn upper_tri_mat_vec_mult<const NDIM: usize, const MDIM: usize, F>(
    matrix: &[F],
    vec: &[F],
    prod: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(NDIM <= MDIM);
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(vec.len() >= MDIM);
    assert!(prod.len() >= NDIM);

    for i_n in 0..NDIM {
        prod[i_n] = F::zero();
        for j_m in i_n..MDIM {
            prod[i_n] += matrix[i_n * MDIM + j_m] * vec[j_m];
        }
    }
}

/// Upper triangle matrix transpose vector product
/// matrix is an upper triangle matrix
/// (values below the diagonal are assumed zero)
/// matrix has dimensions NDIM x MDIM
/// vec has dimensions MDIM
/// prod has dimensions NDIM
/// NDIM <= MDIM
pub(crate) fn upper_tri_mat_t_vec_mult<const NDIM: usize, const MDIM: usize, F>(
    matrix: &[F],
    vec: &[F],
    prod: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(NDIM <= MDIM);
    assert!(matrix.len() >= MDIM * NDIM);
    assert!(vec.len() >= MDIM);
    assert!(prod.len() >= NDIM);

    for i_n in 0..NDIM {
        prod[i_n] = F::zero();
        // M_ji * a_j = p_i
        // Only go down to diagonal
        for j_m in 0..i_n {
            prod[i_n] += matrix[j_m * MDIM + i_n] * vec[j_m];
        }
    }
}

/// Matrix-matrix multiplication
/// matrix1 has dimensions LDIM x NDIM
/// matrix2 has dimensions NDIM x MDIM
/// prod_matrix has dimensions LDIM x MDIM
/// This function will either accumulate the values of
/// the multiplication on the product,
/// or it will zero out the product ahead of time
/// depending on the compile time flag.
pub(crate) fn mat_mat_mult<
    const LDIM: usize,
    const NDIM: usize,
    const MDIM: usize,
    const ZERO_OUT: bool,
    F,
>(
    matrix1: &[F],
    matrix2: &[F],
    prod_matrix: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix1.len() >= LDIM * NDIM);
    assert!(matrix2.len() >= NDIM * MDIM);
    assert!(prod_matrix.len() >= LDIM * MDIM);

    if ZERO_OUT {
        for i_lm in 0..(LDIM * MDIM) {
            prod_matrix[i_lm] = F::zero();
        }
    }

    //prod_matrix_ik = matrix1_ij * matrix2_jk
    for i_l in 0..LDIM {
        for j_n in 0..NDIM {
            for k_m in 0..MDIM {
                prod_matrix[i_l * MDIM + k_m] +=
                    matrix1[i_l * NDIM + j_n] * matrix2[j_n * MDIM + k_m];
            }
        }
    }
}

/// Matrix transpose-matrix multiplication
/// matrix1 has dimensions NDIM x LDIM
/// matrix2 has dimensions NDIM x MDIM
/// prod_matrix has dimensions LDIM x MDIM
/// This function will either accumulate the values of
/// the multiplication on the product,
/// or it will zero out the product ahead of time
/// depending on the run time flag.
pub(crate) fn mat_t_mat_mult<
    const LDIM: usize,
    const NDIM: usize,
    const MDIM: usize,
    const ZERO_OUT: bool,
    F,
>(
    matrix1: &[F],
    matrix2: &[F],
    prod_matrix: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix1.len() >= LDIM * NDIM);
    assert!(matrix2.len() >= NDIM * MDIM);
    assert!(prod_matrix.len() >= LDIM * MDIM);

    if ZERO_OUT {
        for i_lm in 0..(LDIM * MDIM) {
            prod_matrix[i_lm] = F::zero();
        }
    }

    //prod_matrix_ik = matrix1_ji * matrix2_jk
    for j_n in 0..NDIM {
        for i_l in 0..LDIM {
            for k_m in 0..MDIM {
                prod_matrix[i_l * MDIM + k_m] +=
                    matrix1[j_n * LDIM + i_l] * matrix2[j_n * MDIM + k_m];
            }
        }
    }
}

/// Matrix-matrix transpose multiplication
/// matrix1 has dimensions LDIM x NDIM
/// matrix2 has dimensions MDIM x NDIM
/// prod_matrix has dimensions LDIM x MDIM
/// This function will either accumulate the values of
/// the multiplication on the product,
/// or it will zero out the product ahead of time
/// depending on the run time flag.
pub(crate) fn mat_mat_t_mult<
    const LDIM: usize,
    const NDIM: usize,
    const MDIM: usize,
    const ZERO_OUT: bool,
    F,
>(
    matrix1: &[F],
    matrix2: &[F],
    prod_matrix: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(matrix1.len() >= LDIM * NDIM);
    assert!(matrix2.len() >= NDIM * MDIM);
    assert!(prod_matrix.len() >= LDIM * MDIM);

    if ZERO_OUT {
        for i_lm in 0..(LDIM * MDIM) {
            prod_matrix[i_lm] = F::zero();
        }
    }

    //prod_matrix_ik = matrix1_ij * matrix2_kj
    for i_l in 0..LDIM {
        for k_m in 0..MDIM {
            for j_n in 0..NDIM {
                prod_matrix[i_l * MDIM + k_m] +=
                    matrix1[i_l * NDIM + j_n] * matrix2[k_m * NDIM + j_n];
            }
        }
    }
}

/// Performs the triple product operation
/// needed to rotate a matrix of NDIM x NDIM
/// by a rotation matrix
/// The product matrix is zeroed out in this operation
/// Transpose does the operation:
/// prod_matrix_il = rot_matrix_ji matrix_jk rot_matrix_kl
/// prod_matrix = rot_matrix^t * matrix * rot_matrix
/// Non-transpose operation does:
/// prod_matrix_il = rot_matrix_ij matrix_jk rot_matrix_lk
/// prod_matrix = rot_matrix * matrix * rot_matrix^T
pub(crate) fn rotate_matrix<const NDIM: usize, const TRANSPOSE: bool, F>(
    rot_matrix: &[F],
    matrix: &[F],
    prod_matrix: &mut [F],
) where
    F: Float + Zero + One + NumAssignOps + NumOps,
{
    assert!(rot_matrix.len() >= NDIM * NDIM);
    assert!(matrix.len() >= NDIM * NDIM);
    assert!(prod_matrix.len() >= NDIM * NDIM);

    // zero things out first
    for i_nn in 0..(NDIM * NDIM) {
        prod_matrix[i_nn] = F::zero();
    }

    // Now for matrix multiplication
    for i_n in 0..NDIM {
        for j_n in 0..NDIM {
            for k_n in 0..NDIM {
                for l_n in 0..NDIM {
                    if TRANSPOSE {
                        // This is rot_matrix_ji matrix_jk rot_matrix_kl
                        prod_matrix[i_n * NDIM + l_n] += rot_matrix[j_n * NDIM + i_n]
                            * matrix[j_n * NDIM + k_n]
                            * rot_matrix[k_n * NDIM + l_n];
                    } else {
                        // This is rot_matrix_ij matrix_jk rot_matrix_lk
                        prod_matrix[i_n * NDIM + l_n] += rot_matrix[i_n * NDIM + j_n]
                            * matrix[j_n * NDIM + k_n]
                            * rot_matrix[l_n * NDIM + k_n];
                    }
                }
            }
        }
    }
}
