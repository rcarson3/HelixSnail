use core::error::Error;
use core::fmt::{self, Debug, Display};

mod private {
    #[derive(Debug)]
    pub enum Private {}
}

/// The error type used by this library.
/// This can encapsulate our nonlinear solver and linear solver errors,
/// and whatever else we might come up with in the future.
pub enum SolverError {
    /// Initial evaluation of nonlinear problem compute_resid_jacobian failed
    InitialEvalFailure,
    /// Evaluation of nonlinear problem compute_resid_jacobian failed
    EvalFailure,
    /// Failure within delta calculation
    DeltaFailure,
    /// Reached max number of iterations and still not converged
    UnconvergedMaxIter,
    /// Jacobian calculations are not adequately leading to solution to converge
    SlowJacobian,
    /// Solution is not making sufficient convergence progress
    SlowConvergence,
    /// Algorithm failed
    AlgorithmFailure,
    /// Values were unset
    Unset,
    /// A small pivot aka a row nominally full of zeros caused the solver to fail
    SmallPivot,
    #[doc(hidden)]
    __NonExhaustive(private::Private),
}

impl Error for SolverError {}

impl Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::AlgorithmFailure => write!(f, "Nonlinear solver status algorithm failure"),
            SolverError::DeltaFailure => write!(f, "Nonlinear solver status delta failure"),
            SolverError::EvalFailure => write!(f, "Nonlinear solver status eval failure"),
            SolverError::InitialEvalFailure => {
                write!(f, "Nonlinear solver status initial eval failure")
            }
            SolverError::SlowConvergence => write!(f, "Nonlinear solver status slow convergence"),
            SolverError::SlowJacobian => write!(f, "Nonlinear solver status slow jacobian status"),
            SolverError::UnconvergedMaxIter => {
                write!(f, "Nonlinear solver status unconverged max iterations")
            }
            SolverError::Unset => write!(f, "Nonlinear solver status unset"),
            SolverError::SmallPivot => write!(
                f,
                "Linear solver was not able to pivot due to an entire row being almost 0"
            ),
            SolverError::__NonExhaustive(_) => unreachable!(),
        }
    }
}

impl Debug for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
