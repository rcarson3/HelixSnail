use core::fmt::{self, Debug};

mod private {
    #[derive(Debug)]
    pub enum Private {}
}

/// The error type used by this library.
/// This can encapsulate our nonlinear solver and linear solver errors,
/// and whatever else we might come up with in the future.
pub enum Error {
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

impl Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AlgorithmFailure => write!(f, "Nonlinear solver status algorithm failure"),
            Error::DeltaFailure => write!(f, "Nonlinear solver status delta failure"),
            Error::EvalFailure => write!(f, "Nonlinear solver status eval failure"),
            Error::InitialEvalFailure => write!(f, "Nonlinear solver status initial eval failure"),
            Error::SlowConvergence => write!(f, "Nonlinear solver status slow convergence"),
            Error::SlowJacobian => write!(f, "Nonlinear solver status slow jacobian status"),
            Error::UnconvergedMaxIter => {
                write!(f, "Nonlinear solver status unconverged max iterations")
            }
            Error::Unset => write!(f, "Nonlinear solver status unset"),
            Error::SmallPivot => write!(
                f,
                "Linear solver was not able to pivot due to an entire row being almost 0"
            ),
            Error::__NonExhaustive(_) => unreachable!(),
        }
    }
}
