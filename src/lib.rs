//! Pure Rust port of [libLBFGS](https://github.com/chokkan/liblbfgs).
//!
//! Bit-exact reproduction of the original C library: every `f64` produced by
//! this crate matches the reference C implementation down to the last bit.
//!
//! Public API: [`lbfgs`], [`LbfgsParam`], [`LineSearch`], [`OrthantWise`],
//! [`LbfgsError`], [`Convergence`].

#![allow(non_camel_case_types)]

mod arithmetic;
mod lbfgs_core;
mod linesearch;
mod owlqn;

// Re-export the C FFI types for the compare-c feature
#[cfg(feature = "compare-c")]
pub mod c_ffi;

/// Line search algorithm selection.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LineSearch {
    /// `LBFGS_LINESEARCH_MORETHUENTE`: More-Thuente method satisfying strong
    /// Wolfe conditions (default).
    MoreThuente,
    /// `LBFGS_LINESEARCH_BACKTRACKING_ARMIJO`: backtracking with the
    /// sufficient decrease (Armijo) condition only.
    BacktrackingArmijo,
    /// `LBFGS_LINESEARCH_BACKTRACKING_WOLFE`: backtracking with sufficient
    /// decrease (Armijo) and curvature (Wolfe) conditions.
    BacktrackingWolfe,
    /// `LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE`: backtracking with
    /// sufficient decrease and strong curvature conditions.
    BacktrackingStrongWolfe,
}

impl Default for LineSearch {
    fn default() -> Self {
        LineSearch::MoreThuente
    }
}

/// L1 (OWL-QN) regularization parameters.
///
/// Enables OWL-QN (Orthant-Wise Limited-memory Quasi-Newton) L1-regularized
/// optimization. See Andrew & Gao, ICML 2007.
#[derive(Debug, Copy, Clone)]
pub struct OrthantWise {
    /// Coefficient for L1 norm. Positive value activates OWL-QN.
    pub c: f64,
    /// Start index for L1 norm computation (default 0).
    pub start: i32,
    /// End index for L1 norm computation. Negative means use n.
    pub end: i32,
}

/// L-BFGS optimization parameters.
///
/// These fields correspond 1:1 to the C `lbfgs_parameter_t` struct from
/// `lbfgs.h`. Use [`LbfgsParam::default()`] to obtain the same defaults
/// produced by `lbfgs_parameter_init()`.
#[derive(Debug, Copy, Clone)]
pub struct LbfgsParam {
    /// Number of corrections to approximate the inverse Hessian (default 6).
    /// Typical range 3-20; values <3 may be slow, >20 use extra memory.
    pub m: i32,
    /// Epsilon for convergence test (default 1e-5).
    pub epsilon: f64,
    /// Distance in iterations for delta-based convergence (0 = disabled).
    pub past: i32,
    /// Delta for convergence test (default 1e-5).
    pub delta: f64,
    /// Maximum number of iterations (0 = unlimited).
    pub max_iterations: i32,
    /// Line search algorithm.
    pub linesearch: LineSearch,
    /// Maximum number of line search trials (default 40).
    pub max_linesearch: i32,
    /// Minimum step of line search (default 1e-20).
    pub min_step: f64,
    /// Maximum step of line search (default 1e20).
    pub max_step: f64,
    /// Accuracy parameter for line search (Armijo condition, default 1e-4).
    pub ftol: f64,
    /// Wolfe condition coefficient (default 0.9).
    pub wolfe: f64,
    /// Gradient accuracy for line search (default 0.9). Smaller values produce
    /// a more accurate line search.
    pub gtol: f64,
    /// Machine precision estimate (default 1e-16).
    pub xtol: f64,
    /// OWL-QN L1 regularization (None = standard minimization).
    pub orthantwise: Option<OrthantWise>,
}

impl Default for LbfgsParam {
    fn default() -> Self {
        LbfgsParam {
            m: 6,
            epsilon: 1e-5,
            past: 0,
            delta: 1e-5,
            max_iterations: 0,
            linesearch: LineSearch::default(),
            max_linesearch: 40,
            min_step: 1e-20,
            max_step: 1e20,
            ftol: 1e-4,
            wolfe: 0.9,
            gtol: 0.9,
            xtol: 1.0e-16,
            orthantwise: None,
        }
    }
}

/// Progress information passed to the progress callback.
///
/// Snapshot passed to the user progress callback. Return `false` from the
/// callback to cancel optimization (yields [`LbfgsError::Canceled`]).
pub struct ProgressReport<'a> {
    /// Current variables.
    pub x: &'a [f64],
    /// Current gradient.
    pub g: &'a [f64],
    /// Current objective value.
    pub fx: f64,
    /// Euclidean norm of `x`.
    pub xnorm: f64,
    /// Euclidean norm of `g` (or pseudo-gradient when OWL-QN is active).
    pub gnorm: f64,
    /// Line-search step taken this iteration.
    pub step: f64,
    /// Iteration count (1-based).
    pub k: i32,
    /// Line-search evaluations this iteration.
    pub ls: i32,
}

/// Error codes from the L-BFGS optimizer.
///
/// Each variant corresponds to a `LBFGSERR_*` constant from the C library;
/// the `Display` impl produces the same human-readable strings as
/// `lbfgs_strerror()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LbfgsError {
    /// Unknown error.
    UnknownError,
    /// Logic error.
    LogicError,
    /// Insufficient memory.
    OutOfMemory,
    /// The minimization process has been canceled.
    Canceled,
    /// Invalid number of variables specified.
    InvalidN,
    /// Invalid epsilon.
    InvalidEpsilon,
    /// Invalid past.
    InvalidTestPeriod,
    /// Invalid delta.
    InvalidDelta,
    /// Invalid linesearch.
    InvalidLineSearch,
    /// Invalid min_step.
    InvalidMinStep,
    /// Invalid max_step.
    InvalidMaxStep,
    /// Invalid ftol.
    InvalidFtol,
    /// Invalid wolfe.
    InvalidWolfe,
    /// Invalid gtol.
    InvalidGtol,
    /// Invalid xtol.
    InvalidXtol,
    /// Invalid max_linesearch.
    InvalidMaxLineSearch,
    /// Invalid orthantwise_c.
    InvalidOrthantwise,
    /// Invalid orthantwise_start.
    InvalidOrthantwiseStart,
    /// Invalid orthantwise_end.
    InvalidOrthantwiseEnd,
    /// Line-search step out of interval.
    OutOfInterval,
    /// Interval of uncertainty became too small.
    IncorrectTMinMax,
    /// Rounding error in line search.
    RoundingError,
    /// Line-search step became smaller than min_step.
    MinimumStep,
    /// Line-search step became larger than max_step.
    MaximumStep,
    /// Line-search reached maximum evaluations.
    MaximumLineSearch,
    /// Reached maximum number of iterations.
    MaximumIteration,
    /// Interval of uncertainty width at most xtol.
    WidthTooSmall,
    /// Negative line-search step.
    InvalidParameters,
    /// Search direction increases the objective function.
    IncreaseGradient,
}

impl std::fmt::Display for LbfgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            LbfgsError::UnknownError => "Unknown error",
            LbfgsError::LogicError => "Logic error",
            LbfgsError::OutOfMemory => "Insufficient memory",
            LbfgsError::Canceled => "The minimization process has been canceled",
            LbfgsError::InvalidN => "Invalid number of variables specified",
            LbfgsError::InvalidEpsilon => "Invalid epsilon",
            LbfgsError::InvalidTestPeriod => "Invalid past",
            LbfgsError::InvalidDelta => "Invalid delta",
            LbfgsError::InvalidLineSearch => "Invalid linesearch",
            LbfgsError::InvalidMinStep => "Invalid min_step",
            LbfgsError::InvalidMaxStep => "Invalid max_step",
            LbfgsError::InvalidFtol => "Invalid ftol",
            LbfgsError::InvalidWolfe => "Invalid wolfe",
            LbfgsError::InvalidGtol => "Invalid gtol",
            LbfgsError::InvalidXtol => "Invalid xtol",
            LbfgsError::InvalidMaxLineSearch => "Invalid max_linesearch",
            LbfgsError::InvalidOrthantwise => "Invalid orthantwise_c",
            LbfgsError::InvalidOrthantwiseStart => "Invalid orthantwise_start",
            LbfgsError::InvalidOrthantwiseEnd => "Invalid orthantwise_end",
            LbfgsError::OutOfInterval => "Line-search step out of interval",
            LbfgsError::IncorrectTMinMax => "Interval of uncertainty became too small",
            LbfgsError::RoundingError => "Rounding error in line search",
            LbfgsError::MinimumStep => "Line-search step became smaller than min_step",
            LbfgsError::MaximumStep => "Line-search step became larger than max_step",
            LbfgsError::MaximumLineSearch => "Line-search reached maximum evaluations",
            LbfgsError::MaximumIteration => "Reached maximum number of iterations",
            LbfgsError::WidthTooSmall => "Interval of uncertainty width at most xtol",
            LbfgsError::InvalidParameters => "Negative line-search step",
            LbfgsError::IncreaseGradient => "Search direction increases the objective function",
        };
        write!(f, "{}", msg)
    }
}

impl std::error::Error for LbfgsError {}

/// Convergence status from a successful optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Convergence {
    /// Gradient norm convergence: ||g|| / max(1, ||x||) <= epsilon.
    Gradient,
    /// Delta-based convergence: relative improvement < delta.
    Delta,
    /// Initial variables already minimize the objective function.
    AlreadyMinimized,
}

/// Result of an L-BFGS optimization.
pub struct LbfgsResult {
    /// How convergence was achieved.
    pub convergence: Convergence,
    /// Final objective function value.
    pub fx: f64,
}

// ---------------------------------------------------------------------------
// Internal types used by lbfgs_core and linesearch (kept C-compatible for
// the internal implementation, which is a direct translation of the C code).
// ---------------------------------------------------------------------------

// These are the raw integer constants used internally to match the C code exactly.
pub(crate) const LBFGS_SUCCESS: i32 = 0;
pub(crate) const LBFGS_STOP: i32 = 1;
pub(crate) const LBFGS_ALREADY_MINIMIZED: i32 = 2;
pub(crate) const LBFGSERR_UNKNOWNERROR: i32 = -1024;
pub(crate) const LBFGSERR_LOGICERROR: i32 = -1023;
pub(crate) const LBFGSERR_OUTOFMEMORY: i32 = -1022;
pub(crate) const LBFGSERR_CANCELED: i32 = -1021;
pub(crate) const LBFGSERR_INVALID_N: i32 = -1020;
pub(crate) const LBFGSERR_INVALID_EPSILON: i32 = -1017;
pub(crate) const LBFGSERR_INVALID_TESTPERIOD: i32 = -1016;
pub(crate) const LBFGSERR_INVALID_DELTA: i32 = -1015;
pub(crate) const LBFGSERR_INVALID_LINESEARCH: i32 = -1014;
pub(crate) const LBFGSERR_INVALID_MINSTEP: i32 = -1013;
pub(crate) const LBFGSERR_INVALID_MAXSTEP: i32 = -1012;
pub(crate) const LBFGSERR_INVALID_FTOL: i32 = -1011;
pub(crate) const LBFGSERR_INVALID_WOLFE: i32 = -1010;
pub(crate) const LBFGSERR_INVALID_GTOL: i32 = -1009;
pub(crate) const LBFGSERR_INVALID_XTOL: i32 = -1008;
pub(crate) const LBFGSERR_INVALID_MAXLINESEARCH: i32 = -1007;
pub(crate) const LBFGSERR_INVALID_ORTHANTWISE: i32 = -1006;
pub(crate) const LBFGSERR_INVALID_ORTHANTWISE_START: i32 = -1005;
pub(crate) const LBFGSERR_INVALID_ORTHANTWISE_END: i32 = -1004;
pub(crate) const LBFGSERR_OUTOFINTERVAL: i32 = -1003;
pub(crate) const LBFGSERR_INCORRECT_TMINMAX: i32 = -1002;
pub(crate) const LBFGSERR_ROUNDING_ERROR: i32 = -1001;
pub(crate) const LBFGSERR_MINIMUMSTEP: i32 = -1000;
pub(crate) const LBFGSERR_MAXIMUMSTEP: i32 = -999;
pub(crate) const LBFGSERR_MAXIMUMLINESEARCH: i32 = -998;
pub(crate) const LBFGSERR_MAXIMUMITERATION: i32 = -997;
pub(crate) const LBFGSERR_WIDTHTOOSMALL: i32 = -996;
pub(crate) const LBFGSERR_INVALIDPARAMETERS: i32 = -995;
pub(crate) const LBFGSERR_INCREASEGRADIENT: i32 = -994;

pub(crate) const LBFGS_LINESEARCH_MORETHUENTE: u32 = 0;
pub(crate) const LBFGS_LINESEARCH_BACKTRACKING_ARMIJO: u32 = 1;
pub(crate) const LBFGS_LINESEARCH_BACKTRACKING_WOLFE: u32 = 2;
pub(crate) const LBFGS_LINESEARCH_BACKTRACKING: u32 = 2;
pub(crate) const LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE: u32 = 3;

/// Convert internal integer return code to Result.
fn code_to_result(code: i32, fx: f64) -> Result<LbfgsResult, LbfgsError> {
    match code {
        LBFGS_SUCCESS => Ok(LbfgsResult { convergence: Convergence::Gradient, fx }),
        LBFGS_STOP => Ok(LbfgsResult { convergence: Convergence::Delta, fx }),
        LBFGS_ALREADY_MINIMIZED => Ok(LbfgsResult { convergence: Convergence::AlreadyMinimized, fx }),
        LBFGSERR_UNKNOWNERROR => Err(LbfgsError::UnknownError),
        LBFGSERR_LOGICERROR => Err(LbfgsError::LogicError),
        LBFGSERR_OUTOFMEMORY => Err(LbfgsError::OutOfMemory),
        LBFGSERR_CANCELED => Err(LbfgsError::Canceled),
        LBFGSERR_INVALID_N => Err(LbfgsError::InvalidN),
        LBFGSERR_INVALID_EPSILON => Err(LbfgsError::InvalidEpsilon),
        LBFGSERR_INVALID_TESTPERIOD => Err(LbfgsError::InvalidTestPeriod),
        LBFGSERR_INVALID_DELTA => Err(LbfgsError::InvalidDelta),
        LBFGSERR_INVALID_LINESEARCH => Err(LbfgsError::InvalidLineSearch),
        LBFGSERR_INVALID_MINSTEP => Err(LbfgsError::InvalidMinStep),
        LBFGSERR_INVALID_MAXSTEP => Err(LbfgsError::InvalidMaxStep),
        LBFGSERR_INVALID_FTOL => Err(LbfgsError::InvalidFtol),
        LBFGSERR_INVALID_WOLFE => Err(LbfgsError::InvalidWolfe),
        LBFGSERR_INVALID_GTOL => Err(LbfgsError::InvalidGtol),
        LBFGSERR_INVALID_XTOL => Err(LbfgsError::InvalidXtol),
        LBFGSERR_INVALID_MAXLINESEARCH => Err(LbfgsError::InvalidMaxLineSearch),
        LBFGSERR_INVALID_ORTHANTWISE => Err(LbfgsError::InvalidOrthantwise),
        LBFGSERR_INVALID_ORTHANTWISE_START => Err(LbfgsError::InvalidOrthantwiseStart),
        LBFGSERR_INVALID_ORTHANTWISE_END => Err(LbfgsError::InvalidOrthantwiseEnd),
        LBFGSERR_OUTOFINTERVAL => Err(LbfgsError::OutOfInterval),
        LBFGSERR_INCORRECT_TMINMAX => Err(LbfgsError::IncorrectTMinMax),
        LBFGSERR_ROUNDING_ERROR => Err(LbfgsError::RoundingError),
        LBFGSERR_MINIMUMSTEP => Err(LbfgsError::MinimumStep),
        LBFGSERR_MAXIMUMSTEP => Err(LbfgsError::MaximumStep),
        LBFGSERR_MAXIMUMLINESEARCH => Err(LbfgsError::MaximumLineSearch),
        LBFGSERR_MAXIMUMITERATION => Err(LbfgsError::MaximumIteration),
        LBFGSERR_WIDTHTOOSMALL => Err(LbfgsError::WidthTooSmall),
        LBFGSERR_INVALIDPARAMETERS => Err(LbfgsError::InvalidParameters),
        LBFGSERR_INCREASEGRADIENT => Err(LbfgsError::IncreaseGradient),
        _ => Err(LbfgsError::Canceled), // progress callback returned non-zero
    }
}

/// Convert LbfgsParam to the internal lbfgs_parameter_t used by lbfgs_core.
fn param_to_internal(p: &LbfgsParam) -> lbfgs_core::InternalParam {
    let ls_code = match p.linesearch {
        LineSearch::MoreThuente => 0,
        LineSearch::BacktrackingArmijo => 1,
        LineSearch::BacktrackingWolfe => 2,
        LineSearch::BacktrackingStrongWolfe => 3,
    };
    let (ow_c, ow_start, ow_end) = match p.orthantwise {
        Some(ow) => (ow.c, ow.start, ow.end),
        None => (0.0, 0, -1),
    };
    lbfgs_core::InternalParam {
        m: p.m,
        epsilon: p.epsilon,
        past: p.past,
        delta: p.delta,
        max_iterations: p.max_iterations,
        linesearch: ls_code,
        max_linesearch: p.max_linesearch,
        min_step: p.min_step,
        max_step: p.max_step,
        ftol: p.ftol,
        wolfe: p.wolfe,
        gtol: p.gtol,
        xtol: p.xtol,
        orthantwise_c: ow_c,
        orthantwise_start: ow_start,
        orthantwise_end: ow_end,
    }
}

/// Run L-BFGS optimization.
///
/// Safe wrapper over a direct port of `int lbfgs(...)` from the C library;
/// produces bit-exact results matching libLBFGS.
///
/// - `x`: initial variable values (modified in-place to the optimized solution)
/// - `evaluate`: closure `|x: &[f64], g: &mut [f64], step: f64| -> f64` — compute objective
///   value and write gradient into `g`. Returns the objective function value.
/// - `progress`: optional closure receiving progress info, return `false` to cancel
/// - `param`: optimization parameters (use `LbfgsParam::default()` for defaults)
///
/// Returns `Ok(LbfgsResult)` on convergence (gradient norm, delta, or already
/// minimized) or `Err(LbfgsError)` on failure or user cancellation.
pub fn lbfgs(
    x: &mut [f64],
    evaluate: impl FnMut(&[f64], &mut [f64], f64) -> f64,
    progress: Option<&mut dyn FnMut(&ProgressReport) -> bool>,
    param: &LbfgsParam,
) -> Result<LbfgsResult, LbfgsError> {
    let internal_param = param_to_internal(param);
    let mut fx = 0.0f64;
    let code = lbfgs_core::lbfgs_impl_safe(
        x,
        &mut fx,
        evaluate,
        progress,
        &internal_param,
    );
    code_to_result(code, fx)
}
