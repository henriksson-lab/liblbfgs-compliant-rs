//! C FFI bindings for comparison testing against the original C library.
//! Only available with the `compare-c` feature.

use std::os::raw::{c_char, c_int, c_void};

/// Floating-point type used by libLBFGS (`double` in the default build).
pub type lbfgsfloatval_t = f64;

/// C-layout mirror of `lbfgs_parameter_t` from `lbfgs.h`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct lbfgs_parameter_t {
    pub m: c_int,
    pub epsilon: f64,
    pub past: c_int,
    pub delta: f64,
    pub max_iterations: c_int,
    pub linesearch: c_int,
    pub max_linesearch: c_int,
    pub min_step: f64,
    pub max_step: f64,
    pub ftol: f64,
    pub wolfe: f64,
    pub gtol: f64,
    pub xtol: f64,
    pub orthantwise_c: f64,
    pub orthantwise_start: c_int,
    pub orthantwise_end: c_int,
}

/// Callback type for the objective function and gradient evaluation; matches
/// `lbfgs_evaluate_t` in `lbfgs.h`.
pub type lbfgs_evaluate_t = Option<
    unsafe extern "C" fn(
        instance: *mut c_void,
        x: *const f64,
        g: *mut f64,
        n: c_int,
        step: f64,
    ) -> f64,
>;

/// Callback type for per-iteration progress reporting; matches
/// `lbfgs_progress_t` in `lbfgs.h`. Return non-zero to cancel the optimization.
pub type lbfgs_progress_t = Option<
    unsafe extern "C" fn(
        instance: *mut c_void,
        x: *const f64,
        g: *const f64,
        fx: f64,
        xnorm: f64,
        gnorm: f64,
        step: f64,
        n: c_int,
        k: c_int,
        ls: c_int,
    ) -> c_int,
>;

// Imports from the original libLBFGS C library, linked when the `compare-c`
// feature is enabled. Used by the bit-exact conformance tests to compare the
// Rust port's output against the reference implementation.
unsafe extern "C" {
    /// Start an L-BFGS optimization. Returns zero on success or a negative
    /// `LBFGSERR_*` code on error.
    pub fn lbfgs(
        n: c_int,
        x: *mut f64,
        ptr_fx: *mut f64,
        proc_evaluate: lbfgs_evaluate_t,
        proc_progress: lbfgs_progress_t,
        instance: *mut c_void,
        param: *mut lbfgs_parameter_t,
    ) -> c_int;

    /// Initialize an `lbfgs_parameter_t` to libLBFGS default values.
    pub fn lbfgs_parameter_init(param: *mut lbfgs_parameter_t);
    /// Allocate an aligned array of `n` variables (required by SSE/SSE2 builds).
    pub fn lbfgs_malloc(n: c_int) -> *mut f64;
    /// Free an array previously allocated by `lbfgs_malloc`.
    pub fn lbfgs_free(x: *mut f64);
    /// Return a static string description for a return code from `lbfgs()`.
    pub fn lbfgs_strerror(err: c_int) -> *const c_char;
}
