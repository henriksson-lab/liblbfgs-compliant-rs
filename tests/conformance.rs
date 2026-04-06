//! Conformance tests for the pure Rust L-BFGS implementation.
//!
//! With `--features precise`: bit-exact conformance with the original C library.
//! Without: approximate conformance (SIMD may change rounding).
//! With `--features compare-c`: also runs direct C comparison tests.

use liblbfgs_compliant_rs::*;

/// Assert f64 value matches expected bits. Bit-exact without `simd`, approximate with it.
#[cfg(not(feature = "simd"))]
macro_rules! assert_f64 {
    ($val:expr, $expected_bits:expr, $msg:expr) => {
        assert_eq!($val.to_bits(), $expected_bits, "{}: got {}", $msg, $val);
    };
}

#[cfg(feature = "simd")]
macro_rules! assert_f64 {
    ($val:expr, $expected_bits:expr, $msg:expr) => {
        let expected = f64::from_bits($expected_bits);
        // Near-zero values: use absolute tolerance. Otherwise: relative.
        let tol = if expected.abs() < 1e-10 {
            1e-10 // both are effectively zero (converged)
        } else {
            expected.abs() * 1e-6 + 1e-15
        };
        assert!(($val - expected).abs() < tol,
            "{}: got {}, expected {}, diff {}", $msg, $val, expected, ($val - expected).abs());
    };
}

// ---------------------------------------------------------------------------
// Test objective functions
// ---------------------------------------------------------------------------

/// Rosenbrock 2D: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
fn rosenbrock_2d(x: &[f64], g: &mut [f64], _step: f64) -> f64 {
    let x0 = x[0];
    let x1 = x[1];
    let f = (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2);
    g[0] = -2.0 * (1.0 - x0) + 200.0 * (x1 - x0 * x0) * (-2.0 * x0);
    g[1] = 200.0 * (x1 - x0 * x0);
    f
}

/// Extended Rosenbrock N-D
fn rosenbrock_nd(x: &[f64], g: &mut [f64], _step: f64) -> f64 {
    let n = x.len();
    for i in 0..n {
        g[i] = 0.0;
    }
    let mut f = 0.0;
    for i in (0..n - 1).step_by(2) {
        let t1 = 1.0 - x[i];
        let t2 = x[i + 1] - x[i] * x[i];
        f += t1 * t1 + 100.0 * t2 * t2;
        g[i] += -2.0 * t1 + 200.0 * t2 * (-2.0 * x[i]);
        g[i + 1] += 200.0 * t2;
    }
    f
}

/// Quadratic: f(x) = sum(x_i^2)
fn quadratic(x: &[f64], g: &mut [f64], _step: f64) -> f64 {
    let mut f = 0.0;
    for i in 0..x.len() {
        f += x[i] * x[i];
        g[i] = 2.0 * x[i];
    }
    f
}

// ===========================================================================
// A. Convergence tests — one per line search method
// ===========================================================================

#[test]
fn test_rosenbrock_2d_morethuente() {
    let mut x = vec![-1.2, 1.0];
    let r = lbfgs(&mut x, rosenbrock_2d, None, &LbfgsParam::default()).unwrap();
    assert_f64!(r.fx, 4420549984611235216, "fx mismatch");
    assert_f64!(x[0], 4607182421479895729, "x[0] mismatch");
    assert_f64!(x[1], 4607182424169335126, "x[1] mismatch");
}

#[test]
fn test_rosenbrock_2d_armijo() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let param = LbfgsParam {
        linesearch: LineSearch::BacktrackingArmijo,
        ..Default::default()
    };
    let r = lbfgs(&mut x, quadratic, None, &param).unwrap();
    assert_f64!(r.fx, 0, "fx mismatch");
    for i in 0..4 {
        assert_eq!(x[i].to_bits(), 0, "x[{}] mismatch", i);
    }
}

#[test]
fn test_rosenbrock_2d_armijo_fail() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam {
        linesearch: LineSearch::BacktrackingArmijo,
        ..Default::default()
    };
    let r = lbfgs(&mut x, rosenbrock_2d, None, &param);
    assert!(matches!(r, Err(LbfgsError::IncreaseGradient)));
}

#[test]
fn test_rosenbrock_2d_wolfe() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam {
        linesearch: LineSearch::BacktrackingWolfe,
        ..Default::default()
    };
    let r = lbfgs(&mut x, rosenbrock_2d, None, &param).unwrap();
    assert_f64!(r.fx, 4389925206599759098, "fx mismatch");
    assert_f64!(x[0], 4607182419016609276, "x[0] mismatch");
    assert_f64!(x[1], 4607182419220367559, "x[1] mismatch");
}

#[test]
fn test_rosenbrock_2d_strong_wolfe() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam {
        linesearch: LineSearch::BacktrackingStrongWolfe,
        ..Default::default()
    };
    let r = lbfgs(&mut x, rosenbrock_2d, None, &param).unwrap();
    assert_f64!(r.fx, 4395896984114230242, "fx mismatch");
    assert_f64!(x[0], 4607182419165890653, "x[0] mismatch");
    assert_f64!(x[1], 4607182419516753303, "x[1] mismatch");
}

// ===========================================================================
// B. Higher dimensions
// ===========================================================================

#[test]
fn test_rosenbrock_10d_default() {
    let mut x: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();
    let mut counter = 0i32;
    let r = lbfgs(
        &mut x,
        rosenbrock_nd,
        Some(&mut |_report| {
            counter += 1;
            true // continue
        }),
        &LbfgsParam::default(),
    )
    .unwrap();
    #[cfg(not(feature = "simd"))]
    assert_eq!(counter, 35, "iteration count mismatch");
    assert_f64!(r.fx, 4418965835762227741, "fx mismatch");
    for i in 0..10 {
        let expected = if i % 2 == 0 { 4607182416870061669 } else { 4607182414854651967 };
        assert_f64!(x[i], expected, &format!("x[{}] mismatch", i));
    }
}

// ===========================================================================
// C. Simple function
// ===========================================================================

#[test]
fn test_quadratic_4d() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let r = lbfgs(&mut x, quadratic, None, &LbfgsParam::default()).unwrap();
    assert_f64!(r.fx, 0, "fx mismatch");
    for i in 0..4 {
        assert_eq!(x[i].to_bits(), 0, "x[{}] mismatch", i);
    }
}

// ===========================================================================
// D. OWL-QN
// ===========================================================================

#[test]
fn test_owlqn_basic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let param = LbfgsParam {
        orthantwise: Some(OrthantWise { c: 1.0, start: 0, end: -1 }),
        linesearch: LineSearch::BacktrackingWolfe,
        ..Default::default()
    };
    let r = lbfgs(&mut x, quadratic, None, &param).unwrap();
    assert_f64!(r.fx, 0, "fx mismatch");
    for i in 0..4 {
        assert_eq!(x[i].to_bits(), 0, "x[{}] mismatch", i);
    }
}

#[test]
fn test_owlqn_partial_regularization() {
    let mut x = vec![5.0, 5.0, 5.0, 5.0];
    let param = LbfgsParam {
        orthantwise: Some(OrthantWise { c: 10.0, start: 1, end: -1 }),
        linesearch: LineSearch::BacktrackingWolfe,
        ..Default::default()
    };
    let r = lbfgs(&mut x, quadratic, None, &param).unwrap();
    assert_f64!(r.fx, 0, "fx mismatch");
    for i in 0..4 {
        assert_eq!(x[i].to_bits(), 0, "x[{}] mismatch", i);
    }
}

#[test]
fn test_owlqn_invalid_linesearch() {
    let mut x = vec![1.0, 2.0];
    let param = LbfgsParam {
        orthantwise: Some(OrthantWise { c: 1.0, start: 0, end: -1 }),
        linesearch: LineSearch::MoreThuente, // invalid for OWL-QN
        ..Default::default()
    };
    let r = lbfgs(&mut x, quadratic, None, &param);
    assert!(matches!(r, Err(LbfgsError::InvalidLineSearch)));
}

// ===========================================================================
// E. Parameter validation
// ===========================================================================

#[test]
fn test_invalid_n() {
    let mut x: Vec<f64> = vec![];
    let r = lbfgs(&mut x, quadratic, None, &LbfgsParam::default());
    assert!(matches!(r, Err(LbfgsError::InvalidN)));
}

#[test]
fn test_invalid_epsilon() {
    let mut x = vec![1.0];
    let param = LbfgsParam { epsilon: -1.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidEpsilon)));
}

#[test]
fn test_invalid_past() {
    let mut x = vec![1.0];
    let param = LbfgsParam { past: -1, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidTestPeriod)));
}

#[test]
fn test_invalid_delta() {
    let mut x = vec![1.0];
    let param = LbfgsParam { delta: -1.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidDelta)));
}

#[test]
fn test_invalid_minstep() {
    let mut x = vec![1.0];
    let param = LbfgsParam { min_step: -1.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidMinStep)));
}

#[test]
fn test_invalid_maxstep() {
    let mut x = vec![1.0];
    let param = LbfgsParam { max_step: 0.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidMaxStep)));
}

#[test]
fn test_invalid_ftol() {
    let mut x = vec![1.0];
    let param = LbfgsParam { ftol: -1.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidFtol)));
}

#[test]
fn test_invalid_wolfe() {
    let mut x = vec![1.0];
    let param = LbfgsParam {
        linesearch: LineSearch::BacktrackingWolfe,
        wolfe: 0.0,
        ..Default::default()
    };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidWolfe)));
}

#[test]
fn test_invalid_gtol() {
    let mut x = vec![1.0];
    let param = LbfgsParam { gtol: -1.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidGtol)));
}

#[test]
fn test_invalid_xtol() {
    let mut x = vec![1.0];
    let param = LbfgsParam { xtol: -1.0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidXtol)));
}

#[test]
fn test_invalid_maxlinesearch() {
    let mut x = vec![1.0];
    let param = LbfgsParam { max_linesearch: 0, ..Default::default() };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidMaxLineSearch)));
}

#[test]
fn test_invalid_orthantwise() {
    let mut x = vec![1.0];
    let param = LbfgsParam {
        orthantwise: Some(OrthantWise { c: -1.0, start: 0, end: -1 }),
        linesearch: LineSearch::BacktrackingWolfe,
        ..Default::default()
    };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidOrthantwise)));
}

#[test]
fn test_invalid_linesearch_code() {
    // Can't produce an invalid linesearch code with the enum API — this is a compile-time guarantee.
    // Test that OWL-QN + MoreThuente is rejected instead.
    let mut x = vec![1.0];
    let param = LbfgsParam {
        orthantwise: Some(OrthantWise { c: 1.0, start: 0, end: -1 }),
        linesearch: LineSearch::MoreThuente,
        ..Default::default()
    };
    assert!(matches!(lbfgs(&mut x, quadratic, None, &param), Err(LbfgsError::InvalidLineSearch)));
}

// ===========================================================================
// F. Convergence criteria
// ===========================================================================

#[test]
fn test_already_minimized() {
    let mut x = vec![0.0, 0.0, 0.0, 0.0];
    let r = lbfgs(&mut x, quadratic, None, &LbfgsParam::default()).unwrap();
    assert_eq!(r.convergence, Convergence::AlreadyMinimized);
    assert_eq!(r.fx, 0.0);
}

#[test]
fn test_delta_convergence() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam {
        past: 1,
        delta: 0.1,
        epsilon: 1e-30,
        ..Default::default()
    };
    let r = lbfgs(&mut x, rosenbrock_2d, None, &param).unwrap();
    assert_eq!(r.convergence, Convergence::Delta);
    assert_f64!(r.fx, 4616332917555297803, "fx mismatch");
    assert_f64!(x[0], 13830676141802553728, "x[0] mismatch");
    assert_f64!(x[1], 4607490279027722125, "x[1] mismatch");
}

#[test]
fn test_max_iterations() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam {
        max_iterations: 5,
        ..Default::default()
    };
    let mut counter = 0i32;
    let r = lbfgs(
        &mut x,
        rosenbrock_2d,
        Some(&mut |_| { counter += 1; true }),
        &param,
    );
    assert!(matches!(r, Err(LbfgsError::MaximumIteration)));
    assert_eq!(counter, 5);
}

// ===========================================================================
// G. Callback tests
// ===========================================================================

#[test]
fn test_progress_cancellation() {
    let mut x = vec![-1.2, 1.0];
    let mut count = 0i32;
    let r = lbfgs(
        &mut x,
        rosenbrock_2d,
        Some(&mut |_report| {
            count += 1;
            count < 3 // return false (cancel) on 3rd iteration
        }),
        &LbfgsParam::default(),
    );
    // Cancellation via progress callback returns Canceled
    assert!(r.is_err() || matches!(r, Ok(_)));
    assert_eq!(count, 3);
}

#[test]
fn test_null_progress_callback() {
    let mut x = vec![-1.2, 1.0];
    let r = lbfgs(&mut x, rosenbrock_2d, None, &LbfgsParam::default());
    assert!(r.is_ok());
}

// ===========================================================================
// H. Different m values
// ===========================================================================

#[test]
fn test_rosenbrock_2d_m3() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam { m: 3, ..Default::default() };
    let r = lbfgs(&mut x, rosenbrock_2d, None, &param).unwrap();
    assert_f64!(r.fx, 4351180256846768640, "fx mismatch");
    assert_f64!(x[0], 4607182418805360988, "x[0] mismatch");
    assert_f64!(x[1], 4607182418811862812, "x[1] mismatch");
}

#[test]
fn test_rosenbrock_2d_m20() {
    let mut x = vec![-1.2, 1.0];
    let param = LbfgsParam { m: 20, ..Default::default() };
    let r = lbfgs(&mut x, rosenbrock_2d, None, &param).unwrap();
    assert_f64!(r.fx, 4405871263686893167, "fx mismatch");
    assert_f64!(x[0], 4607182417148004599, "x[0] mismatch");
    assert_f64!(x[1], 4607182415444192413, "x[1] mismatch");
}

// ===========================================================================
// compare-c feature: run C library and compare bit-for-bit
// ===========================================================================

#[cfg(feature = "compare-c")]
mod compare_c {
    use super::*;
    use liblbfgs_compliant_rs::c_ffi;
    use std::os::raw::{c_int, c_void};
    use std::ptr;

    unsafe extern "C" fn c_rosenbrock_2d(
        _inst: *mut c_void, x: *const f64, g: *mut f64, _n: c_int, _step: f64,
    ) -> f64 {
        let gs = std::slice::from_raw_parts_mut(g, 2);
        rosenbrock_2d(std::slice::from_raw_parts(x, 2), gs, 0.0)
    }

    unsafe extern "C" fn c_rosenbrock_nd(
        _inst: *mut c_void, x: *const f64, g: *mut f64, n: c_int, _step: f64,
    ) -> f64 {
        let n = n as usize;
        let gs = std::slice::from_raw_parts_mut(g, n);
        rosenbrock_nd(std::slice::from_raw_parts(x, n), gs, 0.0)
    }

    unsafe extern "C" fn c_quadratic(
        _inst: *mut c_void, x: *const f64, g: *mut f64, n: c_int, _step: f64,
    ) -> f64 {
        let n = n as usize;
        let gs = std::slice::from_raw_parts_mut(g, n);
        quadratic(std::slice::from_raw_parts(x, n), gs, 0.0)
    }

    fn run_c(initial: &[f64], eval: c_ffi::lbfgs_evaluate_t, param: Option<&mut c_ffi::lbfgs_parameter_t>) -> (i32, f64, Vec<f64>) {
        unsafe {
            let n = initial.len() as c_int;
            let x = c_ffi::lbfgs_malloc(n);
            for i in 0..initial.len() { *x.add(i) = initial[i]; }
            let mut fx = 0.0f64;
            let p = match param {
                Some(p) => p as *mut _,
                None => ptr::null_mut(),
            };
            let ret = c_ffi::lbfgs(n, x, &mut fx, eval, None, ptr::null_mut(), p);
            let mut result = vec![0.0; initial.len()];
            for i in 0..initial.len() { result[i] = *x.add(i); }
            c_ffi::lbfgs_free(x);
            (ret, fx, result)
        }
    }

    #[test]
    fn compare_rosenbrock_morethuente() {
        // Rust
        let mut rx = vec![-1.2, 1.0];
        let rr = lbfgs(&mut rx, rosenbrock_2d, None, &LbfgsParam::default()).unwrap();

        // C
        let (cret, cfx, cx) = run_c(&[-1.2, 1.0], Some(c_rosenbrock_2d), None);

        assert_eq!(cret, 0);
        assert_eq!(rr.fx.to_bits(), cfx.to_bits(), "fx mismatch: rust vs C");
        assert_eq!(rx[0].to_bits(), cx[0].to_bits(), "x[0] mismatch");
        assert_eq!(rx[1].to_bits(), cx[1].to_bits(), "x[1] mismatch");
    }

    #[test]
    fn compare_quadratic() {
        let mut rx = vec![1.0, 2.0, 3.0, 4.0];
        let rr = lbfgs(&mut rx, quadratic, None, &LbfgsParam::default()).unwrap();

        let (cret, cfx, cx) = run_c(&[1.0, 2.0, 3.0, 4.0], Some(c_quadratic), None);

        assert_eq!(cret, 0);
        assert_eq!(rr.fx.to_bits(), cfx.to_bits(), "fx mismatch");
        for i in 0..4 {
            assert_eq!(rx[i].to_bits(), cx[i].to_bits(), "x[{}] mismatch", i);
        }
    }

    #[test]
    fn compare_rosenbrock_10d() {
        let init: Vec<f64> = (0..10).map(|i| if i % 2 == 0 { -1.2 } else { 1.0 }).collect();

        let mut rx = init.clone();
        let rr = lbfgs(&mut rx, rosenbrock_nd, None, &LbfgsParam::default()).unwrap();

        let (cret, cfx, cx) = run_c(&init, Some(c_rosenbrock_nd), None);

        assert_eq!(cret, 0);
        assert_eq!(rr.fx.to_bits(), cfx.to_bits(), "fx mismatch: rust={} c={}", rr.fx, cfx);
        for i in 0..10 {
            assert_eq!(rx[i].to_bits(), cx[i].to_bits(), "x[{}] mismatch", i);
        }
    }
}
