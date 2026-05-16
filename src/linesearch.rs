//! Line search implementations — direct translation of the C line search functions.

use crate::arithmetic::*;
use crate::lbfgs_core::InternalParam;
use crate::owlqn::*;
use crate::*;

// ---------------------------------------------------------------------------
// Interpolation helpers (from C macros CUBIC_MINIMIZER, etc.)
// ---------------------------------------------------------------------------

/// Return the smaller of two values using `if a <= b { a } else { b }` to match C semantics.
#[inline]
fn min2(a: f64, b: f64) -> f64 {
    if a <= b { a } else { b }
}

/// Return the larger of two values using `if a >= b { a } else { b }` to match C semantics.
#[inline]
fn max2(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}

/// Return the maximum of three values via nested `max2`.
#[inline]
fn max3(a: f64, b: f64, c: f64) -> f64 {
    max2(max2(a, b), c)
}

/// Find the minimizer of the cubic interpolant through `(u, fu, du)` and `(v, fv, dv)`.
#[inline]
fn cubic_minimizer(u: f64, fu: f64, du: f64, v: f64, fv: f64, dv: f64) -> f64 {
    let d = v - u;
    let theta = (fu - fv) * 3.0 / d + du + dv;
    let p = theta.abs();
    let q = du.abs();
    let r = dv.abs();
    let s = max3(p, q, r);
    let a = theta / s;
    let mut gamma = s * (a * a - (du / s) * (dv / s)).sqrt();
    if v < u {
        gamma = -gamma;
    }
    let p2 = gamma - du + theta;
    let q2 = gamma - du + gamma + dv;
    let r2 = p2 / q2;
    u + r2 * d
}

/// Cubic minimizer variant that clamps the discriminant to `>= 0` (for the non-bracketed case) and falls back to `xmin`/`xmax` when the interpolant fails.
#[inline]
fn cubic_minimizer2(
    u: f64, fu: f64, du: f64,
    v: f64, fv: f64, dv: f64,
    xmin: f64, xmax: f64,
) -> f64 {
    let d = v - u;
    let theta = (fu - fv) * 3.0 / d + du + dv;
    let p = theta.abs();
    let q = du.abs();
    let r = dv.abs();
    let s = max3(p, q, r);
    let a = theta / s;
    let mut gamma = s * max2(0.0, a * a - (du / s) * (dv / s)).sqrt();
    if u < v {
        gamma = -gamma;
    }
    let p2 = gamma - dv + theta;
    let q2 = gamma - dv + gamma + du;
    let r2 = p2 / q2;
    if r2 < 0.0 && gamma != 0.0 {
        v - r2 * d
    } else if d > 0.0 {
        xmax
    } else {
        xmin
    }
}

/// Find the minimizer of the quadratic interpolant through `(u, fu, du)` and `(v, fv)`.
#[inline]
fn quad_minimizer(u: f64, fu: f64, du: f64, v: f64, fv: f64) -> f64 {
    let a = v - u;
    u + du / ((fu - fv) / a + du) / 2.0 * a
}

/// Find the minimizer of the quadratic interpolant through `(u, du)` and `(v, dv)` (gradients only — the secant step).
#[inline]
fn quad_minimizer2(u: f64, du: f64, v: f64, dv: f64) -> f64 {
    let a = u - v;
    v + dv / (dv - du) * a
}

// ---------------------------------------------------------------------------
// update_trial_interval
// ---------------------------------------------------------------------------

/// Compute a new safeguarded trial value `t` for the line search and update the interval of
/// uncertainty `[x, y]`. `x` holds the step with the lowest function value seen so far; `t` is
/// the current trial step. The four cases (higher `ft`; lower `ft` with opposite-sign
/// derivatives; lower `ft` with same-sign derivatives and decreasing derivative magnitude; lower
/// `ft` otherwise) select between a cubic interpolant and a quadratic/secant minimizer, after
/// which the result is clamped to `[tmin, tmax]` and, when the minimum is bracketed, to at most
/// 66% of the interval width as a safeguard. Returns `0` on success, or one of
/// `LBFGSERR_OUTOFINTERVAL`, `LBFGSERR_INCREASEGRADIENT`, `LBFGSERR_INCORRECT_TMINMAX`.
/// Reference: Moré & Thuente, "Line Search Algorithms with Guaranteed Sufficient Decrease,"
/// ACM TOMS 20(3), 1994.
pub fn update_trial_interval(
    x: &mut f64, fx: &mut f64, dx: &mut f64,
    y: &mut f64, fy: &mut f64, dy: &mut f64,
    t: &mut f64, ft: &mut f64, dt: &mut f64,
    tmin: f64, tmax: f64,
    brackt: &mut i32,
) -> i32 {
    let bound;
    let dsign = fsigndiff(*dt, *dx);
    let mc;
    let mq;
    let mut newt;

    if *brackt != 0 {
        if *t <= min2(*x, *y) || max2(*x, *y) <= *t {
            return LBFGSERR_OUTOFINTERVAL;
        }
        if 0.0 <= *dx * (*t - *x) {
            return LBFGSERR_INCREASEGRADIENT;
        }
        if tmax < tmin {
            return LBFGSERR_INCORRECT_TMINMAX;
        }
    }

    if *fx < *ft {
        *brackt = 1;
        bound = 1;
        mc = cubic_minimizer(*x, *fx, *dx, *t, *ft, *dt);
        mq = quad_minimizer(*x, *fx, *dx, *t, *ft);
        if (mc - *x).abs() < (mq - *x).abs() {
            newt = mc;
        } else {
            newt = mc + 0.5 * (mq - mc);
        }
    } else if dsign {
        *brackt = 1;
        bound = 0;
        mc = cubic_minimizer(*x, *fx, *dx, *t, *ft, *dt);
        mq = quad_minimizer2(*x, *dx, *t, *dt);
        if (mc - *t).abs() > (mq - *t).abs() {
            newt = mc;
        } else {
            newt = mq;
        }
    } else if (*dt).abs() < (*dx).abs() {
        bound = 1;
        mc = cubic_minimizer2(*x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
        mq = quad_minimizer2(*x, *dx, *t, *dt);
        if *brackt != 0 {
            if (*t - mc).abs() < (*t - mq).abs() {
                newt = mc;
            } else {
                newt = mq;
            }
        } else {
            if (*t - mc).abs() > (*t - mq).abs() {
                newt = mc;
            } else {
                newt = mq;
            }
        }
    } else {
        bound = 0;
        if *brackt != 0 {
            newt = cubic_minimizer(*t, *ft, *dt, *y, *fy, *dy);
        } else if *x < *t {
            newt = tmax;
        } else {
            newt = tmin;
        }
    }

    if *fx < *ft {
        *y = *t;
        *fy = *ft;
        *dy = *dt;
    } else {
        if dsign {
            *y = *x;
            *fy = *fx;
            *dy = *dx;
        }
        *x = *t;
        *fx = *ft;
        *dx = *dt;
    }

    if tmax < newt {
        newt = tmax;
    }
    if newt < tmin {
        newt = tmin;
    }

    if *brackt != 0 && bound != 0 {
        let mq2 = *x + 0.66 * (*y - *x);
        if *x < *y {
            if mq2 < newt {
                newt = mq2;
            }
        } else {
            if newt < mq2 {
                newt = mq2;
            }
        }
    }

    *t = newt;
    0
}

// ---------------------------------------------------------------------------
// line_search_backtracking
// ---------------------------------------------------------------------------

/// Backtracking line search satisfying the Armijo, regular Wolfe, or strong Wolfe condition
/// depending on `param.linesearch`. Starting from the initial step `*stp`, the search trial
/// value is multiplied by `0.5` when the sufficient-decrease test fails and by `2.1` when the
/// curvature condition is too loose, until the requested condition is met or the step leaves
/// `[min_step, max_step]`. On success returns the number of function evaluations performed;
/// otherwise returns a negative error code (`LBFGSERR_INVALIDPARAMETERS`,
/// `LBFGSERR_INCREASEGRADIENT`, `LBFGSERR_MINIMUMSTEP`, `LBFGSERR_MAXIMUMSTEP`, or
/// `LBFGSERR_MAXIMUMLINESEARCH`).
pub fn line_search_backtracking(
    _n: usize,
    x: &mut [f64],
    f: &mut f64,
    g: &mut [f64],
    s: &mut [f64],
    stp: &mut f64,
    xp: &[f64],
    _gp: &[f64],
    _wa: &mut [f64],
    eval_fn: &mut impl FnMut(&[f64], &mut [f64], f64) -> f64,
    param: &InternalParam,
) -> i32 {
    let mut count = 0i32;
    let dec: f64 = 0.5;
    let inc: f64 = 2.1;

    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    let dginit = vecdot(g, s);
    if 0.0 < dginit {
        return LBFGSERR_INCREASEGRADIENT;
    }

    let finit = *f;
    let dgtest = param.ftol * dginit;

    loop {
        veccpy(x, xp);
        vecadd(x, s, *stp);

        *f = eval_fn(x, g, *stp);
        count += 1;

        let width;
        if *f > finit + *stp * dgtest {
            width = dec;
        } else {
            // Sufficient decrease (Armijo) condition satisfied.
            if param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_ARMIJO as i32 {
                return count;
            }

            // Check Wolfe condition.
            let dg = vecdot(g, s);
            if dg < param.wolfe * dginit {
                width = inc;
            } else {
                if param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE as i32 {
                    return count;
                }
                // Check strong Wolfe condition.
                if dg > -param.wolfe * dginit {
                    width = dec;
                } else {
                    return count;
                }
            }
        }

        if *stp < param.min_step {
            return LBFGSERR_MINIMUMSTEP;
        }
        if *stp > param.max_step {
            return LBFGSERR_MAXIMUMSTEP;
        }
        if param.max_linesearch <= count {
            return LBFGSERR_MAXIMUMLINESEARCH;
        }

        *stp *= width;
    }
}

// ---------------------------------------------------------------------------
// line_search_backtracking_owlqn
// ---------------------------------------------------------------------------

/// Backtracking line search specialized for OWL-QN (L1-regularized) optimization. The orthant
/// for the new point is chosen from the previous variables (or `-gp` where `xp == 0`), each
/// trial point is projected back onto that orthant via `owlqn_project`, and the L1-regularized
/// objective `f + c * ||x||_1` is tested against the OWL-QN sufficient-decrease condition based
/// on the inner product of the actual displacement with the pseudo-gradient. The trial step is
/// halved on failure. Returns the number of evaluations on success or a negative error code
/// (`LBFGSERR_INVALIDPARAMETERS`, `LBFGSERR_MINIMUMSTEP`, `LBFGSERR_MAXIMUMSTEP`,
/// `LBFGSERR_MAXIMUMLINESEARCH`).
pub fn line_search_backtracking_owlqn(
    n: usize,
    x: &mut [f64],
    f: &mut f64,
    g: &mut [f64],
    s: &mut [f64],
    stp: &mut f64,
    xp: &[f64],
    gp: &[f64],
    wp: &mut [f64],
    eval_fn: &mut impl FnMut(&[f64], &mut [f64], f64) -> f64,
    param: &InternalParam,
) -> i32 {
    let mut count = 0i32;
    let width: f64 = 0.5;
    let finit = *f;

    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    for i in 0..n {
        wp[i] = if xp[i] == 0.0 { -gp[i] } else { xp[i] };
    }

    loop {
        veccpy(x, xp);
        vecadd(x, s, *stp);
        owlqn_project(x, wp, param.orthantwise_start as usize, param.orthantwise_end as usize);

        *f = eval_fn(x, g, *stp);

        let norm =
            owlqn_x1norm(x, param.orthantwise_start as usize, param.orthantwise_end as usize);
        *f += norm * param.orthantwise_c;

        count += 1;

        let mut dgtest = 0.0;
        for i in 0..n {
            dgtest += (x[i] - xp[i]) * gp[i];
        }

        if *f <= finit + param.ftol * dgtest {
            return count;
        }

        if *stp < param.min_step {
            return LBFGSERR_MINIMUMSTEP;
        }
        if *stp > param.max_step {
            return LBFGSERR_MAXIMUMSTEP;
        }
        if param.max_linesearch <= count {
            return LBFGSERR_MAXIMUMLINESEARCH;
        }

        *stp *= width;
    }
}

// ---------------------------------------------------------------------------
// line_search_morethuente
// ---------------------------------------------------------------------------

/// Moré-Thuente line search satisfying the strong Wolfe condition by iteratively refining an
/// interval of uncertainty. The endpoints `stx` (best step so far) and `sty` (other endpoint)
/// track the bracket; on each iteration a new trial step is produced by `update_trial_interval`
/// using cubic and quadratic interpolants, optionally on the auxiliary "modified" function
/// `f - stp * dgtest` during stage 1. The search terminates when the sufficient-decrease and
/// curvature tests both hold, when the interval width falls below `xtol * stmax`, or when a
/// boundary is reached. Returns the number of function evaluations on success or a negative
/// error code (`LBFGSERR_INVALIDPARAMETERS`, `LBFGSERR_INCREASEGRADIENT`,
/// `LBFGSERR_ROUNDING_ERROR`, `LBFGSERR_MINIMUMSTEP`, `LBFGSERR_MAXIMUMSTEP`,
/// `LBFGSERR_WIDTHTOOSMALL`, or `LBFGSERR_MAXIMUMLINESEARCH`).
/// Reference: Moré & Thuente, "Line Search Algorithms with Guaranteed Sufficient Decrease,"
/// ACM TOMS 20(3):286–307, 1994.
pub fn line_search_morethuente(
    _n: usize,
    x: &mut [f64],
    f: &mut f64,
    g: &mut [f64],
    s: &mut [f64],
    stp: &mut f64,
    xp: &[f64],
    _gp: &[f64],
    _wa: &mut [f64],
    eval_fn: &mut impl FnMut(&[f64], &mut [f64], f64) -> f64,
    param: &InternalParam,
) -> i32 {
    let mut count = 0i32;
    let mut brackt = 0i32;
    let mut stage1 = 1i32;
    let mut uinfo = 0i32;

    if *stp <= 0.0 {
        return LBFGSERR_INVALIDPARAMETERS;
    }

    let dginit = vecdot(g, s);
    if 0.0 < dginit {
        return LBFGSERR_INCREASEGRADIENT;
    }

    let finit = *f;
    let dgtest = param.ftol * dginit;
    let mut width = param.max_step - param.min_step;
    let mut prev_width = 2.0 * width;

    let mut stx = 0.0;
    let mut sty = 0.0;
    let mut fx = finit;
    let mut fy = finit;
    let mut dgx = dginit;
    let mut dgy = dginit;

    loop {
        let (stmin, stmax) = if brackt != 0 {
            (min2(stx, sty), max2(stx, sty))
        } else {
            (stx, *stp + 4.0 * (*stp - stx))
        };

        if *stp < param.min_step {
            *stp = param.min_step;
        }
        if param.max_step < *stp {
            *stp = param.max_step;
        }

        if (brackt != 0
            && ((*stp <= stmin || stmax <= *stp)
                || param.max_linesearch <= count + 1
                || uinfo != 0))
            || (brackt != 0 && (stmax - stmin <= param.xtol * stmax))
        {
            *stp = stx;
        }

        veccpy(x, xp);
        vecadd(x, s, *stp);

        *f = eval_fn(x, g, *stp);
        let dg = vecdot(g, s);

        let ftest1 = finit + *stp * dgtest;
        count += 1;

        if brackt != 0 && ((*stp <= stmin || stmax <= *stp) || uinfo != 0) {
            return LBFGSERR_ROUNDING_ERROR;
        }
        if *stp == param.max_step && *f <= ftest1 && dg <= dgtest {
            return LBFGSERR_MAXIMUMSTEP;
        }
        if *stp == param.min_step && (ftest1 < *f || dgtest <= dg) {
            return LBFGSERR_MINIMUMSTEP;
        }
        if brackt != 0 && (stmax - stmin) <= param.xtol * stmax {
            return LBFGSERR_WIDTHTOOSMALL;
        }
        if param.max_linesearch <= count {
            return LBFGSERR_MAXIMUMLINESEARCH;
        }
        if *f <= ftest1 && dg.abs() <= param.gtol * (-dginit) {
            return count;
        }

        if stage1 != 0 && *f <= ftest1 && min2(param.ftol, param.gtol) * dginit <= dg {
            stage1 = 0;
        }

        if stage1 != 0 && ftest1 < *f && *f <= fx {
            let mut fm = *f - *stp * dgtest;
            let mut fxm = fx - stx * dgtest;
            let mut fym = fy - sty * dgtest;
            let mut dgm = dg - dgtest;
            let mut dgxm = dgx - dgtest;
            let mut dgym = dgy - dgtest;

            uinfo = update_trial_interval(
                &mut stx, &mut fxm, &mut dgxm,
                &mut sty, &mut fym, &mut dgym,
                stp, &mut fm, &mut dgm,
                stmin, stmax, &mut brackt,
            );

            fx = fxm + stx * dgtest;
            fy = fym + sty * dgtest;
            dgx = dgxm + dgtest;
            dgy = dgym + dgtest;
        } else {
            uinfo = update_trial_interval(
                &mut stx, &mut fx, &mut dgx,
                &mut sty, &mut fy, &mut dgy,
                stp, f, &mut { dg },
                stmin, stmax, &mut brackt,
            );
        }

        if brackt != 0 {
            if 0.66 * prev_width <= (sty - stx).abs() {
                *stp = stx + 0.5 * (sty - stx);
            }
            prev_width = width;
            width = (sty - stx).abs();
        }
    }
}
