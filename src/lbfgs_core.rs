//! Main L-BFGS optimization implementation — direct translation of lbfgs() from lbfgs.c.

use crate::arithmetic::*;
use crate::linesearch::*;
use crate::owlqn::*;
use crate::*;

/// One entry of the limited-memory BFGS history, mirroring `tag_iteration_data` in lbfgs.c.
///
/// Each iteration stores the step `s = x_{k+1} - x_k`, the gradient difference
/// `y = g_{k+1} - g_k`, the inner product `ys = dot(y, s)`, and the scalar
/// `alpha` produced by the first (backward) pass of the two-loop recursion and
/// consumed by the second (forward) pass.
struct IterationData {
    alpha: f64,
    s: Vec<f64>,
    y: Vec<f64>,
    ys: f64,
}

/// Internal copy of `lbfgs_parameter_t` used by the optimizer.
///
/// Identical in meaning to the user-facing `LbfgsParam` (see `lib.rs`), but the
/// `linesearch` field is stored as a plain `i32` matching the C
/// `LBFGS_LINESEARCH_*` constants so it can be compared directly during
/// parameter validation and line-search dispatch.
#[derive(Debug, Copy, Clone)]
pub struct InternalParam {
    pub m: i32,
    pub epsilon: f64,
    pub past: i32,
    pub delta: f64,
    pub max_iterations: i32,
    pub linesearch: i32,
    pub max_linesearch: i32,
    pub min_step: f64,
    pub max_step: f64,
    pub ftol: f64,
    pub wolfe: f64,
    pub gtol: f64,
    pub xtol: f64,
    pub orthantwise_c: f64,
    pub orthantwise_start: i32,
    pub orthantwise_end: i32,
}

impl Default for InternalParam {
    fn default() -> Self {
        InternalParam {
            m: 6,
            epsilon: 1e-5,
            past: 0,
            delta: 1e-5,
            max_iterations: 0,
            linesearch: 0,
            max_linesearch: 40,
            min_step: 1e-20,
            max_step: 1e20,
            ftol: 1e-4,
            wolfe: 0.9,
            gtol: 0.9,
            xtol: 1.0e-16,
            orthantwise_c: 0.0,
            orthantwise_start: 0,
            orthantwise_end: -1,
        }
    }
}

/// Line-search routine selected after parameter validation.
///
/// Chosen from the `linesearch` field of [`InternalParam`] together with the
/// OWL-QN activation flag (`orthantwise_c != 0`): when OWL-QN is active only
/// `LBFGS_LINESEARCH_BACKTRACKING` is accepted (mapped to `BacktrackingOwlqn`);
/// otherwise `LBFGS_LINESEARCH_MORETHUENTE` maps to `MoreThuente` and the
/// Armijo / Wolfe / Strong-Wolfe variants all map to `Backtracking`.
#[derive(Copy, Clone)]
enum LineSearchMethod {
    MoreThuente,
    Backtracking,
    BacktrackingOwlqn,
}

/// L-BFGS optimization driver — direct translation of `int lbfgs(...)` from lbfgs.c.
///
/// Performs full parameter validation (returning the appropriate `LBFGSERR_*`
/// code on invalid input), selects the line-search method
/// (`MoreThuente` / `Backtracking` / `BacktrackingOwlqn`), allocates the
/// working vectors and the limited-memory history, and evaluates the initial
/// objective and gradient. When `orthantwise_c != 0` the OWL-QN pseudo-gradient
/// is constructed and used in place of the gradient throughout. The main loop
/// performs, on each iteration, a line search along the current search
/// direction, an optional user progress callback (which may cancel the run),
/// the gradient-norm and delta-based convergence tests, and the L-BFGS
/// two-loop recursion that produces the next search direction.
///
/// Returns an `i32` status code: `LBFGS_SUCCESS` on convergence, `LBFGS_STOP`
/// on the delta-based stopping test, `LBFGS_ALREADY_MINIMIZED` when the
/// initial point already satisfies the gradient criterion, or a negative
/// `LBFGSERR_*` value on failure. The final objective value is written
/// through `out_fx` and the optimized variables into `x`.
///
/// The user-supplied `evaluate` closure is always invoked on the original
/// `n`-element slices; internally the implementation pads its working buffers
/// to length `n` (the C code rounds up to a multiple of 8 for SSE-aligned
/// storage, retained here for bit-exact compatibility).
pub fn lbfgs_impl_safe(
    x: &mut [f64],
    out_fx: &mut f64,
    mut evaluate: impl FnMut(&[f64], &mut [f64], f64) -> f64,
    mut progress: Option<&mut dyn FnMut(&crate::ProgressReport) -> bool>,
    input_param: &InternalParam,
) -> i32 {
    let n = x.len();
    let n_i32 = n as i32;
    let np = n;

    let mut param = *input_param;
    let m = param.m as usize;

    // Wrap evaluate: user's closure operates on original n elements,
    // but internally we work with zero-padded arrays of size np.
    let mut eval_fn = |xp: &[f64], gp: &mut [f64], step: f64| -> f64 {
        evaluate(&xp[..n], &mut gp[..n], step)
    };

    // --- Parameter validation ---
    if n_i32 <= 0 {
        return LBFGSERR_INVALID_N;
    }
    if param.epsilon < 0.0 {
        return LBFGSERR_INVALID_EPSILON;
    }
    if param.past < 0 {
        return LBFGSERR_INVALID_TESTPERIOD;
    }
    if param.delta < 0.0 {
        return LBFGSERR_INVALID_DELTA;
    }
    if param.min_step < 0.0 {
        return LBFGSERR_INVALID_MINSTEP;
    }
    if param.max_step < param.min_step {
        return LBFGSERR_INVALID_MAXSTEP;
    }
    if param.ftol < 0.0 {
        return LBFGSERR_INVALID_FTOL;
    }
    if param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_WOLFE as i32
        || param.linesearch == LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE as i32
    {
        if param.wolfe <= param.ftol || 1.0 <= param.wolfe {
            return LBFGSERR_INVALID_WOLFE;
        }
    }
    if param.gtol < 0.0 {
        return LBFGSERR_INVALID_GTOL;
    }
    if param.xtol < 0.0 {
        return LBFGSERR_INVALID_XTOL;
    }
    if param.max_linesearch <= 0 {
        return LBFGSERR_INVALID_MAXLINESEARCH;
    }
    if param.orthantwise_c < 0.0 {
        return LBFGSERR_INVALID_ORTHANTWISE;
    }
    if param.orthantwise_start < 0 || n_i32 < param.orthantwise_start {
        return LBFGSERR_INVALID_ORTHANTWISE_START;
    }
    if param.orthantwise_end < 0 {
        param.orthantwise_end = n_i32;
    }
    if n_i32 < param.orthantwise_end {
        return LBFGSERR_INVALID_ORTHANTWISE_END;
    }

    // --- Select line search method ---
    let ls_method;
    if param.orthantwise_c != 0.0 {
        match param.linesearch as u32 {
            LBFGS_LINESEARCH_BACKTRACKING => {
                ls_method = LineSearchMethod::BacktrackingOwlqn;
            }
            _ => return LBFGSERR_INVALID_LINESEARCH,
        }
    } else {
        match param.linesearch as u32 {
            LBFGS_LINESEARCH_MORETHUENTE => {
                ls_method = LineSearchMethod::MoreThuente;
            }
            LBFGS_LINESEARCH_BACKTRACKING_ARMIJO
            | LBFGS_LINESEARCH_BACKTRACKING_WOLFE
            | LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE => {
                ls_method = LineSearchMethod::Backtracking;
            }
            _ => return LBFGSERR_INVALID_LINESEARCH,
        }
    }

    // --- Allocate padded working space (multiples of 8 for SSE2) ---
    let mut xw = vec![0.0f64; np]; // padded copy of x
    xw[..n].copy_from_slice(x);
    let mut xp = vec![0.0f64; np];
    let mut g = vec![0.0f64; np];
    let mut gp = vec![0.0f64; np];
    let mut d = vec![0.0f64; np];
    let mut w = vec![0.0f64; np];
    let mut pg = if param.orthantwise_c != 0.0 {
        vec![0.0f64; np]
    } else {
        Vec::new()
    };

    let mut lm: Vec<IterationData> = (0..m)
        .map(|_| IterationData {
            alpha: 0.0,
            s: vec![0.0f64; np],
            y: vec![0.0f64; np],
            ys: 0.0,
        })
        .collect();

    let mut pf = if param.past > 0 {
        vec![0.0f64; param.past as usize]
    } else {
        Vec::new()
    };

    // --- Initial evaluation ---
    let mut fx = eval_fn(&xw, &mut g, 0.0);

    if param.orthantwise_c != 0.0 {
        let xnorm_l1 = owlqn_x1norm(
            &xw, param.orthantwise_start as usize, param.orthantwise_end as usize,
        );
        fx += xnorm_l1 * param.orthantwise_c;
        owlqn_pseudo_gradient(
            &mut pg, &xw, &g, n, param.orthantwise_c,
            param.orthantwise_start as usize, param.orthantwise_end as usize,
        );
    }

    if !pf.is_empty() {
        pf[0] = fx;
    }

    if param.orthantwise_c == 0.0 {
        vecncpy(&mut d, &g);
    } else {
        vecncpy(&mut d, &pg);
    }

    let mut xnorm = vec2norm(&xw);
    let gnorm = if param.orthantwise_c == 0.0 {
        vec2norm(&g)
    } else {
        vec2norm(&pg)
    };
    if xnorm < 1.0 {
        xnorm = 1.0;
    }
    if gnorm / xnorm <= param.epsilon {
        x.copy_from_slice(&xw[..n]);
        *out_fx = fx;
        return LBFGS_ALREADY_MINIMIZED;
    }

    let mut step = vec2norminv(&d);
    let mut k = 1usize;
    let mut end = 0usize;
    let ret;

    // --- Main loop ---
    loop {
        veccpy(&mut xp, &xw);
        veccpy(&mut gp, &g);

        let ls = match ls_method {
            LineSearchMethod::MoreThuente => line_search_morethuente(
                np, &mut xw, &mut fx, &mut g, &mut d, &mut step, &xp, &gp, &mut w,
                &mut eval_fn, &param,
            ),
            LineSearchMethod::Backtracking => line_search_backtracking(
                np, &mut xw, &mut fx, &mut g, &mut d, &mut step, &xp, &gp, &mut w,
                &mut eval_fn, &param,
            ),
            LineSearchMethod::BacktrackingOwlqn => {
                let ls = line_search_backtracking_owlqn(
                    np, &mut xw, &mut fx, &mut g, &mut d, &mut step, &xp, &pg, &mut w,
                    &mut eval_fn, &param,
                );
                owlqn_pseudo_gradient(
                    &mut pg, &xw, &g, n, param.orthantwise_c,
                    param.orthantwise_start as usize, param.orthantwise_end as usize,
                );
                ls
            }
        };

        if ls < 0 {
            veccpy(&mut xw, &xp);
            veccpy(&mut g, &gp);
            ret = ls;
            break;
        }

        xnorm = vec2norm(&xw);
        let gnorm = if param.orthantwise_c == 0.0 {
            vec2norm(&g)
        } else {
            vec2norm(&pg)
        };

        // Progress callback (user sees original n-length slices).
        if let Some(ref mut progress_fn) = progress {
            let report = crate::ProgressReport {
                x: &xw[..n],
                g: &g[..n],
                fx, xnorm, gnorm, step,
                k: k as i32,
                ls,
            };
            if !progress_fn(&report) {
                ret = k as i32;
                break;
            }
        }

        if xnorm < 1.0 {
            xnorm = 1.0;
        }
        if gnorm / xnorm <= param.epsilon {
            ret = LBFGS_SUCCESS;
            break;
        }

        if !pf.is_empty() {
            if param.past as usize <= k {
                let rate = (pf[k % param.past as usize] - fx) / fx;
                if rate.abs() < param.delta {
                    ret = LBFGS_STOP;
                    break;
                }
            }
            pf[k % param.past as usize] = fx;
        }

        if param.max_iterations != 0 && param.max_iterations < (k + 1) as i32 {
            ret = LBFGSERR_MAXIMUMITERATION;
            break;
        }

        // Update s and y vectors.
        let it = &mut lm[end];
        vecdiff(&mut it.s, &xw, &xp);
        vecdiff(&mut it.y, &g, &gp);

        let ys = vecdot(&it.y, &it.s);
        let yy = vecdot(&it.y, &it.y);
        it.ys = ys;

        let bound = if m <= k { m } else { k };
        k += 1;
        end = (end + 1) % m;

        // Two-loop recursion.
        if param.orthantwise_c == 0.0 {
            vecncpy(&mut d, &g);
        } else {
            vecncpy(&mut d, &pg);
        }

        // First loop (backward).
        let mut j = end;
        for _i in 0..bound {
            j = (j + m - 1) % m;
            let it = &mut lm[j];
            it.alpha = vecdot(&it.s, &d);
            it.alpha /= it.ys;
            let alpha = it.alpha;
            unsafe {
                let dp = d.as_mut_ptr();
                let yp = it.y.as_ptr();
                for idx in 0..np {
                    *dp.add(idx) += -alpha * *yp.add(idx);
                }
            }
        }

        vecscale(&mut d, ys / yy);

        // Second loop (forward).
        for _i in 0..bound {
            let it = &lm[j];
            let beta = vecdot(&it.y, &d) / it.ys;
            let alpha = it.alpha;
            unsafe {
                let dp = d.as_mut_ptr();
                let sp = it.s.as_ptr();
                for idx in 0..np {
                    *dp.add(idx) += (alpha - beta) * *sp.add(idx);
                }
            }
            j = (j + 1) % m;
        }

        // OWL-QN direction constraint.
        if param.orthantwise_c != 0.0 {
            for i in param.orthantwise_start as usize..param.orthantwise_end as usize {
                if d[i] * pg[i] >= 0.0 {
                    d[i] = 0.0;
                }
            }
        }

        step = 1.0;
    }

    // Copy result back to user's array.
    x.copy_from_slice(&xw[..n]);
    *out_fx = fx;
    ret
}
