/// Main L-BFGS optimization implementation — direct translation of lbfgs() from lbfgs.c.

use crate::arithmetic::*;
use crate::linesearch::*;
use crate::owlqn::*;
use crate::*;

struct IterationData {
    alpha: f64,
    s: Vec<f64>,
    y: Vec<f64>,
    ys: f64,
}

/// Internal parameter struct matching C layout semantics but using plain Rust types.
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

/// Type for line search function dispatch.
type LineSearchFn = fn(
    usize,
    &mut [f64],
    &mut f64,
    &mut [f64],
    &mut [f64],
    &mut f64,
    &[f64],
    &[f64],
    &mut [f64],
    &mut dyn FnMut(&[f64], &mut [f64], f64) -> f64,
    &InternalParam,
) -> i32;

/// Main L-BFGS optimization — safe Rust API with closures.
pub fn lbfgs_impl_safe(
    x: &mut [f64],
    out_fx: &mut f64,
    mut evaluate: impl FnMut(&[f64], f64) -> (f64, Vec<f64>),
    mut progress: Option<&mut dyn FnMut(&crate::ProgressReport) -> bool>,
    input_param: &InternalParam,
) -> i32 {
    let n = x.len();
    let n_i32 = n as i32;

    // Copy parameters.
    let mut param = *input_param;
    let m = param.m as usize;

    // Wrap the user's evaluate closure into the form line search expects:
    // fn(x, g, step) -> fx  (writes gradient into g)
    let mut eval_fn = |x_slice: &[f64], g_slice: &mut [f64], step: f64| -> f64 {
        let (fx, grad) = evaluate(x_slice, step);
        g_slice[..grad.len()].copy_from_slice(&grad);
        fx
    };

    // Check input parameters for errors.
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

    // Select line search method.
    let linesearch: LineSearchFn;
    if param.orthantwise_c != 0.0 {
        match param.linesearch as u32 {
            LBFGS_LINESEARCH_BACKTRACKING => {
                linesearch = line_search_backtracking_owlqn;
            }
            _ => {
                return LBFGSERR_INVALID_LINESEARCH;
            }
        }
    } else {
        match param.linesearch as u32 {
            LBFGS_LINESEARCH_MORETHUENTE => {
                linesearch = line_search_morethuente;
            }
            LBFGS_LINESEARCH_BACKTRACKING_ARMIJO
            | LBFGS_LINESEARCH_BACKTRACKING_WOLFE
            | LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE => {
                linesearch = line_search_backtracking;
            }
            _ => {
                return LBFGSERR_INVALID_LINESEARCH;
            }
        }
    }

    // Allocate working space.
    let mut xp = vec![0.0f64; n];
    let mut g = vec![0.0f64; n];
    let mut gp = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    let mut pg = if param.orthantwise_c != 0.0 {
        vec![0.0f64; n]
    } else {
        Vec::new()
    };

    // Initialize limited memory storage.
    let mut lm: Vec<IterationData> = (0..m)
        .map(|_| IterationData {
            alpha: 0.0,
            s: vec![0.0f64; n],
            y: vec![0.0f64; n],
            ys: 0.0,
        })
        .collect();

    // Allocate past function values storage.
    let mut pf = if param.past > 0 {
        vec![0.0f64; param.past as usize]
    } else {
        Vec::new()
    };

    // Evaluate the function value and its gradient.
    let mut fx = eval_fn(x, &mut g, 0.0);

    if param.orthantwise_c != 0.0 {
        let xnorm_l1 = owlqn_x1norm(
            x,
            param.orthantwise_start as usize,
            param.orthantwise_end as usize,
        );
        fx += xnorm_l1 * param.orthantwise_c;
        owlqn_pseudo_gradient(
            &mut pg, x, &g, n,
            param.orthantwise_c,
            param.orthantwise_start as usize,
            param.orthantwise_end as usize,
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

    let mut xnorm = vec2norm(x);
    let gnorm = if param.orthantwise_c == 0.0 {
        vec2norm(&g)
    } else {
        vec2norm(&pg)
    };
    if xnorm < 1.0 {
        xnorm = 1.0;
    }
    if gnorm / xnorm <= param.epsilon {
        *out_fx = fx;
        return LBFGS_ALREADY_MINIMIZED;
    }

    let mut step = vec2norminv(&d);

    let mut k = 1usize;
    let mut end = 0usize;
    let ret;

    loop {
        veccpy(&mut xp, x);
        veccpy(&mut gp, &g);

        let ls = if param.orthantwise_c == 0.0 {
            linesearch(
                n, x, &mut fx, &mut g, &mut d, &mut step, &xp, &gp, &mut w,
                &mut eval_fn, &param,
            )
        } else {
            let ls = linesearch(
                n, x, &mut fx, &mut g, &mut d, &mut step, &xp, &pg, &mut w,
                &mut eval_fn, &param,
            );
            owlqn_pseudo_gradient(
                &mut pg, x, &g, n,
                param.orthantwise_c,
                param.orthantwise_start as usize,
                param.orthantwise_end as usize,
            );
            ls
        };

        if ls < 0 {
            veccpy(x, &xp);
            veccpy(&mut g, &gp);
            ret = ls;
            break;
        }

        xnorm = vec2norm(x);
        let gnorm = if param.orthantwise_c == 0.0 {
            vec2norm(&g)
        } else {
            vec2norm(&pg)
        };

        // Report progress.
        if let Some(ref mut progress_fn) = progress {
            let report = crate::ProgressReport {
                x,
                g: &g,
                fx,
                xnorm,
                gnorm,
                step,
                k: k as i32,
                ls,
            };
            if !progress_fn(&report) {
                ret = k as i32; // non-zero, non-error (matches C behavior)
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

        let it = &mut lm[end];
        vecdiff(&mut it.s, x, &xp);
        vecdiff(&mut it.y, &g, &gp);

        let ys = vecdot(&it.y, &it.s);
        let yy = vecdot(&it.y, &it.y);
        it.ys = ys;

        let bound = if m <= k { m } else { k };
        k += 1;
        end = (end + 1) % m;

        if param.orthantwise_c == 0.0 {
            vecncpy(&mut d, &g);
        } else {
            vecncpy(&mut d, &pg);
        }

        let mut j = end;
        for _i in 0..bound {
            j = (j + m - 1) % m;
            let it = &mut lm[j];
            it.alpha = vecdot(&it.s, &d);
            it.alpha /= it.ys;
            let alpha = it.alpha;
            for idx in 0..n {
                d[idx] += -alpha * it.y[idx];
            }
        }

        vecscale(&mut d, ys / yy);

        for _i in 0..bound {
            let it = &lm[j];
            let beta = vecdot(&it.y, &d) / it.ys;
            let alpha = it.alpha;
            for idx in 0..n {
                d[idx] += (alpha - beta) * it.s[idx];
            }
            j = (j + 1) % m;
        }

        if param.orthantwise_c != 0.0 {
            for i in param.orthantwise_start as usize..param.orthantwise_end as usize {
                if d[i] * pg[i] >= 0.0 {
                    d[i] = 0.0;
                }
            }
        }

        step = 1.0;
    }

    *out_fx = fx;
    ret
}
