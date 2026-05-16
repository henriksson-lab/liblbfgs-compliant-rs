//! OWL-QN (Orthant-Wise Limited-memory Quasi-Newton) helpers.
//!
//! Support routines for minimizing `F(x) + C * |x|_1`, where the L1 term is
//! summed over a configurable index range `[start, end)` so that selected
//! variables (e.g. a bias term) can be excluded from regularization.

/// Compute the L1 norm of `x[start..end]`.
///
/// Returns the contribution of the regularized variables to the OWL-QN
/// objective, i.e. `sum_{i in [start, end)} |x[i]|`.
pub fn owlqn_x1norm(x: &[f64], start: usize, end: usize) -> f64 {
    let mut norm = 0.0;
    for i in start..end {
        norm += x[i].abs();
    }
    norm
}

/// Compute the OWL-QN pseudo-gradient `pg` from the raw gradient `g`.
///
/// Outside `[start, end)` the pseudo-gradient equals `g[i]`. Inside that range,
/// for `x[i] != 0` it is the differentiable value `g[i] + c * sign(x[i])`; at
/// `x[i] == 0` it picks the one-sided partial derivative that most decreases
/// the objective (`g[i] + c` if `g[i] < -c`, `g[i] - c` if `g[i] > c`, else
/// `0`). See Andrew & Gao, ICML 2007.
pub fn owlqn_pseudo_gradient(
    pg: &mut [f64],
    x: &[f64],
    g: &[f64],
    n: usize,
    c: f64,
    start: usize,
    end: usize,
) {
    for i in 0..start {
        pg[i] = g[i];
    }
    for i in start..end {
        if x[i] < 0.0 {
            pg[i] = g[i] - c;
        } else if 0.0 < x[i] {
            pg[i] = g[i] + c;
        } else {
            if g[i] < -c {
                pg[i] = g[i] + c;
            } else if c < g[i] {
                pg[i] = g[i] - c;
            } else {
                pg[i] = 0.0;
            }
        }
    }
    for i in end..n {
        pg[i] = g[i];
    }
}

/// Project the search direction `d` onto the orthant defined by `sign`.
///
/// For each `i` in `[start, end)`, zero `d[i]` whenever its sign disagrees with
/// `sign[i]` (i.e. `d[i] * sign[i] <= 0`), keeping the OWL-QN step inside the
/// current orthant.
pub fn owlqn_project(d: &mut [f64], sign: &[f64], start: usize, end: usize) {
    for i in start..end {
        if d[i] * sign[i] <= 0.0 {
            d[i] = 0.0;
        }
    }
}
