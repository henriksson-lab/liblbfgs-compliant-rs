/// OWL-QN (Orthant-Wise Limited-memory Quasi-Newton) helpers.

/// Compute L1 norm of x[start..end].
pub fn owlqn_x1norm(x: &[f64], start: usize, end: usize) -> f64 {
    let mut norm = 0.0;
    for i in start..end {
        norm += x[i].abs();
    }
    norm
}

/// Compute pseudo-gradient for OWL-QN.
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

/// Project direction onto orthant: zero out components where d[i]*sign[i] <= 0.
pub fn owlqn_project(d: &mut [f64], sign: &[f64], start: usize, end: usize) {
    for i in start..end {
        if d[i] * sign[i] <= 0.0 {
            d[i] = 0.0;
        }
    }
}
