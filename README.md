# liblbfgs-compliant-rs

Faithful pure-Rust implementation of the [libLBFGS](https://github.com/chokkan/liblbfgs) optimization library. No C compiler or external dependencies required.

The implementation produces **bit-exact** results matching the original C library, verified by 32 conformance tests with `f64::to_bits()` assertions.

The API is based on the code from https://github.com/messense/liblbfgs-sys, so this crate should be a drop-in replacement

## Precision

Note that there is a feature "simd". Enabling it makes this crate produce different results from the original liblbfgs - possibly even more precise results.
However, this depends on compiler settings, and compiler, used for liblbfgs. So SIMD is enabled by default

With SIMD, this crate is about 2x as fast as the original code

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
liblbfgs-compliant-rs = "0.1"
```

### Minimize the Rosenbrock function

```rust
use liblbfgs_compliant_rs::*;

fn main() {
    let mut x = vec![-1.2, 1.0];

    let result = lbfgs(
        &mut x,
        |x, g, _step| {
            // Objective: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
            let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
            // Gradient
            g[0] = -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0] * x[0]) * (-2.0 * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
            f
        },
        None,
        &LbfgsParam::default(),
    ).unwrap();

    println!("Minimum at x = [{}, {}], f(x) = {}", x[0], x[1], result.fx);
    // Output: Minimum at x = [1.000000595052523, 1.0000011922280314], f(x) ≈ 0
}
```

### With progress reporting

```rust
use liblbfgs_compliant_rs::*;

let mut x = vec![-1.2, 1.0];

let result = lbfgs(
    &mut x,
    |x, g, _step| {
        let f = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        g[0] = -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0] * x[0]) * (-2.0 * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
        f
    },
    Some(&mut |report| {
        println!("iter {}: fx = {:.6e}", report.k, report.fx);
        true // return false to cancel
    }),
    &LbfgsParam::default(),
).unwrap();
```

### Custom parameters

```rust
use liblbfgs_compliant_rs::*;

let param = LbfgsParam {
    m: 10,                                      // more history corrections
    linesearch: LineSearch::BacktrackingWolfe,   // Wolfe line search
    max_iterations: 100,
    ..Default::default()
};
```

### L1-regularized optimization (OWL-QN)

```rust
use liblbfgs_compliant_rs::*;

let param = LbfgsParam {
    linesearch: LineSearch::BacktrackingWolfe,  // required for OWL-QN
    orthantwise: Some(OrthantWise {
        c: 1.0,     // L1 regularization coefficient
        start: 0,   // start index for L1 norm
        end: -1,    // -1 means all variables
    }),
    ..Default::default()
};
```

## API

### `lbfgs(x, evaluate, progress, param) -> Result<LbfgsResult, LbfgsError>`

- **`x: &mut [f64]`** — initial values, modified in-place to the solution
- **`evaluate: FnMut(&[f64], &mut [f64], f64) -> f64`** — compute objective value and write gradient into the second argument. The third argument is the current line search step size. Returns the objective function value.
- **`progress: Option<&mut dyn FnMut(&ProgressReport) -> bool>`** — optional callback called each iteration. Return `false` to cancel.
- **`param: &LbfgsParam`** — optimization parameters

Returns `Ok(LbfgsResult)` with convergence type and final `fx`, or `Err(LbfgsError)` on failure.

### Line search algorithms

| Variant | Description |
|---------|-------------|
| `LineSearch::MoreThuente` | More-Thuente method with cubic/quadratic interpolation (default) |
| `LineSearch::BacktrackingArmijo` | Backtracking with sufficient decrease only |
| `LineSearch::BacktrackingWolfe` | Backtracking with Wolfe curvature condition |
| `LineSearch::BacktrackingStrongWolfe` | Backtracking with strong Wolfe condition |

### Convergence types

| Variant | Description |
|---------|-------------|
| `Convergence::Gradient` | `\|\|g\|\| / max(1, \|\|x\|\|) <= epsilon` |
| `Convergence::Delta` | Relative improvement < delta over `past` iterations |
| `Convergence::AlreadyMinimized` | Initial point is already a minimizer |

## Performance

Compared against the original C library with SSE2 SIMD on an Intel Xeon Gold (with `-C target-cpu=native`):

| N | Rust | C (SSE2) | Ratio |
|---:|---:|---:|---:|
| 10 | 19 us | 18 us | 1.01x |
| 100 | 157 us | 141 us | 1.12x |
| 1,000 | 1,746 us | 1,548 us | 1.13x |
| 10,000 | 17,014 us | 15,976 us | 1.06x |

Within ~5-13% of hand-optimized C+SSE2, with all results verified bit-exact.

## Verifying against the C library

Enable the `compare-c` feature to build the original C library and run comparison tests:

```bash
git submodule update --init --recursive
cargo test --features compare-c
cargo bench --features compare-c
```

This requires CMake and a C compiler. The comparison tests call both implementations and assert bit-identical results.

## References

- [libLBFGS](https://github.com/chokkan/liblbfgs) by Naoaki Okazaki — the original C implementation
- Jorge Nocedal. *Updating Quasi-Newton Matrices with Limited Storage.* Mathematics of Computation, 1980.
- Jorge J. More and David J. Thuente. *Line search algorithm with guaranteed sufficient decrease.* ACM TOMS, 1994.
- Galen Andrew and Jianfeng Gao. *Scalable training of L1-regularized log-linear models.* ICML, 2007.

## License

MIT
