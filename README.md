# liblbfgs-compliant-rs

Faithful pure-Rust implementation of the [libLBFGS](https://github.com/chokkan/liblbfgs) optimization library. No C compiler or external dependencies required.

The API is based on the code from https://github.com/messense/liblbfgs-sys, so this crate should be a drop-in replacement

* 2026-08-01: CI added
* 2026-05-23: Renewed audit, no serious issues found but SIMD disabled by default as better for reproducibility. Benchmark made more faithful
* 2026-05-16: Appears to be a faithful translation

## This is an LLM-mediated faithful (hopefully) translation, not the original code! 

Most users should probably first see if the existing original code works for them, unless they have reason otherwise. The original source
may have newer features and it has had more love in terms of fixing bugs. In fact, we aim to replicate bugs if they are present, for the
sake of reproducibility! (but then we might have added a few more in the process)

There are however cases when you might prefer this Rust version. We generally agree with [this manifesto](https://rewrites.bio/) but more specifically:
* We have had many issues with ensuring that our software works using existing containers (Docker, PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant Javascript. This is a language where even the == operator is broken. Typescript is one step up, but a game changer is the ability to compile Rust code into webassembly, enabling performance and sharing of code with the backend. Translating code to Rust enables new ways of deployment and running code in the browser has especial benefits for science - researchers do not have deep pockets to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are spent creating and communicating via small files, which we can bypass by using code as libraries. Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows that [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820). This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error handling, as opposed to typical command-line scripting

But:

* **This approach should still be considered experimental**. The LLM technology is immature and has sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle. This translation is as much aimed to learn how to improve the technology and get feedback on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to help evaluate the translation. If you want improved performance, you generally have to use this code as a library, and use the additional tricks it offers. We generally accept performance losses in order to reduce our dependency issues
* **Check the original Github pages for information about the package**. This README is kept sparse on purpose. It is not meant to be the primary source of information
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible translation of a snapshot of your code. If we find serious bugs, we will report them to you. Otherwise we will just replicate them, to ensure comparability across studies that claim to use package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we treat it

This blurb might be out of date. Go to [this page](https://github.com/henriksson-lab/rustification) for the latest information and further information about how we approach translation

## Precision

Note that there is a feature "simd". Enabling it makes this crate produce different results from the original liblbfgs - possibly even more precise results.
However, this depends on compiler settings, and compiler, used for liblbfgs. SIMD is therefore opt-in and disabled by default.

With SIMD enabled, this crate may be faster than the original C code for larger problems, but exact
performance depends on CPU, compiler, and whether the C build uses SIMD. See the benchmark section
below for one measured run.

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

Original benchmark baseline: the vendored original C library is libLBFGS
`1.10.0` (`1.10`).

Latest captured run: 2026-07-14, Rust commit `3920187`. Command:
`cargo bench --features compare-c --bench compare_perf -- --nocapture`.
The paired harness compares Rust and the vendored C implementation on
deterministic Rosenbrock workloads. All reported objective values and final
vectors matched exactly to the harness output. Aggregate speedup is **0.92x**
(C time / Rust time; higher is better), so Rust was slightly slower in this
run. The harness reports only process-level RSS (`28,160 KiB` in this capture),
not separate Rust/C RSS values. Raw rows are tracked in
`benchmarks/liblbfgs-compliant-rs.tsv` in the presentation repository.

| N | Rust us | C us | Rust/C ratio | C/Rust speedup | fx abs error | max x abs error |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 15.8 | 15.0 | 1.05x | 0.95x | 0.00e0 | 0.00e0 |
| 100 | 122.1 | 108.2 | 1.13x | 0.89x | 0.00e0 | 0.00e0 |
| 1000 | 1331.5 | 1232.1 | 1.08x | 0.93x | 0.00e0 | 0.00e0 |
| 10000 | 13444.9 | 12488.9 | 1.08x | 0.93x | 0.00e0 | 0.00e0 |

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
