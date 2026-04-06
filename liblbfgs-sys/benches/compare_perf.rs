/// Performance comparison between pure Rust and C implementations.
/// Run with: cargo bench --features compare-c

#[cfg(not(feature = "compare-c"))]
fn main() {
    eprintln!("This benchmark requires the compare-c feature.");
    eprintln!("Run: cargo bench --features compare-c");
    std::process::exit(1);
}

#[cfg(feature = "compare-c")]
fn main() {
    use liblbfgs_compliant_rs::c_ffi;
    use liblbfgs_compliant_rs::*;
    use std::os::raw::{c_int, c_void};
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // Objective functions
    // -----------------------------------------------------------------------

    fn rosenbrock_nd(x: &[f64], _step: f64) -> (f64, Vec<f64>) {
        let n = x.len();
        let mut g = vec![0.0; n];
        let mut f = 0.0;
        for i in (0..n - 1).step_by(2) {
            let t1 = 1.0 - x[i];
            let t2 = x[i + 1] - x[i] * x[i];
            f += t1 * t1 + 100.0 * t2 * t2;
            g[i] += -2.0 * t1 + 200.0 * t2 * (-2.0 * x[i]);
            g[i + 1] += 200.0 * t2;
        }
        (f, g)
    }

    unsafe extern "C" fn c_rosenbrock_nd(
        _inst: *mut c_void,
        x: *const f64,
        g: *mut f64,
        n: c_int,
        _step: f64,
    ) -> f64 {
        let n = n as usize;
        let xs = std::slice::from_raw_parts(x, n);
        let gs = std::slice::from_raw_parts_mut(g, n);
        for i in 0..n {
            gs[i] = 0.0;
        }
        let mut f = 0.0;
        for i in (0..n - 1).step_by(2) {
            let t1 = 1.0 - xs[i];
            let t2 = xs[i + 1] - xs[i] * xs[i];
            f += t1 * t1 + 100.0 * t2 * t2;
            gs[i] += -2.0 * t1 + 200.0 * t2 * (-2.0 * xs[i]);
            gs[i + 1] += 200.0 * t2;
        }
        f
    }

    // -----------------------------------------------------------------------
    // Benchmark helpers
    // -----------------------------------------------------------------------

    fn make_initial(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect()
    }

    fn bench_rust(n: usize, iters: u32) -> (std::time::Duration, f64) {
        let param = LbfgsParam::default();
        let mut total = std::time::Duration::ZERO;
        let mut fx = 0.0;
        for _ in 0..iters {
            let mut x = make_initial(n);
            let start = Instant::now();
            let r = lbfgs(&mut x, rosenbrock_nd, None, &param).unwrap();
            total += start.elapsed();
            fx = r.fx;
        }
        (total / iters, fx)
    }

    fn bench_c(n: usize, iters: u32) -> (std::time::Duration, f64) {
        let mut total = std::time::Duration::ZERO;
        let mut fx_out = 0.0;
        for _ in 0..iters {
            unsafe {
                let x = c_ffi::lbfgs_malloc(n as c_int);
                let init = make_initial(n);
                for i in 0..n {
                    *x.add(i) = init[i];
                }
                let mut fx = 0.0f64;
                let start = Instant::now();
                c_ffi::lbfgs(
                    n as c_int,
                    x,
                    &mut fx,
                    Some(c_rosenbrock_nd),
                    None,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                );
                total += start.elapsed();
                fx_out = fx;
                c_ffi::lbfgs_free(x);
            }
        }
        (total / iters, fx_out)
    }

    // -----------------------------------------------------------------------
    // Run benchmarks
    // -----------------------------------------------------------------------

    println!("Performance comparison: Pure Rust vs C (libLBFGS)");
    println!("=================================================");
    println!("{:>8} {:>12} {:>12} {:>10}", "N", "Rust", "C", "Ratio");
    println!("{:>8} {:>12} {:>12} {:>10}", "---", "---", "---", "---");

    for &n in &[10, 100, 1000, 10000] {
        // Warmup
        let _ = bench_rust(n, 1);
        let _ = bench_c(n, 1);

        let iters = match n {
            10 => 10000,
            100 => 1000,
            1000 => 100,
            10000 => 10,
            _ => 100,
        };

        let (rust_time, rust_fx) = bench_rust(n, iters);
        let (c_time, c_fx) = bench_c(n, iters);

        let ratio = rust_time.as_nanos() as f64 / c_time.as_nanos() as f64;

        println!(
            "{:>8} {:>10.1}us {:>10.1}us {:>9.2}x",
            n,
            rust_time.as_nanos() as f64 / 1000.0,
            c_time.as_nanos() as f64 / 1000.0,
            ratio,
        );

        // Verify same results
        assert_eq!(
            rust_fx.to_bits(),
            c_fx.to_bits(),
            "Results differ for n={}!",
            n
        );
    }

    println!("\nRatio < 1.0 means Rust is faster, > 1.0 means C is faster.");
    println!("All results verified bit-exact between Rust and C.");
}
