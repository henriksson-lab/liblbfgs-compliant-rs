/// Performance and accuracy comparison between pure Rust (SIMD) and C implementations.
/// Run with: cargo bench --features compare-c

#[cfg(not(feature = "compare-c"))]
fn main() {
    eprintln!("Run: cargo bench --features compare-c");
    std::process::exit(1);
}

#[cfg(feature = "compare-c")]
fn main() {
    use liblbfgs_compliant_rs::c_ffi;
    use liblbfgs_compliant_rs::*;
    use std::os::raw::{c_int, c_void};
    use std::time::Instant;

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
        rosenbrock_nd(xs, gs, 0.0)
    }

    fn make_initial(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect()
    }

    fn run_rust(n: usize) -> (f64, Vec<f64>) {
        let mut x = make_initial(n);
        let r = lbfgs(&mut x, rosenbrock_nd, None, &LbfgsParam::default()).unwrap();
        (r.fx, x)
    }

    fn run_c(n: usize) -> (f64, Vec<f64>) {
        unsafe {
            let x = c_ffi::lbfgs_malloc(n as c_int);
            let init = make_initial(n);
            for i in 0..n {
                *x.add(i) = init[i];
            }
            let mut fx = 0.0f64;
            c_ffi::lbfgs(
                n as c_int,
                x,
                &mut fx,
                Some(c_rosenbrock_nd),
                None,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let mut result = vec![0.0; n];
            for i in 0..n {
                result[i] = *x.add(i);
            }
            c_ffi::lbfgs_free(x);
            (fx, result)
        }
    }

    fn bench_rust(n: usize, iters: u32) -> std::time::Duration {
        let param = LbfgsParam::default();
        let mut total = std::time::Duration::ZERO;
        for _ in 0..iters {
            let mut x = make_initial(n);
            let start = Instant::now();
            let _ = lbfgs(&mut x, rosenbrock_nd, None, &param);
            total += start.elapsed();
        }
        total / iters
    }

    fn bench_c(n: usize, iters: u32) -> std::time::Duration {
        let mut total = std::time::Duration::ZERO;
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
                c_ffi::lbfgs_free(x);
            }
        }
        total / iters
    }

    // --- Performance ---
    println!("Performance comparison: Rust (AVX2+FMA) vs C (scalar)");
    println!("======================================================");
    println!("{:>8} {:>12} {:>12} {:>10}", "N", "Rust", "C", "Ratio");
    println!("{:>8} {:>12} {:>12} {:>10}", "---", "---", "---", "---");

    for &n in &[10, 100, 1000, 10000] {
        let _ = bench_rust(n, 2);
        let _ = bench_c(n, 2);
        let iters = match n {
            10 => 10000,
            100 => 2000,
            1000 => 200,
            10000 => 20,
            _ => 100,
        };
        let rust_time = bench_rust(n, iters);
        let c_time = bench_c(n, iters);
        let ratio = rust_time.as_nanos() as f64 / c_time.as_nanos() as f64;
        println!(
            "{:>8} {:>10.1}us {:>10.1}us {:>9.2}x",
            n,
            rust_time.as_nanos() as f64 / 1000.0,
            c_time.as_nanos() as f64 / 1000.0,
            ratio
        );
    }

    // --- Accuracy ---
    println!("\nNumerical accuracy: Rust (AVX2+FMA) vs C (scalar)");
    println!("==================================================");
    println!(
        "{:>8} {:>16} {:>16} {:>14} {:>14}",
        "N", "Rust fx", "C fx", "abs error", "rel error"
    );
    println!(
        "{:>8} {:>16} {:>16} {:>14} {:>14}",
        "---", "---", "---", "---", "---"
    );

    for &n in &[10, 100, 1000, 10000] {
        let (rfx, rx) = run_rust(n);
        let (cfx, cx) = run_c(n);

        let abs_err_fx = (rfx - cfx).abs();
        let rel_err_fx = if cfx != 0.0 {
            abs_err_fx / cfx.abs()
        } else {
            abs_err_fx
        };

        // Max error across x values
        let mut max_abs_x = 0.0f64;
        let mut max_rel_x = 0.0f64;
        for i in 0..n {
            let ae = (rx[i] - cx[i]).abs();
            let re = if cx[i] != 0.0 { ae / cx[i].abs() } else { ae };
            max_abs_x = max_abs_x.max(ae);
            max_rel_x = max_rel_x.max(re);
        }

        println!(
            "{:>8} {:>16.6e} {:>16.6e} {:>14.2e} {:>14.2e}",
            n, rfx, cfx, abs_err_fx, rel_err_fx
        );
        println!(
            "{:>8} {:>33} {:>14.2e} {:>14.2e}",
            "", "max |x_rust - x_c|:", max_abs_x, max_rel_x
        );
    }

    println!("\nRatio < 1.0 means Rust is faster.");
}
