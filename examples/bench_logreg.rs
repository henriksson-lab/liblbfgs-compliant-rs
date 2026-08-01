use liblbfgs_compliant_rs::{lbfgs, LbfgsParam};
use std::env;
use std::time::Instant;

struct Data {
    samples: usize,
    features: usize,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn feature_value(i: usize, j: usize) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z ^= (j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) - 0.5
}

fn make_data(samples: usize, features: usize) -> Data {
    let mut x = Vec::with_capacity(samples * features);
    let mut y = Vec::with_capacity(samples);
    for i in 0..samples {
        let mut score = 0.0;
        for j in 0..features {
            let v = feature_value(i, j);
            if j < 16 {
                let w = ((j % 7) as f64 - 3.0) * 0.15;
                score += v * w;
            }
            x.push(v);
        }
        y.push(if score >= 0.0 { 1.0 } else { -1.0 });
    }
    Data {
        samples,
        features,
        x,
        y,
    }
}

fn eval_logreg(data: &Data, lambda: f64, w: &[f64], g: &mut [f64]) -> f64 {
    g.fill(0.0);
    let mut loss = 0.0;

    for i in 0..data.samples {
        let row = &data.x[i * data.features..(i + 1) * data.features];
        let mut dot = 0.0;
        for j in 0..data.features {
            dot += row[j] * w[j];
        }

        let yz = data.y[i] * dot;
        let coeff = if yz >= 0.0 {
            let e = (-yz).exp();
            loss += e.ln_1p();
            -data.y[i] * e / (1.0 + e)
        } else {
            let e = yz.exp();
            loss += -yz + e.ln_1p();
            -data.y[i] / (1.0 + e)
        };

        for j in 0..data.features {
            g[j] += coeff * row[j];
        }
    }

    for j in 0..data.features {
        loss += 0.5 * lambda * w[j] * w[j];
        g[j] += lambda * w[j];
    }
    loss
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let samples = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let features = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let max_iterations = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(25);
    let lambda = 1e-4;

    let data = make_data(samples, features);
    let mut w = vec![0.0; features];
    for (j, v) in w.iter_mut().enumerate() {
        *v = ((j % 11) as f64 - 5.0) * 0.01;
    }

    let param = LbfgsParam {
        max_iterations,
        epsilon: 1e-12,
        ..Default::default()
    };

    let mut evaluations = 0usize;
    let mut iterations = 0i32;
    let mut last_fx = f64::NAN;
    let started = Instant::now();
    let result = lbfgs(
        &mut w,
        |x, g, _step| {
            evaluations += 1;
            last_fx = eval_logreg(&data, lambda, x, g);
            last_fx
        },
        Some(&mut |report| {
            iterations = report.k;
            true
        }),
        &param,
    );
    let elapsed = started.elapsed();

    let checksum: f64 = w
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64 + 1.0) * v)
        .sum();
    match result {
        Ok(r) => {
            println!(
                "impl=rust samples={} features={} max_iterations={} status=ok convergence={:?} iterations={} evaluations={} fx={:.12e} checksum={:.12e} elapsed_sec={:.6}",
                samples, features, max_iterations, r.convergence, iterations, evaluations, r.fx, checksum, elapsed.as_secs_f64()
            );
        }
        Err(e) => {
            println!(
                "impl=rust samples={} features={} max_iterations={} status=err error={} iterations={} evaluations={} fx={:.12e} checksum={:.12e} elapsed_sec={:.6}",
                samples, features, max_iterations, e, iterations, evaluations, last_fx, checksum, elapsed.as_secs_f64()
            );
        }
    }
}
