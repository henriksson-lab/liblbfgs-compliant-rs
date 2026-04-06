/// Vector operations — direct translation of arithmetic_ansi.h / arithmetic_sse_double.h

#[inline]
#[allow(dead_code)]
pub fn vecset(x: &mut [f64], c: f64) {
    for xi in x.iter_mut() {
        *xi = c;
    }
}

#[inline]
pub fn veccpy(y: &mut [f64], x: &[f64]) {
    y[..x.len()].copy_from_slice(x);
}

#[inline]
pub fn vecncpy(y: &mut [f64], x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = -*xi;
    }
}

#[inline]
pub fn vecadd(y: &mut [f64], x: &[f64], c: f64) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += c * *xi;
    }
}

#[inline]
pub fn vecdiff(z: &mut [f64], x: &[f64], y: &[f64]) {
    for i in 0..z.len() {
        z[i] = x[i] - y[i];
    }
}

#[inline]
pub fn vecscale(y: &mut [f64], c: f64) {
    for yi in y.iter_mut() {
        *yi *= c;
    }
}

#[inline]
pub fn vecdot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        s += *xi * *yi;
    }
    s
}

#[inline]
pub fn vec2norm(x: &[f64]) -> f64 {
    vecdot(x, x).sqrt()
}

#[inline]
pub fn vec2norminv(x: &[f64]) -> f64 {
    1.0 / vec2norm(x)
}

/// Sign-bit comparison matching SSE `_mm_movemask_pd` behavior.
/// Returns true when the IEEE 754 sign bits of x and y differ.
#[inline]
pub fn fsigndiff(x: f64, y: f64) -> bool {
    x.is_sign_negative() != y.is_sign_negative()
}
