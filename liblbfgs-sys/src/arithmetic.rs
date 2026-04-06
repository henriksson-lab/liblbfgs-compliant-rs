/// Vector operations — using unsafe indexing to eliminate bounds checks.

#[inline(always)]
#[allow(dead_code)]
pub fn vecset(x: &mut [f64], c: f64) {
    for xi in x.iter_mut() {
        *xi = c;
    }
}

#[inline(always)]
pub fn veccpy(y: &mut [f64], x: &[f64]) {
    y[..x.len()].copy_from_slice(x);
}

#[inline(always)]
pub fn vecncpy(y: &mut [f64], x: &[f64]) {
    let n = x.len();
    for i in 0..n {
        unsafe {
            *y.get_unchecked_mut(i) = -*x.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub fn vecadd(y: &mut [f64], x: &[f64], c: f64) {
    let n = x.len();
    for i in 0..n {
        unsafe {
            *y.get_unchecked_mut(i) += c * *x.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub fn vecdiff(z: &mut [f64], x: &[f64], y: &[f64]) {
    let n = z.len();
    for i in 0..n {
        unsafe {
            *z.get_unchecked_mut(i) = *x.get_unchecked(i) - *y.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub fn vecscale(y: &mut [f64], c: f64) {
    let n = y.len();
    for i in 0..n {
        unsafe {
            *y.get_unchecked_mut(i) *= c;
        }
    }
}

#[inline(always)]
pub fn vecdot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut s = 0.0f64;
    for i in 0..n {
        unsafe {
            s += *x.get_unchecked(i) * *y.get_unchecked(i);
        }
    }
    s
}

#[inline(always)]
pub fn vec2norm(x: &[f64]) -> f64 {
    vecdot(x, x).sqrt()
}

#[inline(always)]
pub fn vec2norminv(x: &[f64]) -> f64 {
    1.0 / vec2norm(x)
}

/// Sign-bit comparison matching SSE `_mm_movemask_pd` behavior.
#[inline(always)]
pub fn fsigndiff(x: f64, y: f64) -> bool {
    x.is_sign_negative() != y.is_sign_negative()
}
