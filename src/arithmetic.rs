/// Vector operations.
/// Default: AVX2+FMA SIMD on x86_64.
/// Without feature "simd": scalar-only, bit-exact with C libLBFGS.

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
use std::arch::x86_64::*;

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
        unsafe { *y.get_unchecked_mut(i) = -*x.get_unchecked(i); }
    }
}

#[inline(always)]
pub fn vecadd(y: &mut [f64], x: &[f64], c: f64) {
    let n = x.len();
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if n >= 4 {
            unsafe { vecadd_simd(y, x, c, n); }
            return;
        }
    }
    for i in 0..n {
        unsafe { *y.get_unchecked_mut(i) += c * *x.get_unchecked(i); }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vecadd_simd(y: &mut [f64], x: &[f64], c: f64, n: usize) {
    let xp = x.as_ptr();
    let yp = y.as_mut_ptr();
    let vc = _mm256_set1_pd(c);
    let mut i = 0;
    while i + 4 <= n {
        let xv = _mm256_loadu_pd(xp.add(i));
        let yv = _mm256_loadu_pd(yp.add(i));
        let yv = _mm256_fmadd_pd(xv, vc, yv);
        _mm256_storeu_pd(yp.add(i), yv);
        i += 4;
    }
    while i < n {
        *yp.add(i) += c * *xp.add(i);
        i += 1;
    }
}

#[inline(always)]
pub fn vecdiff(z: &mut [f64], x: &[f64], y: &[f64]) {
    let n = z.len();
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if n >= 4 {
            unsafe { vecdiff_simd(z, x, y, n); }
            return;
        }
    }
    for i in 0..n {
        unsafe { *z.get_unchecked_mut(i) = *x.get_unchecked(i) - *y.get_unchecked(i); }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
unsafe fn vecdiff_simd(z: &mut [f64], x: &[f64], y: &[f64], n: usize) {
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    let zp = z.as_mut_ptr();
    let mut i = 0;
    while i + 4 <= n {
        let xv = _mm256_loadu_pd(xp.add(i));
        let yv = _mm256_loadu_pd(yp.add(i));
        _mm256_storeu_pd(zp.add(i), _mm256_sub_pd(xv, yv));
        i += 4;
    }
    while i < n {
        *zp.add(i) = *xp.add(i) - *yp.add(i);
        i += 1;
    }
}

#[inline(always)]
pub fn vecscale(y: &mut [f64], c: f64) {
    let n = y.len();
    for i in 0..n {
        unsafe { *y.get_unchecked_mut(i) *= c; }
    }
}

#[inline(always)]
pub fn vecdot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if n >= 4 {
            return unsafe { vecdot_simd(x, y, n) };
        }
    }
    let mut s = 0.0f64;
    for i in 0..n {
        unsafe { s += *x.get_unchecked(i) * *y.get_unchecked(i); }
    }
    s
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn vecdot_simd(x: &[f64], y: &[f64], n: usize) -> f64 {
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut i = 0;
    while i + 8 <= n {
        let x0 = _mm256_loadu_pd(xp.add(i));
        let y0 = _mm256_loadu_pd(yp.add(i));
        let x1 = _mm256_loadu_pd(xp.add(i + 4));
        let y1 = _mm256_loadu_pd(yp.add(i + 4));
        acc0 = _mm256_fmadd_pd(x0, y0, acc0);
        acc1 = _mm256_fmadd_pd(x1, y1, acc1);
        i += 8;
    }
    while i + 4 <= n {
        let x0 = _mm256_loadu_pd(xp.add(i));
        let y0 = _mm256_loadu_pd(yp.add(i));
        acc0 = _mm256_fmadd_pd(x0, y0, acc0);
        i += 4;
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    let hi128 = _mm256_extractf128_pd(acc0, 1);
    let lo128 = _mm256_castpd256_pd128(acc0);
    let sum128 = _mm_add_pd(lo128, hi128);
    let hi64 = _mm_shuffle_pd(sum128, sum128, 1);
    let sum64 = _mm_add_pd(sum128, hi64);
    let mut s: f64 = 0.0;
    _mm_store_sd(&mut s, sum64);
    while i < n {
        s += *xp.add(i) * *yp.add(i);
        i += 1;
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

#[inline(always)]
pub fn fsigndiff(x: f64, y: f64) -> bool {
    x.is_sign_negative() != y.is_sign_negative()
}
