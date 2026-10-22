//! Log output formatted as iostream messages (core/Log.h).
//!
//! `operator<<` writes a floating value like printf("%g") with precision 6,
//! a bool as 0/1. [`G`] wraps a double so `{}` reproduces that text.

use std::fmt;

/// A double written as `os << value` (printf "%g").
#[derive(Clone, Copy)]
pub struct G(pub f64);

impl fmt::Display for G {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_g(self.0, 6))
    }
}

/// printf("%.*g", precision, x).
pub fn format_g(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() {
            "-nan".into()
        } else {
            "nan".into()
        };
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = if precision == 0 { 1 } else { precision };
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    // Exponent X of the %e conversion at precision P-1 (after rounding).
    let e_repr = format!("{:.*e}", p - 1, x);
    let (mantissa, exp) = e_repr.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if exp < -4 || exp >= p as i32 {
        let mantissa = strip_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        let abs = exp.unsigned_abs();
        if abs < 10 {
            format!("{}e{}0{}", mantissa, sign, abs)
        } else {
            format!("{}e{}{}", mantissa, sign, abs)
        }
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_zeros(&format!("{:.*}", decimals, x)).to_string()
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

/// `std::cout << ... << std::endl`.
#[macro_export]
macro_rules! log_out {
    ($($t:tt)*) => {{
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, $($t)*);
        let _ = out.flush();
    }};
}

/// `std::cout << ...` without newline.
#[macro_export]
macro_rules! log_out_nonl {
    ($($t:tt)*) => {{
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = write!(out, $($t)*);
    }};
}

/// `std::cerr << ... << std::endl`.
#[macro_export]
macro_rules! log_err {
    ($($t:tt)*) => {{
        use std::io::Write;
        let mut out = std::io::stderr().lock();
        let _ = writeln!(out, $($t)*);
        let _ = out.flush();
    }};
}

#[cfg(test)]
#[path = "../../tests/unit/core/log/tests.rs"]
mod tests;
