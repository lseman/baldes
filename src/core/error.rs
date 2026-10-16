//! Exception model of the C++ port.
//!
//! The C++ code throws standard exceptions (invalid_argument, out_of_range,
//! logic_error, overflow_error, runtime_error) that are never caught inside
//! the algorithms: they unwind to the C API guard, which stores `what()` as
//! the last error and returns -1. This crate keeps that control flow: `throw!`
//! unwinds with an [`OrError`] payload and `guarded` (api/internal.rs)
//! catches it at the boundary. Rust bounds-check panics (the `.at()` cases)
//! are caught the same way.

use std::fmt;
use std::sync::Once;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidArgument,
    OutOfRange,
    LogicError,
    Overflow,
    Runtime,
}

#[derive(Clone, Debug)]
pub struct OrError {
    pub kind: ErrorKind,
    pub message: String,
}

impl fmt::Display for OrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[cold]
#[inline(never)]
pub fn raise(kind: ErrorKind, message: String) -> ! {
    install_quiet_hook();
    std::panic::panic_any(OrError { kind, message })
}

/// `throw std::invalid_argument(...)` and friends.
#[macro_export]
macro_rules! throw {
    (invalid_argument, $($t:tt)*) => { $crate::core::error::raise($crate::core::error::ErrorKind::InvalidArgument, format!($($t)*)) };
    (out_of_range, $($t:tt)*) => { $crate::core::error::raise($crate::core::error::ErrorKind::OutOfRange, format!($($t)*)) };
    (logic_error, $($t:tt)*) => { $crate::core::error::raise($crate::core::error::ErrorKind::LogicError, format!($($t)*)) };
    (overflow_error, $($t:tt)*) => { $crate::core::error::raise($crate::core::error::ErrorKind::Overflow, format!($($t)*)) };
    (runtime_error, $($t:tt)*) => { $crate::core::error::raise($crate::core::error::ErrorKind::Runtime, format!($($t)*)) };
}

static HOOK: Once = Once::new();

/// Thrown errors are reported through the API, as the C++ guard does, and
/// are not printed. Other panics keep the default report.
pub fn install_quiet_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.payload().downcast_ref::<OrError>().is_some() {
                return;
            }
            previous(info);
        }));
    });
}

/// `e.what()` of a caught payload; "unknown exception" as `catch (...)`.
pub fn what(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(e) = payload.downcast_ref::<OrError>() {
        return e.message.clone();
    }
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown exception".to_string()
}

/// Runs `f`, returning the thrown error instead of unwinding.
pub fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    install_quiet_hook();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| what(&*p))
}
