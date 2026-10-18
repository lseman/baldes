//! C ABI for the implemented subset of `openrscp/api/rcsp.h`.
//!
//! # Safety contract
//! Non-null opaque handles must be live handles returned by this library. Delete
//! each once; do not mutate a solver concurrently. Non-null input/output buffers
//! must be aligned and valid for their specified lengths and must not alias a
//! mutable handle. Returned route-array pointers are borrowed until that route
//! is deleted. Routes themselves own their data and outlive the solver.
//!
//! Every fallible entry point catches panics before returning through C. Errors
//! are thread-local. Missing symbols are deliberately not advertised as ported.
#![allow(clippy::missing_safety_doc)]
use crate::core::error;
use crate::pricing::Solver;
use crate::{invalid, ResourceKind, Result, Route};
use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::ptr;

pub struct ORECSP {
    solver: Solver,
}
pub struct ORSolution {
    route: Route,
    certified: bool,
}
thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn set_error(message: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
}
fn guard<T: Copy>(failure: T, f: impl FnOnce() -> Result<T>) -> T {
    match error::catch(f) {
        Ok(Ok(value)) => {
            set_error("");
            value
        }
        Ok(Err(e)) => {
            set_error(&e.message);
            failure
        }
        Err(e) => {
            set_error(&e);
            failure
        }
    }
}
unsafe fn solver<'a>(p: *mut ORECSP) -> Result<&'a mut Solver> {
    // SAFETY: Caller guarantees a live, exclusively borrowed opaque handle.
    unsafe { p.as_mut() }
        .map(|p| &mut p.solver)
        .ok_or_else(|| invalid("null solver"))
}
unsafe fn route<'a>(p: *const ORSolution) -> Result<&'a Route> {
    // SAFETY: Caller guarantees a live route handle and shared access.
    unsafe { p.as_ref() }
        .map(|p| &p.route)
        .ok_or_else(|| invalid("null solution"))
}
unsafe fn input<'a, T>(p: *const T, count: usize) -> Result<&'a [T]> {
    if count == 0 {
        return Ok(&[]);
    }
    if p.is_null() || count > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(invalid("invalid input buffer"));
    }
    // SAFETY: Non-null/length checked; validity/alignment/lifetime are ABI preconditions.
    Ok(unsafe { std::slice::from_raw_parts(p, count) })
}
fn count(n: i32) -> Result<usize> {
    usize::try_from(n).map_err(|_| invalid("negative count or ID"))
}
fn kind(t: i32) -> Result<ResourceKind> {
    match t {
        0 => Ok(ResourceKind::Disposable),
        1 => Ok(ResourceKind::Equality),
        _ => Err(invalid("invalid resource type")),
    }
}
fn resource(s: &Solver, id: i32) -> Result<usize> {
    let r = count(id)?;
    if r >= s.graph.resources.len() {
        return Err(invalid("resource out of range"));
    }
    Ok(r)
}

mod handles;
pub use handles::*;
mod graph;
pub use graph::*;
mod options;
pub use options::*;
mod pricing;
pub use pricing::*;
mod solution;
pub use solution::*;
mod relaxation;
pub use relaxation::*;
mod fixing;
pub use fixing::*;
mod statistics;
use pricing::publish;
pub use statistics::*;
