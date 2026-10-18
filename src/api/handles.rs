//! C ABI handles operations.
use super::*;

#[no_mangle]
pub extern "C" fn or_rcsp_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn or_rcsp_new(n: i32, r: i32) -> *mut ORECSP {
    guard(ptr::null_mut(), || {
        Ok(Box::into_raw(Box::new(ORECSP {
            solver: Solver::new(count(n)?, count(r)?)?,
        })))
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_delete(p: *mut ORECSP) {
    if !p.is_null() {
        // SAFETY: The ABI transfers exclusive ownership back exactly once.
        drop(unsafe { Box::from_raw(p) });
    }
}
#[no_mangle]
pub unsafe extern "C" fn or_solution_delete(p: *mut ORSolution) {
    if !p.is_null() {
        // SAFETY: The ABI transfers exclusive ownership back exactly once.
        drop(unsafe { Box::from_raw(p) });
    }
}
