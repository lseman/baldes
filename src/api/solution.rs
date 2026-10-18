//! C ABI solution operations.
use super::*;

#[no_mangle]
pub unsafe extern "C" fn or_solution_get_cost(p: *const ORSolution) -> f64 {
    guard(f64::NAN, || Ok(unsafe { route(p)? }.cost))
}
#[no_mangle]
pub unsafe extern "C" fn or_solution_get_nb_nodes(p: *const ORSolution) -> i32 {
    guard(-1, || Ok(unsafe { route(p)? }.nodes.len() as i32))
}
#[no_mangle]
pub unsafe extern "C" fn or_solution_get_nodes(p: *const ORSolution, n: *mut i32) -> *const i32 {
    guard(ptr::null(), || {
        if n.is_null() {
            return Err(invalid("node count output"));
        }
        let r = unsafe { route(p)? };
        unsafe {
            *n = r.nodes.len() as i32;
        }
        Ok(r.nodes.as_ptr())
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_solution_get_arc_ids(p: *const ORSolution, n: *mut i32) -> *const i32 {
    guard(ptr::null(), || {
        if n.is_null() {
            return Err(invalid("arc count output"));
        }
        let r = unsafe { route(p)? };
        unsafe {
            *n = r.arcs.len() as i32;
        }
        Ok(r.arcs.as_ptr())
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_solution_get_res_consumption(p: *const ORSolution, r: i32) -> f64 {
    guard(f64::NAN, || {
        unsafe { route(p)? }
            .resources
            .get(count(r)?)
            .copied()
            .ok_or_else(|| invalid("solution resource out of range"))
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_solution_is_certified(p: *const ORSolution) -> i32 {
    guard(-1, || {
        let solution = unsafe { p.as_ref() }.ok_or_else(|| invalid("null solution"))?;
        Ok(solution.certified as i32)
    })
}
