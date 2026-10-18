//! C ABI fixing operations.
use super::*;

#[no_mangle]
pub unsafe extern "C" fn or_rcsp_enumerate(
    p: *mut ORECSP,
    out: *mut *mut ORSolution,
    n: i32,
) -> i32 {
    // This is the route-listing API, not the persistent enumeration controller.
    let mut minimum = 0.;
    let mut exact = 0;
    unsafe { publish(p, out, n, &mut minimum, &mut exact, false, 0) }
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_enumerated_count(p: *const ORECSP) -> i32 {
    guard(-1, || {
        Ok(unsafe { solver(p.cast_mut())? }
            .enumeration
            .as_ref()
            .map_or(0, |e| e.paths.len) as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_enumerated(p: *const ORECSP, index: i32) -> *mut ORSolution {
    guard(ptr::null_mut(), || {
        let s = unsafe { solver(p.cast_mut())? };
        let routes = s
            .enumeration
            .as_ref()
            .ok_or_else(|| invalid("no enumerated route universe"))?
            .paths
            .routes(&s.graph)?;
        let route = routes
            .into_iter()
            .nth(count(index)?)
            .ok_or_else(|| invalid("enumerated route index out of range"))?;
        Ok(Box::into_raw(Box::new(ORSolution {
            route,
            certified: true,
        })))
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_rollback_point_saved(p: *mut ORECSP, saved: i32) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? }.rollback_point_saved = saved != 0;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_pricing_requires_rollback(p: *const ORECSP) -> i32 {
    guard(-1, || {
        Ok(unsafe { solver(p.cast_mut())? }.requires_rollback as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_complete_labels_after_halfway(
    p: *mut ORECSP,
    enforce: i32,
) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? }.complete_labels(enforce != 0)?;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_restore_arcs(p: *mut ORECSP) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? }.restore_arcs();
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_eliminate_arcs(p: *mut ORECSP, cutoff: f64) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        s.complete_labels(true)?;
        let r = s.fix(cutoff)?;
        Ok((r.eliminated_forward + r.eliminated_backward) as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_run_fixing_enumeration(
    p: *mut ORECSP,
    mode: i32,
    zero: f64,
    cutoff: f64,
    eliminated: *mut i32,
    state: *mut i32,
) -> i32 {
    if !eliminated.is_null() {
        unsafe {
            *eliminated = 0;
        }
    }
    if !state.is_null() {
        unsafe {
            *state = 0;
        }
    }
    guard(-1, || {
        if eliminated.is_null() || state.is_null() {
            return Err(invalid("null fixing output"));
        }
        let result = unsafe { solver(p)? }.run_fixing_enumeration(mode, zero, cutoff)?;
        unsafe {
            *eliminated = result.eliminated as i32;
            *state = result.state;
        }
        Ok(0)
    })
}
