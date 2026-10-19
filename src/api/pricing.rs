//! C ABI pricing operations.
use super::*;

unsafe fn update_costs(
    p: *mut ORECSP,
    costs: *const f64,
    n: i32,
    duals: *const f64,
    cut_duals: *const f64,
    cuts: i32,
    rounded: bool,
) -> Result<i32> {
    if cuts != 0 {
        return Err(invalid("cuts are deferred in the Rust port"));
    }
    let _ = cut_duals;
    let s = unsafe { solver(p)? };
    let costs = unsafe { input(costs, count(n)?)? };
    let duals = unsafe { input(duals, s.graph.vertices.len())? };
    s.graph.update_reduced_costs(costs, duals, rounded)?;
    Ok(0)
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_update_reduced_costs(
    p: *mut ORECSP,
    c: *const f64,
    n: i32,
    d: *const f64,
    cd: *const f64,
    nc: i32,
) -> i32 {
    guard(-1, || unsafe { update_costs(p, c, n, d, cd, nc, false) })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_update_reduced_costs_rounded(
    p: *mut ORECSP,
    c: *const f64,
    n: i32,
    d: *const f64,
    cd: *const f64,
    nc: i32,
) -> i32 {
    guard(-1, || unsafe { update_costs(p, c, n, d, cd, nc, true) })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_solve(
    p: *mut ORECSP,
    cost: *mut f64,
    out: *mut *mut ORSolution,
) -> i32 {
    // SAFETY: Optional caller-provided output pointers are writable when non-null.
    if !out.is_null() {
        unsafe { *out = ptr::null_mut() };
    }
    if !cost.is_null() {
        unsafe { *cost = f64::INFINITY };
    }
    guard(-1, || {
        if cost.is_null() || out.is_null() {
            return Err(invalid("solve output"));
        }
        let result = unsafe { solver(p)? }.price(1, false)?;
        if let Some(best) = result.columns.into_iter().next() {
            unsafe {
                *cost = best.cost;
                *out = Box::into_raw(Box::new(ORSolution {
                    route: best,
                    certified: result.exact,
                }));
            }
            Ok(1)
        } else {
            Ok(0)
        }
    })
}
pub(super) unsafe fn publish(
    p: *mut ORECSP,
    out: *mut *mut ORSolution,
    n: i32,
    minimum: *mut f64,
    exact: *mut i32,
    negative: bool,
    phase: usize,
) -> i32 {
    if !out.is_null() && n > 0 {
        unsafe { std::slice::from_raw_parts_mut(out, n as usize) }.fill(ptr::null_mut());
    }
    if !minimum.is_null() {
        unsafe { *minimum = f64::INFINITY };
    }
    if !exact.is_null() {
        unsafe { *exact = 0 };
    }
    guard(-1, || {
        let maximum = count(n)?;
        if maximum == 0 || out.is_null() || minimum.is_null() || exact.is_null() {
            return Err(invalid("pricing output"));
        }
        // SAFETY: out has maximum writable pointer slots by ABI contract.
        let slots = unsafe { std::slice::from_raw_parts_mut(out, maximum) };
        slots.fill(ptr::null_mut());
        let s = unsafe { solver(p)? };
        let result = if phase == usize::MAX {
            s.price(maximum, negative)?
        } else {
            s.price_phase(phase, maximum, negative)?
        };
        let count = result.columns.len();
        for (slot, route) in slots.iter_mut().zip(result.columns) {
            *slot = Box::into_raw(Box::new(ORSolution {
                route,
                certified: result.exact,
            }));
        }
        unsafe {
            *minimum = result.minimum_reduced_cost;
            *exact = i32::from(result.exact);
        }
        Ok(count as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_price(
    p: *mut ORECSP,
    out: *mut *mut ORSolution,
    n: i32,
    minimum: *mut f64,
    exact: *mut i32,
) -> i32 {
    unsafe { publish(p, out, n, minimum, exact, true, usize::MAX) }
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_price_phase(
    p: *mut ORECSP,
    index: i32,
    out: *mut *mut ORSolution,
    n: i32,
    minimum: *mut f64,
    exact: *mut i32,
) -> i32 {
    unsafe { publish(p, out, n, minimum, exact, false, index as usize) }
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_arc_variable_coefficients(
    p: *mut ORECSP,
    id: i32,
    pure: f64,
    variables: *const i32,
    coefficients: *const f64,
    n: i32,
) -> i32 {
    guard(-1, || {
        if !pure.is_finite() {
            return Err(invalid("finite pure arc cost required"));
        }
        let ids = unsafe { input(variables, count(n)?)? };
        let values = unsafe { input(coefficients, ids.len())? };
        let mut entries = vec![];
        for (&id, &a) in ids.iter().zip(values) {
            if !a.is_finite() {
                return Err(invalid("finite variable coefficient required"));
            }
            entries.push((count(id)?, a));
        }
        let s = unsafe { solver(p)? };
        let arc = s.graph.arc_mut(id)?;
        arc.cost = pure;
        arc.variables = entries;
        s.variable_costs.clear();
        s.restore_arcs();
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_maximum_variable_id(p: *mut ORECSP, id: i32) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        s.maximum_variable_id = if id == -1 { None } else { Some(count(id)?) };
        s.variable_costs.clear();
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_cache_variable_costs(
    p: *mut ORECSP,
    costs: *const f64,
    n: i32,
) -> i32 {
    guard(-1, || {
        let costs = unsafe { input(costs, count(n)?)? };
        unsafe { solver(p)? }.cache_variable_costs(costs)?;
        Ok(0)
    })
}
/// Return raw variable-cost candidates and pricing bounds. Variable-cost
/// validation shares the pricing budget and may require a second sweep over
/// the domain before cost-dependent fixing. A heuristic phase leaves the lower
/// bound unknown (negative infinity). Empty calls use `zero` as the upper
/// baseline and, for an exact phase, the lower baseline.
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_price_functor_phase(
    p: *mut ORECSP,
    index: i32,
    zero: f64,
    out: *mut *mut ORSolution,
    n: i32,
    lower: *mut f64,
    upper: *mut f64,
    state: *mut i32,
    column_count: *mut i32,
) -> i32 {
    if !lower.is_null() {
        unsafe {
            *lower = 1e12;
        }
    }
    if !upper.is_null() {
        unsafe {
            *upper = 1e12;
        }
    }
    if !state.is_null() {
        unsafe {
            *state = 3;
        }
    }
    if !column_count.is_null() {
        unsafe {
            *column_count = 0;
        }
    }
    guard(-1, || {
        let maximum = count(n)?;
        if lower.is_null()
            || upper.is_null()
            || state.is_null()
            || column_count.is_null()
            || out.is_null()
            || maximum == 0
            || !zero.is_finite()
        {
            return Err(invalid("functor output arguments"));
        }
        let slots = unsafe { std::slice::from_raw_parts_mut(out, maximum) };
        slots.fill(ptr::null_mut());
        let s = unsafe { solver(p)? };
        if index < 0 {
            return Ok(0);
        }
        let result = match s.price_functor_phase(index as usize, maximum) {
            Ok(r) => r,
            Err(_) => {
                unsafe {
                    *state = if s.requires_rollback { 2 } else { 3 };
                }
                return Ok(0);
            }
        };
        let upper_bound = result
            .columns
            .iter()
            .map(|route| route.cost)
            .reduce(f64::min)
            .unwrap_or(zero);
        let lower_bound = if result.exact {
            if result.columns.is_empty() {
                zero
            } else {
                result.minimum_reduced_cost
            }
        } else {
            f64::NEG_INFINITY
        };
        let count = result.columns.len();
        for (slot, route) in slots.iter_mut().zip(result.columns) {
            *slot = Box::into_raw(Box::new(ORSolution {
                route,
                certified: result.exact,
            }));
        }
        unsafe {
            *lower = lower_bound;
            *upper = upper_bound;
            *state = 0;
            *column_count = count as i32;
        }
        Ok((count > 0) as i32)
    })
}
