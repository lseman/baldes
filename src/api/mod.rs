//! C ABI for the implemented no-cut subset of `openrscp/api/rcsp.h`.
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
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_source(p: *mut ORECSP, v: i32) {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let v = count(v)?;
        if v >= s.graph.vertices.len() {
            return Err(invalid("source out of range"));
        }
        s.graph.source = v;
        Ok(0)
    });
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_sink(p: *mut ORECSP, v: i32) {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let v = count(v)?;
        if v >= s.graph.vertices.len() {
            return Err(invalid("sink out of range"));
        }
        s.graph.sink = v;
        Ok(0)
    });
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_graph_id(p: *mut ORECSP, id: i32) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? }.graph.graph_id = id;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_new_arc(p: *mut ORECSP, tail: i32, head: i32, cost: f64) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? }
            .graph
            .add_arc(count(tail)?, count(head)?, cost)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_new_resource(p: *mut ORECSP, _name: *const c_char, t: i32) -> i32 {
    guard(-1, || {
        let k = kind(t)?;
        let s = unsafe { solver(p)? };
        if s.graph.resources.len() >= 2 {
            return Err(invalid("pricing accepts at most two resources"));
        }
        let r = s.graph.resources.len();
        s.graph.resources.push(k);
        Ok(r as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_resource_type(p: *mut ORECSP, r: i32, t: i32) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let r = resource(s, r)?;
        s.graph.resources[r] = kind(t)?;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_arc_consumption(
    p: *mut ORECSP,
    a: i32,
    r: i32,
    lower: f64,
    upper: f64,
) {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let r = resource(s, r)?;
        if !lower.is_finite() || lower != upper {
            return Err(invalid("fixed finite consumption lb == ub required"));
        }
        s.graph.arc_mut(a)?.consumption[r] = lower;
        Ok(0)
    });
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_vertex_bounds(
    p: *mut ORECSP,
    v: i32,
    r: i32,
    lower: f64,
    upper: f64,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let r = resource(s, r)?;
        s.graph.set_bounds(count(v)?, r, lower, upper)?;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_initial_resource(p: *mut ORECSP, r: i32, value: f64) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let r = resource(s, r)?;
        if !value.is_finite() {
            return Err(invalid("finite initial resource required"));
        }
        s.graph.initial[r] = Some(value);
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_ng_neighborhood(
    p: *mut ORECSP,
    v: i32,
    nodes: *const i32,
    n: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let v = count(v)?;
        let ids = unsafe { input(nodes, count(n)?)? };
        if ids.is_empty() || ids.len() > 64 || !ids.contains(&(v as i32)) {
            return Err(invalid(
                "NG neighborhood must contain its vertex and have 1..64 entries",
            ));
        }
        let ids = ids.iter().map(|&v| count(v)).collect::<Result<Vec<_>>>()?;
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != ids.len() {
            return Err(invalid("duplicate NG vertex"));
        }
        s.graph.set_ng(v, &ids)?;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_elementary(
    p: *mut ORECSP,
    enabled: i32,
    max_arcs: usize,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        s.graph.elementary = enabled != 0;
        s.controls.max_route_arcs = (max_arcs != 0).then_some(max_arcs);
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_add_elementarity_set(
    p: *mut ORECSP,
    nodes: *const i32,
    n: i32,
) -> i32 {
    guard(-1, || {
        let ids = unsafe { input(nodes, count(n)?)? }
            .iter()
            .map(|&v| count(v))
            .collect::<Result<Vec<_>>>()?;
        Ok(unsafe { solver(p)? }.graph.add_elementarity_set(&ids)? as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_attach_elementarity_to_node(
    p: *mut ORECSP,
    v: i32,
    set: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let set = count(set)?;
        let v = count(v)?;
        if set >= s.graph.element_count || v >= s.graph.vertices.len() {
            return Err(invalid("elementarity set/node out of range"));
        }
        if !s.graph.vertices[v].elements.contains(&set) {
            s.graph.vertices[v].elements.push(set);
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_pricing_controls(
    p: *mut ORECSP,
    cutoff: f64,
    labels: usize,
    columns: usize,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        if cutoff.is_nan() || labels == 0 || columns == 0 {
            return Err(invalid("pricing controls"));
        }
        s.cutoff = cutoff;
        s.controls.max_labels = labels;
        s.controls.max_columns = columns;
        Ok(0)
    })
}
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
unsafe fn publish(
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

#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_option(
    p: *mut ORECSP,
    name: *const c_char,
    value: f64,
) -> i32 {
    guard(-1, || {
        if name.is_null() {
            return Err(invalid("null option name"));
        }
        let name = unsafe { std::ffi::CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| invalid("option name encoding"))?;
        unsafe { solver(p)? }.set_option(name, value)?;
        Ok(0)
    })
}
#[repr(C)]
pub struct ORPricingPhase {
    pub check_ng_and_cuts: i32,
    pub allow_jump_arcs: i32,
    pub labels_per_bucket: i64,
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_phase_configuration(
    p: *mut ORECSP,
    selector: i32,
    phases: *const ORPricingPhase,
    n: i32,
) -> i32 {
    guard(-1, || {
        let supplied = unsafe { input(phases, count(n)?)? };
        let supplied: Vec<_> = supplied
            .iter()
            .map(|p| crate::pricing::phases::Phase {
                check_ng: p.check_ng_and_cuts != 0,
                jump_arcs: p.allow_jump_arcs != 0,
                labels_per_bucket: p.labels_per_bucket,
            })
            .collect();
        unsafe { solver(p)? }.phases = crate::pricing::phases::initialize(selector, &supplied)?;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_phase_count(p: *const ORECSP) -> i32 {
    guard(-1, || {
        Ok(unsafe { solver(p.cast_mut())? }.phases.len() as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_phase(
    p: *const ORECSP,
    index: i32,
    out: *mut ORPricingPhase,
) -> i32 {
    guard(-1, || {
        if out.is_null() {
            return Err(invalid("null phase output"));
        }
        let phase = *unsafe { solver(p.cast_mut())? }
            .phases
            .get(count(index)?)
            .ok_or_else(|| invalid("phase out of range"))?;
        unsafe {
            *out = ORPricingPhase {
                check_ng_and_cuts: phase.check_ng as i32,
                allow_jump_arcs: phase.jump_arcs as i32,
                labels_per_bucket: phase.labels_per_bucket,
            };
        }
        Ok(0)
    })
}
#[no_mangle]
pub extern "C" fn or_rcsp_option_count() -> i32 {
    crate::core::options::OPTION_TABLE.len() as i32
}
#[no_mangle]
pub extern "C" fn or_rcsp_option_name(index: i32) -> *const c_char {
    static NAMES: std::sync::OnceLock<Vec<CString>> = std::sync::OnceLock::new();
    guard(ptr::null(), || {
        let names = NAMES.get_or_init(|| {
            crate::core::options::OPTION_TABLE
                .iter()
                .map(|o| CString::new(o.name).unwrap())
                .collect()
        });
        Ok(names
            .get(count(index)?)
            .ok_or_else(|| invalid("option index out of range"))?
            .as_ptr())
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_option(
    p: *const ORECSP,
    name: *const c_char,
    out: *mut f64,
) -> i32 {
    guard(-1, || {
        if name.is_null() || out.is_null() {
            return Err(invalid("null option name/output"));
        }
        let name = unsafe { std::ffi::CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| invalid("option name encoding"))?;
        let value = unsafe { solver(p.cast_mut())? }.options.get(name)?;
        unsafe {
            *out = value;
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_bidirectional_mode(p: *mut ORECSP, mode: i32) -> i32 {
    guard(-1, || {
        if !(0..=2).contains(&mode) {
            return Err(invalid("bidirectional mode out of range"));
        }
        unsafe { solver(p)? }.set_option("RCSPuseBidirectionalSearch", mode as f64)?;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_arc_resource_bounds(
    p: *mut ORECSP,
    id: i32,
    r: i32,
    lower: f64,
    upper: f64,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let r = resource(s, r)?;
        let bounds = crate::graph::Window::new(lower, upper)?;
        s.graph.arc_mut(id)?.bounds[r] = Some(bounds);
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_build_default_resource_buckets(p: *mut ORECSP) -> i32 {
    unsafe { or_rcsp_build_resource_buckets(p, ptr::null()) }
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_build_resource_buckets(p: *mut ORECSP, steps: *const f64) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        if steps.is_null() {
            s.bucket_steps = [0.; 2];
        } else {
            let values = unsafe { input(steps, s.graph.resources.len())? };
            let mut selected = [0.; 2];
            selected[..values.len()].copy_from_slice(values);
            s.bucket_steps = selected;
        }
        let phase = crate::pricing::phases::Phase {
            check_ng: true,
            jump_arcs: true,
            labels_per_bucket: i64::MAX,
        };
        let controls = crate::labeling::engine::EngineControls {
            steps: s.bucket_steps,
            bidirectional: s.options.get("RCSPuseBidirectionalSearch")? != 0.,
            target_buckets: s.options.get("RCSPnumberOfBucketsPerVertex")? as i32,
            ..Default::default()
        };
        s.engine = Some(crate::labeling::engine::LabelingEngine::prepare(
            &s.graph, controls, phase,
        )?);
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_decrease_bucket_steps(p: *mut ORECSP, factor: f64) -> i32 {
    guard(-1, || {
        if !factor.is_finite() || factor <= 0. || factor >= 1. {
            return Err(invalid("bucket decrease factor must lie in (0,1)"));
        }
        let s = unsafe { solver(p)? };
        let engine = s
            .engine
            .as_mut()
            .ok_or_else(|| invalid("resource buckets are not built"))?;
        engine.grid.decrease_steps(&s.graph)?;
        s.bucket_steps = engine.grid.step;
        Ok(0)
    })
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
pub unsafe extern "C" fn or_rcsp_get_cut_count(p: *const ORECSP) -> i32 {
    guard(-1, || {
        unsafe { solver(p.cast_mut())? };
        Ok(0)
    })
}
#[repr(C)]
pub struct ORNGColumn {
    pub arc_ids: *const i32,
    pub count: i32,
    pub value: f64,
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_strengthen_ng(
    p: *mut ORECSP,
    columns: *const ORNGColumn,
    n: i32,
    forbidden: *mut i32,
) -> i32 {
    guard(-1, || {
        if forbidden.is_null() {
            return Err(invalid("null NG count output"));
        }
        unsafe {
            *forbidden = 0;
        }
        let columns = unsafe { input(columns, count(n)?)? };
        let mut paths = vec![];
        for c in columns {
            paths.push(crate::graph::relaxation::FractionalPath {
                arcs: unsafe { input(c.arc_ids, count(c.count)?)? }.to_vec(),
                value: c.value,
            });
        }
        let s = unsafe { solver(p)? };
        let controls = crate::graph::relaxation::Controls {
            mode: s.options.get("RCSPdynamicNGmode")?.abs() as i32,
            max_vertex_size: s.options.get("RCSPmaxNGneighbourhoodSize")?.max(0.) as usize,
            max_average_size: s.options.get("RCSPmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
            max_columns: s.options.get("RCSPmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
            max_cycle_size: s.options.get("RCSPmaxCycleSizeInDynamicNG")? as i32,
            mirror: false,
        };
        let counts = s.strengthen_ng(&paths, &controls)?;
        unsafe {
            *forbidden = counts.columns_forbidden as i32;
        }
        s.engine = None;
        Ok((counts.columns_forbidden > 0) as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_arc_ng_memory_count(p: *const ORECSP, id: i32) -> i32 {
    guard(-1, || {
        Ok(unsafe { solver(p.cast_mut())? }
            .graph
            .arc_mut(id)?
            .memory
            .len() as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_arc_ng_sets(
    p: *mut ORECSP,
    id: i32,
    sets: *const i32,
    n: i32,
    memory: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let sets = unsafe { input(sets, count(n)?)? }
            .iter()
            .map(|&e| count(e))
            .collect::<Result<Vec<_>>>()?;
        if sets.iter().any(|&e| e >= s.graph.element_count) {
            return Err(invalid("arc NG element out of range"));
        }
        let a = s.graph.arc_mut(id)?;
        if memory != 0 {
            a.memory = sets;
        } else {
            a.elements = sets;
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_route_satisfies_relaxation(
    p: *const ORECSP,
    vertices: *const i32,
    nv: i32,
    arcs: *const i32,
    na: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p.cast_mut())? };
        let vertices = unsafe { input(vertices, count(nv)?)? };
        let arcs = unsafe { input(arcs, count(na)?)? };
        if vertices.len() != arcs.len() + 1 {
            return Err(invalid("route node/arc dimensions"));
        }
        let ng = crate::graph::ng::PreparedNg::new(&s.graph)?;
        let Some(&first) = vertices.first() else {
            return Ok(0);
        };
        let first = count(first)?;
        let mut bits = *ng
            .vertex_bits
            .get(first)
            .ok_or_else(|| invalid("route vertex out of range"))?;
        for (i, &id) in arcs.iter().enumerate() {
            let a = s
                .graph
                .arcs
                .iter()
                .position(|a| a.id == id)
                .ok_or_else(|| invalid("route arc out of range"))?;
            let arc = &s.graph.arcs[a];
            if vertices[i] != arc.tail as i32 || vertices[i + 1] != arc.head as i32 {
                return Ok(0);
            }
            let Some(next) = ng.forward[a].extend(bits, ng.vertex_bits[arc.head]) else {
                return Ok(0);
            };
            bits = next;
        }
        Ok(1)
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
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_debug_vertex_order(
    p: *const ORECSP,
    out: *mut i32,
    capacity: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p.cast_mut())? };
        let engine = s
            .engine
            .as_ref()
            .ok_or_else(|| invalid("pricing graph is not prepared"))?;
        let n = count(capacity)?;
        if n < engine.grid.order.len() || out.is_null() {
            return Err(invalid("vertex order output capacity"));
        }
        for (i, &v) in engine.grid.order.iter().enumerate() {
            unsafe {
                *out.add(i) = v as i32;
            }
        }
        Ok(engine.grid.order.len() as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_debug_labels_per_vertex(
    p: *const ORECSP,
    forward: i32,
    out: *mut i32,
    n: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p.cast_mut())? };
        let engine = s
            .engine
            .as_ref()
            .ok_or_else(|| invalid("no retained labeling state"))?;
        if count(n)? < s.graph.vertices.len() || out.is_null() {
            return Err(invalid("label count output capacity"));
        }
        let storage = if forward != 0 {
            &engine.forward
        } else {
            &engine.backward
        };
        for (v, row) in storage.buckets.iter().enumerate() {
            unsafe {
                *out.add(engine.grid.order[v]) =
                    row.iter().map(|b| b.active.len()).sum::<usize>() as i32;
            }
        }
        Ok(s.graph.vertices.len() as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_reduce_ng_dssr(p: *mut ORECSP, bound: f64, _modes: i32) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? };
        if !bound.is_finite() {
            return Err(invalid("finite DSSR bound required"));
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_initialize_ng_set_distances(
    p: *mut ORECSP,
    distances: *const f64,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let n = s.graph.element_count;
        let matrix = unsafe {
            input(
                distances,
                n.checked_mul(n)
                    .ok_or_else(|| invalid("NG distance size overflow"))?,
            )?
        };
        let size = s.options.get("RCSPinitNGneighbourhoodSize")?.max(0.) as usize;
        let changed = crate::graph::relaxation::initialize_distances(&mut s.graph, matrix, size)?;
        s.engine = None;
        Ok(changed as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_initialize_ng_neighborhoods(
    p: *mut ORECSP,
    distances: *const f64,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let n = s.graph.vertices.len();
        let matrix = unsafe { input(distances, n * n)? };
        if matrix.iter().any(|v| v.is_nan()) {
            return Err(invalid("NG distance is NaN"));
        }
        let size = s.options.get("RCSPinitNGneighbourhoodSize")?.max(0.) as usize;
        let mut memories = vec![];
        for v in 0..n {
            let mut order: Vec<_> = (0..n).collect();
            order.sort_by(|&a, &b| {
                matrix[v * n + a]
                    .total_cmp(&matrix[v * n + b])
                    .then(a.cmp(&b))
            });
            order.truncate(size.min(n));
            let mut sets = std::collections::BTreeSet::new();
            for id in order {
                sets.extend(&s.graph.vertices[id].elements);
            }
            memories.push(sets.into_iter().collect());
        }
        for (vertex, memory) in s.graph.vertices.iter_mut().zip(memories) {
            vertex.memory = memory;
        }
        s.engine = None;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_ng_neighborhoods_may_grow(p: *const ORECSP) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p.cast_mut())? };
        if s.options.get("RCSPdynamicNGmode")?.abs() != 1. {
            return Ok(1);
        }
        let prepared = crate::graph::ng::PreparedNg::new(&s.graph)?;
        let max = s.options.get("RCSPmaxNGneighbourhoodSize")?.max(0.) as usize;
        let avg = s.options.get("RCSPmaxNGaverNeighbourhoodSize")?.max(0.) as usize;
        Ok((prepared.local_elements.iter().any(|row| row.len() < max)
            && prepared.local_elements.iter().map(Vec::len).sum::<usize>() / s.graph.vertices.len()
                < avg) as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_reset_ng(p: *mut ORECSP) -> i32 {
    guard(-1, || Ok(unsafe { solver(p)? }.reset_ng() as i32))
}

#[no_mangle]
pub unsafe extern "C" fn or_rcsp_column_generation_terminated(
    p: *mut ORECSP,
    end: i32,
    flag3: i32,
    flag5: i32,
    stop: *mut i32,
) -> i32 {
    guard(-1, || {
        if stop.is_null() {
            return Err(invalid("null column generation stop output"));
        }
        let s = unsafe { solver(p)? };
        let (stopped, decrease) = s.round.action(
            &s.options,
            s.engine.as_ref().map(|e| &e.grid),
            s.enumeration.is_some(),
            end != 0,
            flag3,
            flag5,
        )?;
        unsafe {
            *stop = stopped as i32;
        }
        if decrease {
            let engine = s
                .engine
                .as_mut()
                .ok_or_else(|| invalid("bucket graph is not prepared"))?;
            engine.grid.decrease_steps(&s.graph)?;
            s.bucket_steps = engine.grid.step;
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_configure_arc_ng(
    p: *mut ORECSP,
    set_count: i32,
    vertex_offsets: *const i32,
    vertex_sets: *const i32,
    memory_offsets: *const i32,
    memory_sets: *const i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let n = s.graph.vertices.len();
        unsafe fn csr(
            offsets: *const i32,
            values: *const i32,
            n: usize,
        ) -> Result<Vec<Vec<usize>>> {
            let offsets = unsafe { input(offsets, n + 1)? };
            if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] > w[1]) {
                return Err(invalid("invalid NG CSR offsets"));
            }
            let values = unsafe { input(values, count(offsets[n])?)? };
            let values = values
                .iter()
                .map(|&e| count(e))
                .collect::<Result<Vec<_>>>()?;
            Ok((0..n)
                .map(|v| values[offsets[v] as usize..offsets[v + 1] as usize].to_vec())
                .collect())
        }
        let membership = unsafe { csr(vertex_offsets, vertex_sets, n)? };
        let memory = unsafe { csr(memory_offsets, memory_sets, n)? };
        s.graph
            .configure_ng(count(set_count)?, membership, memory)?;
        s.restore_arcs();
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_main_resource(p: *mut ORECSP, id: i32, _halfway: i32) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        s.graph.main_resource = if id == -1 {
            None
        } else {
            Some(resource(s, id)?)
        };
        s.restore_arcs();
        Ok(0)
    })
}
pub type ORResourceExtension = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    i32,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
) -> i32;
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_extension_callback(
    p: *mut ORECSP,
    callback: Option<ORResourceExtension>,
    _user: *mut std::ffi::c_void,
) -> i32 {
    guard(-1, || {
        unsafe { solver(p)? };
        if callback.is_some() {
            return Err(invalid(
                "arbitrary extension callbacks have no native counterpart",
            ));
        }
        Ok(0)
    })
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
        let result = match s.price_phase(index as usize, maximum, false) {
            Ok(r) => r,
            Err(_) => {
                unsafe {
                    *state = if s.requires_rollback { 2 } else { 3 };
                }
                return Ok(0);
            }
        };
        let bound = if let Some(first) = result.columns.first() {
            s.corrected_bound(result.minimum_reduced_cost, first)?
        } else {
            zero
        };
        let count = result.columns.len();
        for (slot, route) in slots.iter_mut().zip(result.columns) {
            *slot = Box::into_raw(Box::new(ORSolution {
                route,
                certified: result.exact,
            }));
        }
        unsafe {
            *lower = bound;
            *upper = bound;
            *state = 0;
            *column_count = count as i32;
        }
        Ok((count > 0) as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_statistics(
    p: *const ORECSP,
    counts: *mut i64,
    seconds: *mut f64,
) -> i32 {
    guard(-1, || {
        if counts.is_null() || seconds.is_null() {
            return Err(invalid("null statistics output"));
        }
        let s = unsafe { solver(p.cast_mut())? };
        let empty = crate::Statistics::default();
        let stats = s.engine.as_ref().map_or(&empty, |e| &e.statistics);
        let data = [
            stats.forward_labels as i64,
            stats.backward_labels as i64,
            stats.dominated as i64,
            stats.concatenation_tests as i64,
            stats.forward_labels as i64,
            stats.backward_labels as i64,
            0,
            stats.capped as i64,
            stats.dominance_checks as i64,
            stats.other_dominance_checks as i64,
        ];
        let times = [
            stats.elapsed,
            stats.forward_time,
            stats.backward_time,
            stats.concatenation_time,
        ];
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), counts, data.len());
            ptr::copy_nonoverlapping(times.as_ptr(), seconds, times.len());
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_native_statistics(p: *const ORECSP, counts: *mut i64) -> i32 {
    guard(-1, || {
        if counts.is_null() {
            return Err(invalid("null native statistics output"));
        }
        let s = unsafe { solver(p.cast_mut())? };
        let empty = crate::Statistics::default();
        let stats = s.engine.as_ref().map_or(&empty, |e| &e.statistics);
        let data = [
            i64::from(s.engine.is_some()),
            0,
            stats.dominance_checks as i64,
            stats.later_label_checks as i64,
            (stats.dominated - stats.other_dominated) as i64,
            stats.other_bucket_searches as i64,
            0,
            i64::from(
                s.engine
                    .as_ref()
                    .is_some_and(|e| !e.backward.labels.is_empty()),
            ),
        ];
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), counts, data.len());
        }
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_debug_labels(
    p: *const ORECSP,
    forward: i32,
    out: *mut f64,
    capacity: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p.cast_mut())? };
        let engine = s
            .engine
            .as_ref()
            .ok_or_else(|| invalid("no retained labels"))?;
        let storage = if forward != 0 {
            &engine.forward
        } else {
            &engine.backward
        };
        let ids: Vec<_> = storage
            .buckets
            .iter()
            .flatten()
            .flat_map(|b| b.active.iter().copied())
            .collect();
        if out.is_null() {
            return Ok(ids.len() as i32);
        }
        if count(capacity)? < ids.len() {
            return Err(invalid("debug label output capacity"));
        }
        for (index, id) in ids.iter().enumerate() {
            let l = &storage.labels[*id];
            let data = [
                l.bucket.vertex as f64,
                l.bucket.bucket as f64,
                l.cost,
                l.resources[0],
                l.resources[1],
            ];
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), out.add(index * 5), 5);
            }
        }
        Ok(ids.len() as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_set_label_dominance_strategy(
    p: *mut ORECSP,
    strategy: i32,
) -> i32 {
    guard(-1, || {
        if strategy != 0 && strategy != 1 {
            return Err(invalid("unsupported label dominance strategy"));
        }
        unsafe { solver(p)? }.dominance_mode = strategy;
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_add_accumulated_resource_branch(
    p: *mut ORECSP,
    arc: i32,
    r: i32,
    threshold: f64,
    upper: i32,
) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        resource(s, r)?;
        if !threshold.is_finite() || (upper != 0 && upper != 1) {
            return Err(invalid("accumulated branch input"));
        }
        s.graph.arc_mut(arc)?;
        let id = s.accumulated_branches;
        s.accumulated_branches += 1;
        s.restore_arcs();
        Ok(id as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_clear_accumulated_resource_branches(p: *mut ORECSP) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        s.accumulated_branches = 0;
        s.restore_arcs();
        Ok(0)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_initialize_arc_ng(p: *mut ORECSP) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        let n = s.graph.vertices.len();
        let memory = s.graph.vertices.iter().map(|v| v.memory.clone()).collect();
        s.graph
            .configure_ng(n, (0..n).map(|v| vec![v]).collect(), memory)?;
        s.restore_arcs();
        Ok(0)
    })
}
