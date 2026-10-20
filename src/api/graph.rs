//! C ABI graph operations.
use super::*;

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
        let r = s.graph.add_resource(k)?;
        s.bucket_steps[r] = 0.;
        s.engine = None;
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
