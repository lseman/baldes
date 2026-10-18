//! C ABI options operations.
use super::*;

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

#[repr(C)]
pub struct ORPricingPhase {
    pub check_ng_and_cuts: i32,
    pub allow_jump_arcs: i32,
    pub labels_per_bucket: i64,
}
