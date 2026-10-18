//! C ABI statistics operations.
use super::*;

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
            let l = storage.labels.view(*id);
            let data = [
                l.bucket().vertex as f64,
                l.bucket().bucket as f64,
                l.cost(),
                l.resources()[0],
                l.resources()[1],
            ];
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), out.add(index * 5), 5);
            }
        }
        Ok(ids.len() as i32)
    })
}

#[no_mangle]
pub unsafe extern "C" fn or_rcsp_get_cut_count(p: *const ORECSP) -> i32 {
    guard(-1, || {
        unsafe { solver(p.cast_mut())? };
        Ok(0)
    })
}
