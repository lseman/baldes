//! C ABI relaxation operations.
use super::*;

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
            mode: s.options.get("BALDESdynamicNGmode")?.abs() as i32,
            max_vertex_size: s.options.get("BALDESmaxNGneighbourhoodSize")?.max(0.) as usize,
            max_average_size: s.options.get("BALDESmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
            max_columns: s.options.get("BALDESmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
            max_cycle_size: s.options.get("BALDESmaxCycleSizeInDynamicNG")? as i32,
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
        Ok(s.route_satisfies_relaxation(vertices, arcs)? as i32)
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
        let size = s.options.get("BALDESinitNGneighbourhoodSize")?.max(0.) as usize;
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
        let size = s.options.get("BALDESinitNGneighbourhoodSize")?.max(0.) as usize;
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
        if s.options.get("BALDESdynamicNGmode")?.abs() != 1. {
            return Ok(1);
        }
        let prepared = crate::graph::ng::PreparedNg::new(&s.graph)?;
        let max = s.options.get("BALDESmaxNGneighbourhoodSize")?.max(0.) as usize;
        let avg = s.options.get("BALDESmaxNGaverNeighbourhoodSize")?.max(0.) as usize;
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

            for (axis, &slot) in engine.grid.resource_slots.iter().enumerate() {
                s.bucket_steps[slot] = engine.grid.step[axis];
            }
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
        let id = s.add_accumulated_resource_branch(
            crate::pricing::restrictions::AccumulatedResourceBranch {
                arc,
                resource: r as usize,
                threshold,
                upper: upper != 0,
            },
        )?;
        Ok(id as i32)
    })
}
#[no_mangle]
pub unsafe extern "C" fn or_rcsp_clear_accumulated_resource_branches(p: *mut ORECSP) -> i32 {
    guard(-1, || {
        let s = unsafe { solver(p)? };
        s.clear_accumulated_resource_branches()?;
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

#[repr(C)]
pub struct ORNGColumn {
    pub arc_ids: *const i32,
    pub count: i32,
    pub value: f64,
}
