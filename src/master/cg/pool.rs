//! Optional reuse of previously generated columns before another oracle call.
use super::controller::apply_column_cap;
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn restore<M: LpMaster, C: Controller<M>>(
    master: &mut M,
    controller: &mut C,
    duals: &[f64],
    pool: &[Column],
    loaded: &mut Vec<Option<Column>>,
    seen: &mut HashSet<ColumnKey>,
    out: &mut Output,
    tolerance: f64,
    deadline: Option<Instant>,
) -> Result<Option<f64>> {
    if !controller.uses_inactive_column_pool() {
        return Ok(None);
    }
    let first = loaded.len();
    let mut minimum: Option<f64> = None;
    let mut pending = Vec::new();
    let mut pending_keys = HashSet::new();
    for cached in pool {
        if deadline.is_some_and(|end| Instant::now() >= end) {
            return Ok(None);
        }
        if !cached.acceptable || master_reduced_cost(cached, duals) >= -tolerance {
            continue;
        }
        let key = cached.signature();
        if seen.contains(&key) || pending_keys.contains(&key) {
            continue;
        }
        if controller.column_upper_bound(cached.tag) == Some(0.) {
            continue;
        }
        let Some(mut column) = controller.reprice_inactive_column(cached, duals)? else {
            continue;
        };
        if !column.acceptable || column.tag != cached.tag || column.signature() != key {
            continue;
        }
        if !column.cost.is_finite()
            || column
                .entries
                .iter()
                .any(|&(row, a)| row >= duals.len() || !a.is_finite())
        {
            return Err(invalid("invalid inactive column"));
        }
        let reduced = master_reduced_cost(&column, duals);
        if !reduced.is_finite() {
            return Err(invalid("inactive reduced cost overflow"));
        }
        if reduced >= -tolerance {
            continue;
        }
        column.reduced_cost = reduced;
        minimum = Some(minimum.map_or(reduced, |previous| previous.min(reduced)));
        pending_keys.insert(key);
        pending.push(column);
    }
    if pending.is_empty() {
        return Ok(None);
    }
    let batch: Vec<_> = pending
        .iter()
        .map(|c| (c.cost, c.entries.as_slice()))
        .collect();
    for (index, column) in master.add_columns(&batch)?.zip(&pending) {
        apply_column_cap(master, controller, index, column.tag)?;
    }
    for column in pending {
        out.column_tags.push(column.tag);
        out.inserted_tags.push(column.tag);
        loaded.push(Some(column));
    }
    debug_assert!(loaded.len() > first);
    seen.extend(pending_keys);
    out.columns = master.costs().len();
    Ok(minimum)
}
