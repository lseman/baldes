//! Cached-column coefficient synchronization and row remapping.
use super::*;
use std::collections::HashMap;

/// Reuse the refreshed integer cut incidence of loaded routes. Base/branch
/// coefficients stay in their original representation, including tiny values
/// the LP backend may have dropped. Routing tags are local to this node.
pub(super) fn synchronize_pool(
    pool: &mut [Column],
    loaded: &[Option<Column>],
    route_count: usize,
    base_rows: &[bool],
    mut refresh: impl FnMut(&mut Column) -> Result<()>,
) -> Result<()> {
    let mut by_tag = vec![None; route_count];
    for current in loaded.iter().flatten() {
        if let Some(slot) = usize::try_from(current.tag)
            .ok()
            .and_then(|tag| by_tag.get_mut(tag))
        {
            *slot = Some(current);
        }
    }
    let is_base = |row: usize| base_rows.get(row).copied().unwrap_or(false);
    for column in pool {
        if let Some(current) = usize::try_from(column.tag)
            .ok()
            .and_then(|tag| by_tag.get(tag))
            .and_then(|c| *c)
            .filter(|c| c.identity == column.identity && c.cost == column.cost)
        {
            column.entries.retain(|&(row, _)| is_base(row));
            column.entries.extend(
                current
                    .entries
                    .iter()
                    .copied()
                    .filter(|&(row, value)| !is_base(row) && value != 0.),
            );
            column.entries.sort_by_key(|&(row, _)| row);
        } else {
            refresh(column)?;
        }
    }
    Ok(())
}

pub(super) fn restore_limited_memory_columns<const D: usize>(
    problem: &RootProblem<D>,
    state: &mut State<D>,
    master: &mut ClpMaster,
    loaded: &mut [Option<Column>],
) -> Result<bool> {
    if state.solver.enumeration.is_some() || state.cut.enumerated.is_empty() {
        return Ok(false);
    }
    // Enumeration prices full-memory coefficients. Once ordinary labeling
    // resumes (for example after phase I), loaded routes must use the same
    // limited-memory coefficients as the labeling engine.
    let mut refreshed = vec![];
    for (index, column) in loaded.iter().enumerate() {
        let Some(column) = column else { continue };
        let route = state
            .routes
            .get(usize::try_from(column.tag).map_err(|_| invalid("cached route tag"))?)
            .ok_or_else(|| invalid("cached route tag"))?;
        let mut entries = problem.base_entries(route)?;
        entries.extend(
            state
                .cut
                .cached_entries(problem, &state.solver, route, false)?,
        );
        entries.sort_by_key(|&(row, _)| row);
        refreshed.push((index, entries));
    }
    for (index, entries) in refreshed {
        let column = loaded[index].as_mut().unwrap();
        let old: HashMap<_, _> = column.entries.iter().copied().collect();
        let new: HashMap<_, _> = entries.iter().copied().collect();
        for row in old
            .keys()
            .chain(new.keys())
            .copied()
            .collect::<HashSet<_>>()
        {
            let value = new.get(&row).copied().unwrap_or(0.);
            if old.get(&row).copied().unwrap_or(0.) != value {
                master.set_coefficient(row, index, value)?;
            }
        }
        column.entries = entries;
    }
    state.cut.enumerated.clear();
    Ok(true)
}

impl<const D: usize> RootController<D> {
    pub(super) fn restore_limited_memory_columns(
        &mut self,
        master: &mut ClpMaster,
        loaded: &mut [Option<Column>],
    ) -> Result<bool> {
        restore_limited_memory_columns(&self.problem, &mut self.state.borrow_mut(), master, loaded)
    }

    pub(super) fn refresh_pool(&mut self, pool: &mut [Column]) -> Result<()> {
        let state = self.state.borrow();
        for column in pool {
            let route = state
                .routes
                .get(usize::try_from(column.tag).map_err(|_| invalid("cached route tag"))?)
                .ok_or_else(|| invalid("cached route tag"))?;
            let mut entries = self.problem.base_entries(route)?;
            entries.extend(state.cut_entries(&self.problem, route)?);
            entries.sort_by_key(|&(row, _)| row);
            column.entries = entries;
        }
        Ok(())
    }

    pub(super) fn synchronize_pool_entries(
        &mut self,
        pool: &mut [Column],
        loaded: &[Option<Column>],
    ) -> Result<()> {
        let state = self.state.borrow();
        // Scaled knapsack coefficients need not survive the backend's small
        // element threshold. Reconstruct them from their original scales.
        if state.cut.knapsack_rows.iter().any(Option::is_some) {
            drop(state);
            return self.refresh_generated_columns(pool);
        }
        let route_count = state.routes.len();
        drop(state);
        let map = self.problem.row_map.borrow();
        let mut base_rows = vec![false; map.iter().max().map_or(0, |r| r + 1)];
        for &row in map.iter() {
            base_rows[row] = true;
        }
        drop(map);
        synchronize_pool(pool, loaded, route_count, &base_rows, |column| {
            self.refresh_generated_columns(std::slice::from_mut(column))
        })
    }

    pub(super) fn remap_pool_rows(&mut self, rows: &[Option<usize>]) -> Result<()> {
        for row in self.problem.row_map.borrow_mut().iter_mut() {
            *row = rows[*row].ok_or_else(|| invalid("attempt to remove a base or branch row"))?;
        }
        let mut state = self.state.borrow_mut();
        for index in 0..state.cut.knapsack_rows.len() {
            if let Some(row) = state.cut.knapsack_rows[index] {
                state.cut.knapsack_rows[index] = rows[row];
                if rows[row].is_none() {
                    state.solver.cuts.knapsack_cuts[index].dual = 0.;
                }
            }
        }
        for row in &mut state.cut.capacity_rows {
            *row = row.and_then(|r| rows[r]);
        }
        for entries in &mut state.cut.capacity_arc_entries {
            *entries = entries
                .iter()
                .filter_map(|&(r, a)| rows[r].map(|row| (row, a)))
                .collect();
        }
        for index in 0..state.cut.rows.len() {
            let Some(row) = state.cut.rows[index] else {
                continue;
            };
            state.cut.rows[index] = rows[row];
            if rows[row].is_none() {
                state.solver.cuts.retire(index);
            }
        }
        Ok(())
    }
}
