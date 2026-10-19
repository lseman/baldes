//! Capacity and complement cut separation.
use super::*;

impl RootController {
    pub(super) fn separate_capacity(
        &self,
        state: &mut State,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        fractional: &[FractionalColumn],
    ) -> Result<usize> {
        let Some(separator) = state.cut.capacity_separator.as_mut() else {
            return Ok(0);
        };
        let columns: Vec<_> = fractional
            .iter()
            .map(|c| (c.vertices.clone(), c.value))
            .collect();
        let mut added = 0;
        for cut in separator.separate(&columns)? {
            let entries: Vec<_> = tagged
                .iter()
                .filter_map(|&(column, tag)| {
                    let coefficient = cut.coefficient(&State::customers(&state.routes[tag]));
                    (coefficient != 0).then_some((column, coefficient as f64))
                })
                .collect();
            let sense = if cut.greater_equal() {
                RowSense::GreaterEqual
            } else {
                RowSense::LessEqual
            };
            let rhs = cut.rhs as f64
                - self
                    .problem
                    .fixed_sum(|route| Ok(cut.coefficient(&State::customers(route)) as f64))?;
            let row = rows.push(sense, rhs, entries);
            for (index, arc) in self.problem.graph.arcs.iter().enumerate() {
                let u = if arc.tail == self.problem.graph.sink {
                    0
                } else {
                    arc.tail
                };
                let v = if arc.head == self.problem.graph.sink {
                    0
                } else {
                    arc.head
                };
                let coefficient = cut.edge_coefficient(u, v);
                if coefficient != 0. {
                    state.cut.capacity_arc_entries[index].push((row, coefficient));
                }
            }
            state.cut.capacity_rows.push(Some(row));
            state.cut.capacity_cuts.push(cut);
            added += 1;
        }
        Ok(added)
    }
}
