//! Resource-knapsack cut separation.
use super::*;

impl RootController {
    pub(super) fn separate_knapsack(
        &self,
        state: &mut State,
        rows: &mut PendingRows,
        tagged: &[(usize, usize)],
        values: &[f64],
    ) -> Result<usize> {
        let graph = state.solver.graph.graph_id.max(0);
        let mut columns: Vec<_> = tagged
            .iter()
            .filter(|&&(column, _)| values[column] > 1e-9)
            .map(|&(column, tag)| KnapsackColumn {
                graph_id: graph,
                resources: state.routes[tag].resources.clone(),
                value: values[column],
            })
            .collect();
        columns.extend(
            self.problem
                .fixed_routes
                .iter()
                .map(|(route, quantity)| KnapsackColumn {
                    graph_id: graph,
                    resources: route.resources.clone(),
                    value: f64::from(*quantity),
                }),
        );
        let Some(separator) = &mut state.cut.knapsack_separator else {
            return Ok(0);
        };
        let indices = std::collections::BTreeMap::from([(graph, state.solver.knapsack_indices())]);
        let cuts = separator.separate_mapped(&columns, &indices)?;
        let count = cuts.len();
        for cut in cuts {
            let entries = tagged
                .iter()
                .filter_map(|&(column, tag)| {
                    let coefficient = state
                        .solver
                        .knapsack_coefficient(&cut, &state.routes[tag].resources);
                    (coefficient != 0).then_some((column, coefficient as f64))
                })
                .collect();
            let rhs = cut.cut.rhs as f64
                - self.problem.fixed_sum(|route| {
                    Ok(state.solver.knapsack_coefficient(&cut, &route.resources) as f64)
                })?;
            let row = rows.push(RowSense::LessEqual, rhs, entries);
            state.cut.knapsack_rows.push(Some(row));
            state.solver.cuts.knapsack_cuts.push(cut);
        }
        Ok(count)
    }
}
