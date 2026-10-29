//! Theta optimality cuts for route recourse.
use super::*;
use crate::master::routing::recourse::ThetaCut;

impl<const D: usize> RootController<D> {
    /// Integral routes, master columns at one or fixed routes, whose anchor
    /// variable is below their excess and whose cut has no active row.
    fn theta_violations(
        &self,
        state: &State<D>,
        values: &[f64],
        tags: &[i32],
    ) -> Result<Vec<(ThetaCut, usize, Route)>> {
        let Some(recourse) = &self.problem.recourse else {
            return Ok(vec![]);
        };
        let mut integral: Vec<&Route> = tags
            .iter()
            .enumerate()
            .filter(|&(column, &tag)| tag >= 0 && values[column] >= 1. - 1e-6)
            .map(|(_, &tag)| &state.routes[tag as usize])
            .collect();
        integral.extend(self.problem.fixed_routes.iter().map(|(r, _)| r));
        let mut violated = vec![];
        for route in integral {
            let customers = State::<D>::customers(route);
            let Some(&anchor) = customers.iter().min() else {
                continue;
            };
            let Some(column) = recourse.columns.get(anchor).copied().flatten() else {
                return Err(invalid("route recourse anchor has no Theta variable"));
            };
            let excess = self.problem.route_excess(route)?;
            if values[column] >= excess - 1e-6 * (1. + excess) {
                continue;
            }
            let mut arcs = route.arcs.clone();
            arcs.sort_unstable();
            arcs.dedup();
            let active = state
                .cut
                .theta_cuts
                .iter()
                .zip(&state.cut.theta_rows)
                .any(|(cut, row)| row.is_some() && cut.arcs == arcs);
            if !active
                && !violated
                    .iter()
                    .any(|(c, _, _): &(ThetaCut, _, _)| c.arcs == arcs)
            {
                let cut = ThetaCut {
                    arcs,
                    anchor,
                    excess,
                };
                violated.push((cut, column, route.clone()));
            }
        }
        Ok(violated)
    }

    /// Whether an integral route still needs its Theta cut.
    pub(super) fn requires_theta(&self, master: &ClpMaster, tags: &[i32]) -> Result<bool> {
        if self.problem.recourse.is_none() || !master.solved() {
            return Ok(false);
        }
        let state = self.state.borrow();
        Ok(!self
            .theta_violations(&state, &master.values(), tags)?
            .is_empty())
    }

    /// Cut off every integral route whose anchor variable is below its
    /// excess. These cuts make an integral master solution pay its exact
    /// cost, so they run before any stopping test and never tail off.
    pub(super) fn separate_theta(
        &self,
        state: &mut State<D>,
        master: &mut ClpMaster,
        tags: &[i32],
    ) -> Result<usize> {
        let values = master.values().to_vec();
        let violated = self.theta_violations(state, &values, tags)?;
        if violated.is_empty() {
            return Ok(0);
        }
        let graph = &self.problem.model.networks[0].graph;
        let mut rows = PendingRows::new(master.rhs().len());
        let mut added = 0;
        for (cut, column, route) in violated {
            let mut entries = vec![(column, 1. / cut.excess)];
            for (index, &tag) in tags.iter().enumerate() {
                if tag < 0 {
                    continue;
                }
                let coefficient = cut.coefficient(&state.routes[tag as usize]);
                if coefficient != 0. {
                    entries.push((index, coefficient));
                }
            }
            let rhs = 1.
                - route.arcs.len() as f64
                - self.problem.fixed_sum(|fixed| Ok(cut.coefficient(fixed)))?;
            let row = rows.push(RowSense::GreaterEqual, rhs, entries);
            // Pricing charges every arc of the cut route with its dual.
            for (index, arc) in graph.arcs.iter().enumerate() {
                if cut.arcs.binary_search(&arc.id).is_ok() {
                    state.cut.capacity_arc_entries[index].push((row, -1.));
                }
            }
            state.cut.theta_rows.push(Some(row));
            state.cut.theta_cuts.push(cut);
            added += 1;
        }
        rows.install(master)?;
        Ok(added)
    }
}
