//! Full-universe arc-flow propagation and Ryan–Foster consequences.
use super::*;
use crate::master::preprocessing::{propagate, Bounds, Constraint, Report};

impl<const D: usize> RoutingContext<D> {
    pub(super) fn preprocess(&self, additional: &[Constraint]) -> Result<(Report, Vec<Bounds>)> {
        if self.model.networks[0].arc_entries.len() != self.model.networks[0].graph.arcs.len() {
            return Err(invalid("preprocessing arc dimensions"));
        }
        let mut rows: Vec<_> = self
            .model
            .rows
            .iter()
            .map(|r| Constraint {
                entries: vec![],
                sense: r.sense,
                rhs: r.rhs,
            })
            .collect();
        for (a, entries) in self.model.networks[0].arc_entries.iter().enumerate() {
            for &(r, coefficient) in entries {
                rows.get_mut(r)
                    .ok_or_else(|| invalid("preprocessing row index"))?
                    .entries
                    .push((a, coefficient));
            }
        }
        // Every source-to-sink path enters the sink once, so a constant
        // per-route coefficient is a coefficient on every sink-entering arc.
        for &(r, coefficient) in &self.model.networks[0].route_entries {
            let row = rows
                .get_mut(r)
                .ok_or_else(|| invalid("preprocessing route row index"))?;
            for (a, arc) in self.model.networks[0].graph.arcs.iter().enumerate() {
                if arc.head == self.model.networks[0].graph.sink {
                    row.entries.push((a, coefficient));
                }
            }
        }
        // Nonlinear Ryan–Foster coefficients cannot be represented as arc
        // activities. Their resource implementation remains authoritative.
        for &(r, ref branch) in &self.branch_constraints {
            if branch.nonlinear() {
                rows[r].entries.clear();
                // A negative residual RHS certifies that the fixed part
                // already violated this nonnegative violation row.
            }
        }
        let graph = &self.model.networks[0].graph;
        let mut owner_coordinates = std::collections::BTreeMap::new();
        let mut owner_limits = vec![];
        let mut owner_states = std::collections::BTreeMap::new();
        let quantity = graph.arcs.len();
        let quantity_bounds = Bounds {
            lower: self.model.networks[0].multiplicity.lower,
            upper: self.model.networks[0].multiplicity.upper,
        };
        if let Some(kernel) = graph.coupled_kernel() {
            let resources: std::collections::BTreeSet<_> = kernel
                .paired_owner_registration_bounds()
                .iter()
                .map(|&(resource, _)| resource)
                .chain(
                    kernel
                        .paired_associated_costs()
                        .iter()
                        .map(|&(resource, _)| resource),
                )
                .chain(
                    self.model.networks[0]
                        .activity_entries
                        .iter()
                        .map(|entry| entry.resource),
                )
                .collect();
            for resource in resources {
                if !matches!(
                    kernel.definitions().get(resource),
                    Some(crate::graph::coupled::CoupledResource::PairedFront { .. })
                ) {
                    continue;
                }
                let input = kernel
                    .paired_owner_registration_bounds()
                    .iter()
                    .find(|&&(index, _)| index == resource)
                    .map(|&(_, bounds)| bounds);
                let mut fixed_upper = ExactSum::default();
                let mut negative_fixed_lower = ExactSum::default();
                for (route, count) in &self.fixed_routes {
                    let arcs = route
                        .arcs
                        .iter()
                        .map(|id| {
                            self.arc_position(*id)
                                .ok_or_else(|| invalid("unknown fixed owner arc"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let value = crate::master::model::accumulated_activity_column_value(
                        graph, route, &arcs, resource,
                    )?;
                    if fixed_upper.product(value, f64::from(*count)).is_none()
                        || negative_fixed_lower
                            .product(-value, f64::from(*count))
                            .is_none()
                    {
                        return Err(invalid("fixed owner activity overflow"));
                    }
                }
                let fixed = [
                    -negative_fixed_lower
                        .upper()
                        .ok_or_else(|| invalid("fixed owner activity overflow"))?,
                    fixed_upper
                        .upper()
                        .ok_or_else(|| invalid("fixed owner activity overflow"))?,
                ];
                let owner = crate::master::preprocessing::paired_owner::OwnerBounds::new(
                    input,
                    quantity_bounds,
                    fixed,
                );
                let index = quantity + 1 + owner_limits.len();
                owner_coordinates.insert(resource, index);
                owner_limits.push(owner.aggregate);
                owner_states.insert(resource, owner);
            }
        }
        for entry in &self.model.networks[0].activity_entries {
            let Some(&index) = owner_coordinates.get(&entry.resource) else {
                return Err(invalid(
                    "routing owner preprocessing requires represented paired activities",
                ));
            };
            rows[entry.row].entries.push((index, entry.coefficient));
        }
        rows.extend_from_slice(additional);
        // Rewrite pure incoming-flow rows on outgoing flow as well. This is
        // conservation substitution, not an assumption of elementarity: a
        // unit covering row then caps both incoming and outgoing arc sums.
        let mut outgoing_rows = vec![];
        for row in &rows {
            let Some(&(first, coefficient)) = row.entries.first() else {
                continue;
            };
            let Some(first_arc) = self.model.networks[0].graph.arcs.get(first) else {
                continue;
            };
            let vertex = first_arc.head;
            if vertex == self.model.networks[0].graph.source
                || vertex == self.model.networks[0].graph.sink
            {
                continue;
            }
            let incoming: Vec<_> = self.model.networks[0]
                .graph
                .arcs
                .iter()
                .enumerate()
                .filter_map(|(a, arc)| (arc.head == vertex).then_some(a))
                .collect();
            if row.entries.len() == incoming.len()
                && incoming.iter().all(|&a| {
                    row.entries
                        .iter()
                        .filter(|&&(i, c)| i == a && c == coefficient)
                        .count()
                        == 1
                })
            {
                outgoing_rows.push(Constraint {
                    entries: self.model.networks[0]
                        .graph
                        .arcs
                        .iter()
                        .enumerate()
                        .filter_map(|(a, arc)| (arc.tail == vertex).then_some((a, coefficient)))
                        .collect(),
                    sense: row.sense,
                    rhs: row.rhs,
                });
            }
        }
        rows.extend(outgoing_rows);
        rows.extend(crate::master::preprocessing::flow::conservation(
            &self.model.networks[0].graph,
            0,
            (!owner_states.is_empty()).then_some(quantity),
        ));
        let mut bounds = vec![
            Bounds {
                lower: 0.,
                upper: f64::INFINITY
            };
            self.model.networks[0].graph.arcs.len()
        ];
        // A bounded worklist cannot delay a node indefinitely on cyclic,
        // asymptotically convergent continuous constraints.
        let work = rows.len().saturating_mul(100).max(1000);
        let fixed = self.propagate_ryan_foster(&mut bounds);
        if !owner_states.is_empty() {
            bounds.push(quantity_bounds);
        }
        bounds.extend(owner_limits);
        let mut report = Report::default();
        loop {
            if report.work_used >= work {
                break;
            }
            let pass = propagate(&mut bounds, &rows, 1e-8, work - report.work_used)?;
            report.work_used += pass.work_used;
            report.rows_processed += pass.rows_processed;
            report.bound_updates += pass.bound_updates;
            report.infeasible |= pass.infeasible;
            report.complete = pass.complete;
            if report.infeasible || !pass.complete {
                break;
            }
            let mut changed = false;
            for (&resource, owner) in &mut owner_states {
                let index = owner_coordinates[&resource];
                changed |= owner.update(bounds[index], bounds[quantity]);
                bounds[index] = owner.aggregate;
            }
            if !changed {
                break;
            }
        }
        if !report.infeasible && (!owner_states.is_empty() && bounds[quantity].upper > 0.) {
            for &resource in owner_coordinates.keys() {
                let current = owner_states[&resource].local;
                if crate::graph::coupled::CoupledKernel::owner_registration_bound_is_unsupported([
                    current.lower,
                    current.upper,
                ]) {
                    return Err(invalid(format!("non-zero lower bounds on variables associated to arcs are not supported (paired resource {resource}, propagated owner bounds [{}, {}])",current.lower,current.upper)));
                }
            }
        }
        bounds.truncate(graph.arcs.len());
        report.bound_updates += fixed;
        Ok((report, bounds))
    }

    /// Together rules form equivalence classes of route membership. Apart
    /// rules exclude routes meeting both classes. An internally inconsistent
    /// class must be absent; covering/assignment rows decide whether that makes
    /// the node infeasible, so optional vertices are not assumed mandatory.
    fn propagate_ryan_foster(&self, bounds: &mut [Bounds]) -> usize {
        fn find(parent: &mut [usize], mut vertex: usize) -> usize {
            while parent[vertex] != vertex {
                parent[vertex] = parent[parent[vertex]];
                vertex = parent[vertex];
            }
            vertex
        }
        let mut parent: Vec<_> = (0..self.model.networks[0].graph.vertices.len()).collect();
        let mut size = vec![1; parent.len()];
        for (_, branch) in &self.branch_constraints {
            if let BranchConstraint::RyanFoster(rule) = branch {
                if rule.together {
                    let mut a = find(&mut parent, rule.first);
                    let mut b = find(&mut parent, rule.second);
                    if a != b {
                        if size[a] < size[b] {
                            std::mem::swap(&mut a, &mut b);
                        }
                        parent[b] = a;
                        size[a] += size[b];
                    }
                }
            }
        }
        let groups: Vec<_> = (0..parent.len()).map(|v| find(&mut parent, v)).collect();
        let mut absent = vec![false; parent.len()];
        let mut apart = std::collections::BTreeSet::new();
        for (_, branch) in &self.branch_constraints {
            if let BranchConstraint::RyanFoster(rule) = branch {
                if !rule.together {
                    let (a, b) = (groups[rule.first], groups[rule.second]);
                    if a == b {
                        absent[a] = true;
                    } else {
                        apart.insert((a.min(b), a.max(b)));
                    }
                }
            }
        }
        let mut fixed = 0;
        for (arc, bound) in self.model.networks[0].graph.arcs.iter().zip(bounds) {
            let (a, b) = (groups[arc.tail], groups[arc.head]);
            if bound.upper > 0. && (absent[a] || absent[b] || apart.contains(&(a.min(b), a.max(b))))
            {
                bound.upper = 0.;
                fixed += 1;
            }
        }
        fixed
    }
}

impl CutState {
    /// Only cuts with a full-universe additive arc representation are eligible.
    /// Row identities and RHS come from the current physical master, after any
    /// row retirement/remapping. No restricted-column activities are consulted.
    pub(super) fn preprocessing_rows<const D: usize>(
        &self,
        graph: &Graph<D>,
        master: &ClpMaster,
    ) -> Result<Vec<Constraint>> {
        if self.capacity_cuts.len() != self.capacity_rows.len() {
            return Err(invalid("preprocessing capacity row dimensions"));
        }
        if self.capacity_arc_entries.len() != graph.arcs.len() {
            return Err(invalid("preprocessing capacity arc dimensions"));
        }
        let mut rows = vec![];
        let mut indices = std::collections::BTreeMap::new();
        for row in self.capacity_rows.iter().flatten().copied() {
            let sense = *master
                .senses
                .get(row)
                .ok_or_else(|| invalid("preprocessing cut row index"))?;
            let rhs = *master
                .rhs
                .get(row)
                .ok_or_else(|| invalid("preprocessing cut RHS index"))?;
            indices.insert(row, rows.len());
            rows.push(Constraint {
                entries: vec![],
                sense,
                rhs,
            });
        }
        for (arc, entries) in self.capacity_arc_entries.iter().enumerate() {
            for &(row, coefficient) in entries {
                if let Some(&index) = indices.get(&row) {
                    rows[index].entries.push((arc, coefficient));
                }
            }
        }
        Ok(rows)
    }
}

pub(super) fn infeasible_output<const D: usize>(
    report: Report,
    started: std::time::Instant,
    bound_mode: BoundMode,
) -> RootOutput<D> {
    RootOutput {
        fixed_routes: vec![],
        fixed_cost: 0.,
        snapshot: None,
        simplex_iterations: 0,
        ipm_iterations: 0,
        inherited_cuts: 0,
        master: Output {
            status: cg::Status::Infeasible,
            feasibility: vec![],
            enumeration_requested: false,
            enumeration_failure: None,
            lp: f64::INFINITY,
            bound: f64::INFINITY,
            bound_mode,
            numerical_bound: f64::INFINITY,
            certified_bound: f64::INFINITY,
            iterations: 0,
            pricing_calls: 0,
            columns: 0,
            history: vec![],
            column_tags: vec![],
            inserted_tags: vec![],
            artificials: vec![],
            penalty_updates: 0,
            pruned_by_bound: false,
            cut_rounds: 0,
            warm_start: cg::WarmStart {
                bound: f64::NEG_INFINITY,
                bound_mode,
                numerical_bound: f64::NEG_INFINITY,
                certified_bound: f64::NEG_INFINITY,
                stabilization: None,
                proximal: None,
                generated_pool: vec![],
                columns: vec![],
                artificials: vec![],
                cut_artificials: vec![],
                initial_rows: 0,
            },
        },
        routes: vec![],
        values: vec![],
        duals: vec![],
        incumbent: f64::INFINITY,
        pricing_seconds: 0.,
        oracle_calls: 0,
        solver_seconds: started.elapsed().as_secs_f64(),
        cuts: 0,
        lp_before_cuts: f64::NAN,
        fixing_calls: 0,
        enumerated_routes: 0,
        preprocessing: report,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/routing/preprocessing/tests.rs"]
mod tests;

#[cfg(all(test, feature = "multi-time-windows"))]
#[path = "../../../tests/unit/master/routing/preprocessing/window_tests.rs"]
mod window_tests;
