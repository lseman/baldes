//! Full-universe arc-flow propagation and Ryan–Foster consequences.
use super::*;
use crate::master::preprocessing::{propagate, Bounds, Constraint, Report};

impl<const D: usize> RootProblem<D> {
    pub(super) fn preprocess(&self, additional: &[Constraint]) -> Result<(Report, Vec<Bounds>)> {
        if self.arc_entries.len() != self.graph.arcs.len() {
            return Err(invalid("preprocessing arc dimensions"));
        }
        let mut rows: Vec<_> = self
            .rows
            .iter()
            .map(|r| Constraint {
                entries: vec![],
                sense: r.sense,
                rhs: r.rhs,
            })
            .collect();
        for (a, entries) in self.arc_entries.iter().enumerate() {
            for &(r, coefficient) in entries {
                rows.get_mut(r)
                    .ok_or_else(|| invalid("preprocessing row index"))?
                    .entries
                    .push((a, coefficient));
            }
        }
        // Every source-to-sink path enters the sink once, so a constant
        // per-route coefficient is a coefficient on every sink-entering arc.
        for &(r, coefficient) in &self.route_entries {
            let row = rows
                .get_mut(r)
                .ok_or_else(|| invalid("preprocessing route row index"))?;
            for (a, arc) in self.graph.arcs.iter().enumerate() {
                if arc.head == self.graph.sink {
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
        rows.extend_from_slice(additional);
        // Rewrite pure incoming-flow rows on outgoing flow as well. This is
        // conservation substitution, not an assumption of elementarity: a
        // unit covering row then caps both incoming and outgoing arc sums.
        let mut outgoing_rows = vec![];
        for row in &rows {
            let Some(&(first, coefficient)) = row.entries.first() else {
                continue;
            };
            let vertex = self.graph.arcs[first].head;
            if vertex == self.graph.source || vertex == self.graph.sink {
                continue;
            }
            let incoming: Vec<_> = self
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
                    entries: self
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
        for v in 0..self.graph.vertices.len() {
            if v == self.graph.source || v == self.graph.sink {
                continue;
            }
            let mut entries = vec![];
            for (a, arc) in self.graph.arcs.iter().enumerate() {
                if arc.head == v {
                    entries.push((a, 1.));
                }
                if arc.tail == v {
                    entries.push((a, -1.));
                }
            }
            rows.push(Constraint {
                entries,
                sense: RowSense::Equal,
                rhs: 0.,
            });
        }
        let mut bounds = vec![
            Bounds {
                lower: 0.,
                upper: f64::INFINITY
            };
            self.graph.arcs.len()
        ];
        // A bounded worklist cannot delay a node indefinitely on cyclic,
        // asymptotically convergent continuous constraints.
        let work = rows.len().saturating_mul(100).max(1000);
        let fixed = self.propagate_ryan_foster(&mut bounds);
        let mut report = propagate(&mut bounds, &rows, 1e-8, work)?;
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
        let mut parent: Vec<_> = (0..self.graph.vertices.len()).collect();
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
        for (arc, bound) in self.graph.arcs.iter().zip(bounds) {
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
            status: column_generation::Status::Infeasible,
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
            warm_start: column_generation::WarmStart {
                bound: f64::NEG_INFINITY,
                bound_mode,
                numerical_bound: f64::NEG_INFINITY,
                certified_bound: f64::NEG_INFINITY,
                stabilization: None,
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
        solver_seconds: started.elapsed().as_secs_f64(),
        cuts: 0,
        lp_before_cuts: f64::NAN,
        fixing_calls: 0,
        enumerated_routes: 0,
        preprocessing: report,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/root/preprocessing/tests.rs"]
mod tests;
