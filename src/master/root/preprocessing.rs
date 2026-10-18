//! Full-universe preprocessing on original aggregate arc flows.
use super::*;
use crate::master::preprocessing::{propagate, Bounds, Constraint, Report};

impl RootProblem {
    pub(super) fn preprocess(&self) -> Result<(Report, Vec<Bounds>)> {
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
            if matches!(branch, BranchConstraint::RyanFoster(_)) {
                rows[r].entries.clear();
                rows[r].rhs = 0.;
            }
        }
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
        let report = propagate(&mut bounds, &rows, 1e-8, work)?;
        Ok((report, bounds))
    }
}

pub(super) fn infeasible_output(report: Report, started: std::time::Instant) -> RootOutput {
    RootOutput {
        snapshot: None,
        simplex_iterations: 0,
        inherited_cuts: 0,
        master: Output {
            status: column_generation::Status::Infeasible,
            lp: f64::INFINITY,
            bound: f64::INFINITY,
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
