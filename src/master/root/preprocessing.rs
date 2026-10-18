//! Full-universe arc-flow propagation and Ryan–Foster consequences.
use super::*;
use crate::master::preprocessing::{propagate, Bounds, Constraint, Report};

impl RootProblem {
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
            if matches!(branch, BranchConstraint::RyanFoster(_)) {
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
    pub(super) fn preprocessing_rows(
        &self,
        graph: &Graph,
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

pub(super) fn infeasible_output(report: Report, started: std::time::Instant) -> RootOutput {
    RootOutput {
        fixed_routes: vec![],
        fixed_cost: 0.,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::branching::RyanFosterConstraint;

    fn complete_problem() -> RootProblem {
        let mut graph = Graph::new(6, 0).unwrap();
        graph.elementary = true;
        for tail in 0..5 {
            for head in 1..6 {
                if tail != head {
                    graph.add_arc(tail, head, 1.).unwrap();
                }
            }
        }
        RootProblem::set_partitioning(graph, 4.).unwrap()
    }

    fn paths(
        graph: &Graph,
        vertex: usize,
        seen: u64,
        path: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if vertex == graph.sink {
            out.push(path.clone());
            return;
        }
        for (a, arc) in graph.arcs.iter().enumerate() {
            if arc.tail == vertex && seen & (1 << arc.head) == 0 {
                path.push(a);
                paths(graph, arc.head, seen | (1 << arc.head), path, out);
                path.pop();
            }
        }
    }

    #[test]
    fn ryan_foster_arc_deductions_preserve_every_admissible_elementary_path() {
        let base = complete_problem();
        let mut universe = vec![];
        paths(
            &base.graph,
            base.graph.source,
            1,
            &mut vec![],
            &mut universe,
        );
        let pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];
        let mut total_fixed = 0;
        for mut code in 0..3usize.pow(pairs.len() as u32) {
            let mut problem = base.clone();
            let mut rules = vec![];
            for (first, second) in pairs {
                let state = code % 3;
                code /= 3;
                if state > 0 {
                    let rule = RyanFosterConstraint {
                        first,
                        second,
                        together: state == 1,
                    };
                    problem.branch(BranchConstraint::RyanFoster(rule)).unwrap();
                    rules.push(rule);
                }
            }
            let mut bounds = vec![
                Bounds {
                    lower: 0.,
                    upper: f64::INFINITY
                };
                problem.graph.arcs.len()
            ];
            total_fixed += problem.propagate_ryan_foster(&mut bounds);
            for path in &universe {
                let nodes: Vec<_> = path.iter().map(|&a| problem.graph.arcs[a].head).collect();
                if rules
                    .iter()
                    .all(|rule| rule.satisfied_by(nodes.iter().copied()))
                {
                    assert!(
                        path.iter().all(|&a| bounds[a].upper > 0.),
                        "rules={rules:?}, path={path:?}"
                    );
                }
            }
        }
        assert!(total_fixed > 0);
    }

    #[test]
    fn apart_rule_excludes_arcs_between_transitive_together_classes() {
        let mut problem = complete_problem();
        for (first, second, together) in [(1, 2, true), (2, 3, false)] {
            problem
                .branch(BranchConstraint::RyanFoster(RyanFosterConstraint {
                    first,
                    second,
                    together,
                }))
                .unwrap();
        }
        let (report, bounds) = problem.preprocess(&[]).unwrap();
        assert!(!report.infeasible);
        for (a, arc) in problem.graph.arcs.iter().enumerate() {
            if (arc.tail == 1 && arc.head == 3) || (arc.tail == 3 && arc.head == 1) {
                assert_eq!(bounds[a].upper, 0.);
            }
        }
    }

    #[test]
    fn contradictory_optional_component_is_excluded_without_false_infeasibility() {
        let mut problem = complete_problem();
        for vertex in [1, 2] {
            problem.rows[vertex].rhs = 0.;
        }
        for together in [true, false] {
            problem
                .branch(BranchConstraint::RyanFoster(RyanFosterConstraint {
                    first: 1,
                    second: 2,
                    together,
                }))
                .unwrap();
        }
        let (report, bounds) = problem.preprocess(&[]).unwrap();
        assert!(!report.infeasible && report.complete);
        for (arc, bound) in problem.graph.arcs.iter().zip(bounds) {
            if [1, 2].contains(&arc.tail) || [1, 2].contains(&arc.head) {
                assert_eq!(bound.upper, 0.);
            }
        }
    }

    #[test]
    fn inherited_cut_rows_follow_physical_remapping_and_retirement() {
        use crate::vrptw::{Instance, Rounding};
        let instance = Instance::parse("triangle\nVEHICLE\nNUMBER CAPACITY\n3 2\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 0 1 0 100 0\n2 0 1 1 0 100 0\n3 1 1 1 0 100 0\n").unwrap();
        let problem =
            RootProblem::set_partitioning(instance.graph(3, Rounding::None).unwrap(), 3.).unwrap();
        let mut options = RootOptions::default()
            .with_capacity_cuts(CapacityCutConfig::new(vec![0., 1., 1., 1.], 2.));
        options.capture_state = true;
        options.enumerate = false;
        let parent = super::super::solve(problem, options).unwrap();
        let snapshot = parent.snapshot.unwrap();
        let graph = &snapshot.problem.graph;
        let mut master = snapshot.master.try_clone().unwrap();
        let mut state = snapshot.cut.clone();
        let before = state.preprocessing_rows(graph, &master).unwrap();
        assert_eq!(before.len(), 1);
        let old = state.capacity_rows[0].unwrap();
        // A new physical index must use its current RHS and sense, while
        // retaining the cut's arc coefficients after remapping.
        let new = master.add_row(RowSense::GreaterEqual, 17., &[]).unwrap();
        state.capacity_rows[0] = Some(new);
        for entries in &mut state.capacity_arc_entries {
            for (row, _) in entries {
                if *row == old {
                    *row = new;
                }
            }
        }
        let remapped = state.preprocessing_rows(graph, &master).unwrap();
        assert_eq!(remapped[0].sense, RowSense::GreaterEqual);
        assert_eq!(remapped[0].rhs, 17.);
        assert_eq!(remapped[0].entries, before[0].entries);
        state.capacity_rows[0] = None;
        assert!(state.preprocessing_rows(graph, &master).unwrap().is_empty());
        // The cloned changes above must not alter the parent's active pool.
        let unchanged = snapshot
            .cut
            .preprocessing_rows(graph, &snapshot.master)
            .unwrap();
        assert_eq!(unchanged[0].rhs, before[0].rhs);
        assert_eq!(unchanged[0].entries, before[0].entries);
    }
}
