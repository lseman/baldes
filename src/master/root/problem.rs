//! Routing master rows, route coefficients, and reduced costs.
use super::*;
use crate::labeling::extend_resources;
#[derive(Clone, Debug)]
pub struct Row {
    pub sense: RowSense,
    pub rhs: f64,
}
#[derive(Clone, Debug)]
pub struct RootProblem<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub graph: Graph<D>,
    pub rows: Vec<Row>,
    pub arc_entries: Vec<Vec<(usize, f64)>>,
    pub route_entries: Vec<(usize, f64)>,
    /// Vertices with a unit covering row: the rank-one cut customers.
    pub customers: Vec<usize>,
    pub(super) fixed_routes: Vec<(Route, u32)>,
    pub(super) branch_constraints: Vec<(usize, BranchConstraint)>,
    /// Arc id -> first index in `graph.arcs`.
    pub(super) arc_index: std::collections::HashMap<i32, usize>,
    /// Logical base/branch row -> physical master row. Branches appended
    /// after inherited cuts stay valid when those cuts are retired.
    pub(super) row_map: RefCell<Vec<usize>>,
}
impl<const D: usize> RootProblem<D> {
    /// Cover each nonterminal vertex and bound the number of vehicle routes.
    /// Arc entries charge the head vertex, retaining repeated-visit coefficients.
    pub fn set_partitioning(graph: Graph<D>, vehicles: f64) -> Result<Self> {
        if !vehicles.is_finite() || vehicles <= 0. {
            return Err(invalid("positive finite vehicle multiplicity required"));
        }
        let mut rows = vec![Row {
            sense: RowSense::LessEqual,
            rhs: vehicles,
        }];
        let mut customer_row = vec![None; graph.vertices.len()];
        let mut customers = vec![];
        for v in 0..graph.vertices.len() {
            if v != graph.source && v != graph.sink {
                customer_row[v] = Some(rows.len());
                customers.push(v);
                rows.push(Row {
                    sense: RowSense::Equal,
                    rhs: 1.,
                });
            }
        }
        let fleet = 0;
        let arc_entries = graph
            .arcs
            .iter()
            .map(|a| {
                if a.head == graph.sink {
                    vec![(fleet, 1.)]
                } else {
                    customer_row[a.head].map_or_else(Vec::new, |r| vec![(r, 1.)])
                }
            })
            .collect();
        let mut arc_index = std::collections::HashMap::with_capacity(graph.arcs.len());
        for (index, a) in graph.arcs.iter().enumerate() {
            arc_index.entry(a.id).or_insert(index);
        }
        let row_map = RefCell::new((0..rows.len()).collect());
        Ok(Self {
            graph,
            rows,
            arc_entries,
            route_entries: vec![],
            customers,
            fixed_routes: vec![],
            branch_constraints: vec![],
            arc_index,
            row_map,
        })
    }
    pub fn arc_position(&self, id: i32) -> Option<usize> {
        self.arc_index.get(&id).copied()
    }
    /// Add a branch to a fresh problem before constructing its master.
    pub fn branch(&mut self, constraint: BranchConstraint) -> Result<()> {
        constraint.validate(self)?;
        if self.tighten_vehicle_upper(&constraint)? {
            return Ok(());
        }
        self.append_constraint(constraint, self.rows.len())
    }
    /// A matching vehicle upper bound is a bound change on the fleet row.
    /// Keep its branch history without adding another copy of its coefficients.
    pub(super) fn tighten_vehicle_upper(&mut self, constraint: &BranchConstraint) -> Result<bool> {
        let BranchConstraint::NumberOfVehicles {
            sense: RowSense::LessEqual,
            rhs,
            ..
        } = constraint
        else {
            return Ok(false);
        };
        if self
            .rows
            .first()
            .map_or(true, |row| row.sense != RowSense::LessEqual)
            || self
                .route_entries
                .iter()
                .any(|&(row, value)| row == 0 && value != 0.)
            || self.arc_entries.iter().enumerate().any(|(arc, entries)| {
                entries
                    .iter()
                    .filter(|&&(row, _)| row == 0)
                    .map(|&(_, value)| value)
                    .sum::<f64>()
                    != constraint.arc_coefficient(self, arc)
            })
        {
            return Ok(false);
        }
        let residual = rhs - self.fixed_sum(|route| Ok(constraint.coefficient(self, route)))?;
        self.rows[0].rhs = self.rows[0].rhs.min(residual);
        self.branch_constraints.push((0, constraint.clone()));
        Ok(true)
    }
    /// The node's branch constraints, in insertion order.
    pub(crate) fn branches(&self) -> impl Iterator<Item = &BranchConstraint> {
        self.branch_constraints.iter().map(|(_, branch)| branch)
    }
    pub(crate) fn append_constraint(
        &mut self,
        constraint: BranchConstraint,
        physical: usize,
    ) -> Result<()> {
        constraint.validate(self)?;
        // An empty intersection forbids this optional arc, rather than the
        // whole node. Express it as an additive zero-flow row for propagation.
        let constraint = if let BranchConstraint::AccumulatedResourceGroup(c) = constraint {
            BranchConstraint::AccumulatedResourceGroup(c.prepare(&self.graph)?)
        } else if let BranchConstraint::AccumulatedResource(c) = &constraint {
            if c.window(&self.graph)?.is_none() {
                BranchConstraint::ArcFlow {
                    arc: self.arc_position(c.arc).unwrap(),
                    sense: RowSense::LessEqual,
                    rhs: 0.,
                }
            } else {
                constraint
            }
        } else {
            constraint
        };
        let logical = self.rows.len();
        let (sense, mut rhs) = constraint.row();
        rhs -= self.fixed_sum(|route| Ok(constraint.coefficient(self, route)))?;
        let in_pricing = crate::master::branching::enforcement::installs_in_pricing(&constraint);
        if let (true, BranchConstraint::AccumulatedResource(c)) = (in_pricing, &constraint) {
            let window = c.window(&self.graph)?.expect("nonempty resource branch");
            self.graph.arc_mut(c.arc)?.bounds[c.resource] = Some(window);
        } else if let (true, BranchConstraint::AccumulatedResourceGroup(c)) =
            (in_pricing, &constraint)
        {
            c.install(&mut self.graph)?;
        }

        for a in 0..self.graph.arcs.len() {
            let value = constraint.arc_coefficient(self, a);
            if value != 0. {
                self.arc_entries[a].push((logical, value));
            }
        }
        self.rows.push(Row { sense, rhs });
        self.row_map.borrow_mut().push(physical);
        self.branch_constraints.push((logical, constraint));
        Ok(())
    }
    pub(crate) fn base_entries(&self, route: &Route) -> Result<Vec<(usize, f64)>> {
        let map = self.row_map.borrow();
        let mut coefficients = std::collections::BTreeMap::new();
        for &(r, a) in &self.route_entries {
            *coefficients.entry(map[r]).or_insert(0.) += a;
        }
        for id in &route.arcs {
            let physical = self
                .arc_position(*id)
                .ok_or_else(|| invalid("unknown priced route arc"))?;
            for &(r, a) in &self.arc_entries[physical] {
                *coefficients.entry(map[r]).or_insert(0.) += a;
            }
        }
        for (r, c) in &self.branch_constraints {
            if c.nonlinear() {
                let value = c.nonlinear_coefficient(self, route);
                if value != 0. {
                    *coefficients.entry(map[*r]).or_insert(0.) += value;
                }
            }
        }
        Ok(coefficients.into_iter().filter(|(_, a)| *a != 0.).collect())
    }
    pub(super) fn route_cost(&self, route: &Route) -> (f64, Vec<i32>) {
        let mut identities = route.arcs.clone();
        identities.sort_unstable();
        let mut cost = 0.;
        let mut counts = std::collections::BTreeMap::new();
        for &id in &identities {
            *counts.entry(id).or_insert(0usize) += 1;
        }
        for (id, count) in counts {
            cost += self.graph.arcs[self.arc_position(id).unwrap()].cost * count as f64;
        }
        if self.graph.has_coupled_resources() {
            let positions: Vec<_> = route
                .arcs
                .iter()
                .map(|id| self.arc_position(*id).unwrap())
                .collect();
            cost += self
                .graph
                .coupled_path_cost(&positions)
                .ok()
                .flatten()
                .unwrap_or(f64::INFINITY);
        }
        (cost, identities)
    }
    pub(super) fn reprice(
        &self,
        solver: &mut Solver<D>,
        duals: &[f64],
        round: bool,
        capacity_entries: &[Vec<(usize, f64)>],
    ) -> Result<f64> {
        let map = self.row_map.borrow();
        if duals.len() < self.rows.len() || self.arc_entries.len() != self.graph.arcs.len() {
            return Err(invalid("root master dimensions"));
        }
        for (index, arc) in solver.graph.arcs.iter_mut().enumerate() {
            let mut cost = self.graph.arcs[index].cost;
            for &(r, a) in &self.arc_entries[index] {
                cost -= duals[map[r]] * a;
            }
            for &(r, a) in &capacity_entries[index] {
                cost -= duals[r] * a;
            }
            arc.reduced_cost = if round {
                (cost * 1e8 + 0.5).floor() / 1e8
            } else {
                cost
            };
        }
        Ok(-self
            .route_entries
            .iter()
            .map(|&(r, a)| duals[map[r]] * a)
            .sum::<f64>())
    }
}

impl<const D: usize> RootProblem<D> {
    pub(crate) fn elementary_route(&self, ids: &[i32]) -> Option<Route> {
        let g = &self.graph;
        let mut at = g.source;
        let mut nodes = vec![at as i32];
        let mut visited = HashSet::new();
        let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
        for (r, value) in q.iter_mut().enumerate().take(g.resources.len()) {
            *value = g.root_resource(r, true)?;
            if *value < g.vertices[at].bounds[r].lower || *value > g.vertices[at].bounds[r].upper {
                return None;
            }
        }
        let mut resource_profile = vec![q[..g.resources.len()].to_vec()];
        let mut cost = 0.;
        for &id in ids {
            let arc = &g.arcs[self.arc_position(id)?];
            if arc.tail != at || at == g.sink {
                return None;
            }
            q = extend_resources(g, arc, &q, true)?;
            at = arc.head;
            if at == g.source || (at != g.sink && !visited.insert(at)) {
                return None;
            }
            cost += arc.cost;
            nodes.push(at as i32);
            resource_profile.push(q[..g.resources.len()].to_vec());
        }
        if at != g.sink || visited.is_empty() {
            return None;
        }
        let positions: Vec<_> = ids
            .iter()
            .map(|id| self.arc_position(*id))
            .collect::<Option<_>>()?;
        cost += g.coupled_path_cost(&positions).ok()??;
        Some(Route {
            nodes,
            arcs: ids.to_vec(),
            cost,
            original_cost: cost,
            resources: q[..g.resources.len()].to_vec(),
            resource_profile,
        })
    }
}
