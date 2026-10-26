//! Optional simultaneous service. Pricing remains route-local; the master owns
//! coverage and finite integer start domains shared by all participating fleets.
use super::*;
use crate::{graph::ResourceKind, master::preprocessing::Bounds};
use serde::{Deserialize, Serialize};
mod witness;
pub(super) use witness::witness;

/// One occurrence of a synchronized job in a pricing network. Multiplicity
/// represents distinct interchangeable vehicles, so two copies of a route may
/// supply two vehicles. Each physical vertex belongs to at most one service.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SynchronizedServiceMember {
    pub network: i32,
    pub vertex: usize,
}

/// All selected copies start at one integer resource value in `start` (inclusive).
/// Adds a binary served variable, a single reward, and the equality
/// `sum(visits) = vehicles_required * served`. Existing route costs and shared
/// rows are retained; do not also charge this reward on each incoming arc.
/// Time is an independent disposable scalar axis with nonnegative
/// consumption. Separate coupled-resource domains may coexist on the routes.
/// Resource-load rows on this axis are replayed after start-domain changes,
/// including fixed-route RHS updates.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SynchronizedService {
    pub id: i32,
    pub resource: usize,
    pub vehicles_required: u32,
    pub reward: f64,
    pub start: [i64; 2],
    pub members: Vec<SynchronizedServiceMember>,
}

#[derive(Clone)]
pub(super) struct Service {
    pub definition: SynchronizedService,
    pub domain: [i64; 2],
    variable: i32,
}

impl Service {
    /// Master variable of the binary served decision.
    pub(in crate::master::mp) fn variable(&self) -> i32 {
        self.variable
    }
}

pub(super) fn prepare<const D: usize>(
    problem: &mut Problem<D>,
    variables: &mut Vec<MasterVariable>,
    definitions: Vec<SynchronizedService>,
) -> Result<Vec<Service>> {
    let mut ids = HashSet::new();
    let mut occurrences = HashSet::new();
    let next = variables
        .iter()
        .map(|v| i64::from(v.id) + 1)
        .max()
        .unwrap_or(0);
    let mut services = vec![];
    for (next, definition) in (next..).zip(definitions) {
        if definition.id < 0
            || !ids.insert(definition.id)
            || definition.vehicles_required < 2
            || !definition.reward.is_finite()
            || definition.members.is_empty()
            || definition.start[0] > definition.start[1]
            || definition
                .start
                .iter()
                .any(|&t| !(-(1_i64 << 52)..=(1_i64 << 52)).contains(&t))
        {
            return Err(invalid(
                "synchronized service identity, vehicles, reward or integer start domain",
            ));
        }
        let row = problem.rows.len();
        problem.rows.push(Row {
            sense: RowSense::Equal,
            rhs: 0.,
        });
        let mut variable_entries = vec![(row, -f64::from(definition.vehicles_required))];
        let mut networks = HashSet::new();
        for member in &definition.members {
            let network = problem
                .networks
                .iter_mut()
                .find(|n| n.id == member.network)
                .ok_or_else(|| invalid("unknown synchronized service network"))?;
            let graph = &mut network.graph;
            // Coverage counts distinct route copies. A repeated service visit
            // in one bounded walk must not impersonate another vehicle.
            if graph.max_route_arcs().is_some()
                && (!graph.hard_walk_groups()
                    || graph
                        .vertices
                        .get(member.vertex)
                        .is_none_or(|v| v.elements.is_empty()))
            {
                return Err(invalid("a bounded-walk synchronized service member requires a declared vertex elementarity group"));
            }
            let r = definition.resource;
            if !networks.insert(member.network)
                || !occurrences.insert((member.network, member.vertex))
                || member.vertex >= graph.vertices.len()
                || member.vertex == graph.source
                || member.vertex == graph.sink
                || graph.resources.get(r) != Some(&ResourceKind::Disposable)
                || graph.arcs.iter().any(|a| a.consumption[r] < 0.)
            {
                return Err(invalid("synchronized service needs unique nonterminal vertices and a nonnegative disposable scalar time axis"));
            }
            let old = graph.vertices[member.vertex].bounds[r];
            let lo = old.lower.max(definition.start[0] as f64);
            let hi = old.upper.min(definition.start[1] as f64);
            if lo <= hi {
                graph.set_bounds(member.vertex, r, lo, hi)?;
            } else {
                let forbidden = problem.rows.len();
                problem.rows.push(Row {
                    sense: RowSense::LessEqual,
                    rhs: 0.,
                });
                for (arc, entries) in graph.arcs.iter().zip(&mut network.arc_entries) {
                    if arc.head == member.vertex {
                        entries.push((forbidden, 1.));
                    }
                }
            }
            let fleet_row = problem.rows.len();
            problem.rows.push(Row {
                sense: RowSense::LessEqual,
                rhs: 0.,
            });
            let capacity = network
                .multiplicity
                .upper
                .floor()
                .min(f64::from(definition.vehicles_required));
            variable_entries.push((fleet_row, -capacity));
            for (arc, entries) in graph.arcs.iter().zip(&mut network.arc_entries) {
                if arc.head == member.vertex {
                    entries.push((row, 1.));
                    entries.push((fleet_row, 1.));
                }
            }
        }
        let variable = i32::try_from(next)
            .map_err(|_| invalid("synchronized service variable identity overflow"))?;
        variables.push(MasterVariable {
            id: variable,
            cost: -definition.reward,
            bounds: Bounds {
                lower: 0.,
                upper: 1.,
            },
            integer: true,
            entries: variable_entries,
        });
        services.push(Service {
            domain: definition.start,
            definition,
            variable,
        });
    }
    Ok(services)
}

/// Apply branch windows to a replay graph. Impossible vertices lose incoming
/// arcs, while valid intersections retain physical IDs and disjoint windows.
pub(super) fn install_windows<const D: usize>(
    oracle: &Oracle<D>,
    graph: &mut Graph<D>,
    blocked: &mut HashSet<i32>,
) {
    for &(v, r, window) in &oracle.synchronized_windows {
        let old = graph.vertices[v].bounds[r];
        let lower = old.lower.max(window.lower);
        let upper = old.upper.min(window.upper);
        if lower <= upper {
            graph.vertices[v].bounds[r] = Window { lower, upper };
        } else {
            blocked.extend(graph.arcs.iter().filter(|a| a.head == v).map(|a| a.id));
        }
    }
}

impl<const D: usize> NodeSnapshot<D> {
    /// Restrict one synchronized start domain on a private fork. Restrictions
    /// affect retained columns, inactive columns, fixed routes and new pricing.
    pub fn with_synchronized_start(&self, id: i32, lower: i64, upper: i64) -> Result<Self> {
        let index = self
            .state
            .synchronized_services
            .iter()
            .position(|s| s.definition.id == id)
            .ok_or_else(|| invalid("unknown synchronized service"))?;
        let service = &self.state.synchronized_services[index];
        let domain = [lower.max(service.domain[0]), upper.min(service.domain[1])];
        if domain[0] > domain[1] {
            return Err(invalid("empty synchronized start domain"));
        }
        let definition = service.definition.clone();
        let mut node = self.fork()?;
        node.state.synchronized_services[index].domain = domain;
        let mut touched = HashSet::new();
        let mut forbidden = vec![];
        for member in &definition.members {
            touched.insert(member.network);
            let oracle = node
                .state
                .oracles
                .iter_mut()
                .find(|o| o.network.id == member.network)
                .unwrap();
            let r = definition.resource;
            let window = Window {
                lower: domain[0] as f64,
                upper: domain[1] as f64,
            };
            oracle.synchronized_windows.push((member.vertex, r, window));
            let old = oracle.solver.graph.vertices[member.vertex].bounds[r];
            let lo = old.lower.max(window.lower);
            let hi = old.upper.min(window.upper);
            if lo <= hi {
                oracle.solver.graph.set_bounds(member.vertex, r, lo, hi)?;
            } else {
                // The master row excludes visits even if pricing retains this
                // vertex. No inconsistent Window is passed into the labeler.
                forbidden.push(RowCoefficients {
                    network: member.network,
                    arcs: oracle
                        .network
                        .graph
                        .arcs
                        .iter()
                        .map(|a| f64::from(a.head == member.vertex))
                        .collect(),
                    per_route: 0.,
                });
            }
        }
        // Several local occurrences may belong to the same network.
        let mut coefficients = BTreeMap::<i32, RowCoefficients>::new();
        for c in forbidden {
            if let Some(old) = coefficients.get_mut(&c.network) {
                for (a, b) in old.arcs.iter_mut().zip(c.arcs) {
                    *a += b;
                }
            } else {
                coefficients.insert(c.network, c);
            }
        }
        if !coefficients.is_empty() {
            node = node.with_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: 0.,
                },
                &coefficients.into_values().collect::<Vec<_>>(),
            )?;
        }
        if resource_groups::profiles::refresh(&mut node, &self.state, &touched)? {
            node = node.with_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: -1.,
                },
                &[],
            )?;
        }
        node.state.correction = None;
        node.state.gradient = None;
        node.warm.stabilization = None;
        Ok(node)
    }
}

pub(super) fn children<const D: usize>(
    output: &Output<D>,
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let node = &output.snapshot;
    for service in &node.state.synchronized_services {
        let [lower, upper] = service.domain;
        let value = output
            .master_values
            .iter()
            .find(|&&(id, _)| id == service.variable)
            .map_or(0., |&(_, v)| v);
        if lower >= upper || value <= tolerance {
            continue;
        }
        let midpoint = lower + (upper - lower) / 2;
        return Ok(Some(vec![
            node.with_synchronized_start(service.definition.id, lower, midpoint)?,
            node.with_synchronized_start(service.definition.id, midpoint + 1, upper)?,
        ]));
    }
    Ok(None)
}
