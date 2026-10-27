//! Robust PDPTW of Abreu, Battarra, Costa and Munari (2026).
//!
//! Requests pair pickup i with delivery i+n. Capacity is the nominal load
//! plus the Γq largest deviations of requests currently on board (Prop. 1).
//! Pricing uses a pairing plugin and the ordinary BudgetedTime kernel.
//! See `docs/extensions/robust-pd.md` for data conventions and limitations.
use crate::{invalid, Graph, Result};
use std::path::Path;

mod heuristic;
mod physical_graph;

pub use heuristic::{HeuristicOptions, HeuristicSolution};

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub service: f64,
    pub demand: f64,
    pub open: f64,
    pub close: f64,
}

/// Original benchmark data, including separate start and end depots.
#[derive(Clone, Debug)]
pub struct Instance {
    requests: usize,
    capacity: f64,
    nodes: Vec<Node>,
}

impl Instance {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| invalid(format!("cannot read PDPTW instance: {e}")))?;
        Self::from_text(&text)
    }

    /// Header: `n N Q CAPACITY r_mod VALUE`; rows: id x y service demand open close.
    pub fn from_text(text: &str) -> Result<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<_> = lines
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .collect();
        if header.len() != 6 || header[0] != "n" || header[2] != "Q" || header[4] != "r_mod" {
            return Err(invalid("expected n N Q CAPACITY r_mod VALUE"));
        }
        let requests: usize = header[1]
            .parse()
            .map_err(|_| invalid("invalid request count"))?;
        let capacity: f64 = header[3].parse().map_err(|_| invalid("invalid capacity"))?;
        let count = requests
            .checked_mul(2)
            .and_then(|x| x.checked_add(2))
            .ok_or_else(|| invalid("request count overflow"))?;
        let mut nodes = Vec::new();
        for line in lines {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() != 7 || fields[0].parse::<usize>().ok() != Some(nodes.len()) {
                return Err(invalid("PDPTW rows need consecutive ids and seven fields"));
            }
            let values: Vec<f64> = fields[1..]
                .iter()
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|_| invalid("invalid PDPTW node value"))
                })
                .collect::<Result<_>>()?;
            if values.iter().any(|x| !x.is_finite()) || values[2] < 0. || values[4] > values[5] {
                return Err(invalid(
                    "nonfinite node value, negative service or reversed window",
                ));
            }
            nodes.push(Node {
                x: values[0],
                y: values[1],
                service: values[2],
                demand: values[3],
                open: values[4],
                close: values[5],
            });
        }
        if requests == 0 || !capacity.is_finite() || capacity <= 0. || nodes.len() != count {
            return Err(invalid(
                "positive request count/capacity and 2n+2 nodes required",
            ));
        }
        for i in 1..=requests {
            if nodes[i].demand <= 0. || nodes[i + requests].demand != -nodes[i].demand {
                return Err(invalid(
                    "pickup i must have positive demand paired with delivery i+n",
                ));
            }
        }
        for depot in [0, count - 1] {
            if nodes[depot].demand != 0. || nodes[depot].service != 0. {
                return Err(invalid("depots must have zero demand and service"));
            }
        }
        Ok(Self {
            requests,
            capacity,
            nodes,
        })
    }

    pub fn requests(&self) -> usize {
        self.requests
    }
    pub fn capacity(&self) -> f64 {
        self.capacity
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DistanceRounding {
    FullPrecision,
    /// Nearest 0.01, matching the upstream workbook's five-request objectives.
    NearestHundredth,
}

#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub demand_budget: usize,
    pub time_budget: usize,
    pub demand_deviation: f64,
    pub time_deviation: f64,
    pub vehicle_cost: f64,
    pub distance_rounding: DistanceRounding,
    /// Apply the nominal-demand adjustment in paper §6.1, including Γq=0.
    pub adjust_demands: bool,
    /// Requests in each vertex's NG neighbourhood. `None`, or at least the
    /// request count, prices only elementary routes.
    pub ng_size: Option<usize>,
}
impl Default for Parameters {
    fn default() -> Self {
        Self {
            demand_budget: 0,
            time_budget: 0,
            demand_deviation: 0.,
            time_deviation: 0.,
            vehicle_cost: 10_000.,
            distance_rounding: DistanceRounding::NearestHundredth,
            adjust_demands: true,
            ng_size: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Problem {
    instance: Instance,
    parameters: Parameters,
    demands: Vec<f64>,
    deviations: Vec<f64>,
    travel: Vec<Vec<f64>>,
}

/// Independent route replay, one entry per node including both depots.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub loads: Vec<f64>,
    /// Each vector contains worst-case service starts for γ=0..=Γt.
    pub starts: Vec<Vec<f64>>,
    pub cost: f64,
}

impl Problem {
    pub fn new(instance: Instance, parameters: Parameters) -> Result<Self> {
        if [
            parameters.demand_deviation,
            parameters.time_deviation,
            parameters.vehicle_cost,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.)
        {
            return Err(invalid(
                "finite nonnegative deviations and vehicle cost required",
            ));
        }
        let mut demands = Vec::with_capacity(instance.requests);
        let mut deviations = Vec::with_capacity(instance.requests);
        for node in &instance.nodes[1..=instance.requests] {
            let adjustment = (0.3 * node.demand).ceil();
            let demand =
                if parameters.adjust_demands && node.demand + adjustment > instance.capacity {
                    instance.capacity - adjustment
                } else {
                    node.demand
                };
            let deviation = (parameters.demand_deviation * demand).ceil();
            if demand <= 0. || !deviation.is_finite() {
                return Err(invalid("invalid adjusted demand or deviation overflow"));
            }
            demands.push(demand);
            deviations.push(deviation);
        }
        // The same explicit nominal distance convention for time and cost.
        let travel: Vec<Vec<f64>> = instance
            .nodes
            .iter()
            .map(|a| {
                instance
                    .nodes
                    .iter()
                    .map(|b| {
                        let distance = (a.x - b.x).hypot(a.y - b.y);
                        match parameters.distance_rounding {
                            DistanceRounding::FullPrecision => distance,
                            DistanceRounding::NearestHundredth => (100. * distance).round() / 100.,
                        }
                    })
                    .collect()
            })
            .collect();
        if travel.iter().flatten().any(|t| {
            !t.is_finite()
                || !(parameters.time_deviation * t).ceil().is_finite()
                || !(t + (parameters.time_deviation * t).ceil() + parameters.vehicle_cost)
                    .is_finite()
        }) {
            return Err(invalid("travel time or cost overflow"));
        }
        Ok(Self {
            instance,
            parameters,
            demands,
            deviations,
            travel,
        })
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }
    pub fn parameters(&self) -> Parameters {
        self.parameters
    }
    pub fn demands(&self) -> &[f64] {
        &self.demands
    }
    pub fn deviations(&self) -> &[f64] {
        &self.deviations
    }

    fn load(&self, open: &[bool]) -> f64 {
        let nominal: f64 = self
            .demands
            .iter()
            .zip(open)
            .filter(|(_, o)| **o)
            .map(|(q, _)| q)
            .sum();
        let mut deviations: Vec<_> = self
            .deviations
            .iter()
            .zip(open)
            .filter(|(_, o)| **o)
            .map(|(q, _)| *q)
            .collect();
        deviations.sort_by(|a, b| b.total_cmp(a));
        nominal
            + deviations
                .iter()
                .take(self.parameters.demand_budget)
                .sum::<f64>()
    }

    /// Validate pairing, precedence, elementarity, robust load and every window.
    /// Route ids use the original file's distinct depots 0 and 2n+1.
    pub fn replay(&self, route: &[usize]) -> Result<Schedule> {
        let n = self.instance.requests;
        let sink = 2 * n + 1;
        if route.len() < 4
            || route.first() != Some(&0)
            || route.last() != Some(&sink)
            || route[1..route.len() - 1]
                .iter()
                .any(|v| *v == 0 || *v >= sink)
        {
            return Err(invalid("nonempty PD route must start at 0 and end at 2n+1"));
        }
        let mut seen = vec![false; sink + 1];
        seen[0] = true;
        let mut open = vec![false; n];
        let mut start =
            vec![self.instance.nodes[0].open; self.parameters.time_budget.min(2 * n + 1) + 1];
        let mut result = Schedule {
            loads: vec![0.],
            starts: vec![start.clone()],
            cost: self.parameters.vehicle_cost,
        };
        for edge in route.windows(2) {
            let (u, v) = (edge[0], edge[1]);
            if seen[v] || (v == sink && open.iter().any(|x| *x)) {
                return Err(invalid("repeated node or uncompleted request"));
            }
            seen[v] = true;
            if v <= n {
                open[v - 1] = true;
            } else if v < sink {
                if !open[v - n - 1] {
                    return Err(invalid("delivery before pickup"));
                }
                open[v - n - 1] = false;
            }
            let load = self.load(&open);
            if load > self.instance.capacity {
                return Err(invalid("robust capacity exceeded"));
            }
            let duration = self.instance.nodes[u].service + self.travel[u][v];
            let deviation = (self.parameters.time_deviation * self.travel[u][v]).ceil();
            let node = self.instance.nodes[v];
            let next: Vec<_> = start
                .iter()
                .enumerate()
                .map(|(g, &b)| {
                    let mut value = node.open.max(b + duration);
                    if g > 0 {
                        value = value.max(start[g - 1] + duration + deviation);
                    }
                    value
                })
                .collect();
            if next.iter().any(|b| !b.is_finite() || *b > node.close) {
                return Err(invalid("robust time window exceeded"));
            }
            start = next;
            result.loads.push(load);
            result.starts.push(start.clone());
            result.cost += self.travel[u][v];
        }
        if !result.cost.is_finite() {
            return Err(invalid("route cost overflow"));
        }
        Ok(result)
    }

    /// Physical pricing graph. Duals are per request, charged at pickups.
    pub fn pricing_graph(&self, duals: &[f64], fleet_dual: f64) -> Result<Graph> {
        self.build_graph(duals, fleet_dual)
    }

    /// Set partitioning with one equality per request and up to n vehicles.
    #[cfg(feature = "clp")]
    pub fn master_problem(&self) -> Result<crate::master::model::Problem> {
        use crate::master::{model, routing::Row, RowSense};
        let n = self.instance.requests;
        let graph = self.pricing_graph(&vec![0.; n], 0.)?;
        let arc_entries = graph
            .arcs
            .iter()
            .map(|a| {
                if (1..=n).contains(&a.head) {
                    vec![(a.head - 1, 1.)]
                } else {
                    vec![]
                }
            })
            .collect();
        Ok(model::Problem {
            rows: (0..n)
                .map(|_| Row {
                    sense: RowSense::Equal,
                    rhs: 1.,
                })
                .collect(),
            networks: vec![model::Network {
                id: 0,
                graph,
                multiplicity: crate::Window::new(0., n as f64)?,
                arc_entries,
                route_entries: vec![],
                activity_entries: vec![],
            }],
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/extensions/robust_pd/tests.rs"]
mod tests;
