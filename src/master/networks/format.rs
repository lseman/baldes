//! Shared JSON input and output for the CLI and Python multi-network front ends.
use super::*;
use crate::graph::ResourceKind;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sense {
    Eq,
    Le,
    Ge,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SharedRow {
    pub sense: Sense,
    pub rhs: f64,
}
impl SharedRow {
    pub fn row(&self) -> Row {
        Row {
            sense: match self.sense {
                Sense::Eq => RowSense::Equal,
                Sense::Le => RowSense::LessEqual,
                Sense::Ge => RowSense::GreaterEqual,
            },
            rhs: self.rhs,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Resource {
    #[serde(default = "disposable")]
    pub kind: String,
    #[serde(default)]
    pub initial: Option<f64>,
    /// One [lower, upper] pair per vertex; None uses Graph defaults.
    #[serde(default)]
    pub bounds: Option<Vec<[f64; 2]>>,
}
fn disposable() -> String {
    "disposable".into()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Arc {
    pub tail: usize,
    pub head: usize,
    pub cost: f64,
    #[serde(default)]
    pub consumption: Vec<f64>,
    #[serde(default)]
    pub bounds: Vec<Option<[f64; 2]>>,
    #[serde(default)]
    pub entries: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PricingNetwork {
    pub id: i32,
    pub vertices: usize,
    pub multiplicity: [f64; 2],
    #[serde(default)]
    pub source: usize,
    #[serde(default)]
    pub sink: Option<usize>,
    #[serde(default)]
    pub resources: Vec<Resource>,
    pub arcs: Vec<Arc>,
    #[serde(default)]
    pub route_entries: Vec<(usize, f64)>,
    #[serde(default)]
    pub ng: Option<Vec<Vec<usize>>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub rows: Vec<SharedRow>,
    pub networks: Vec<PricingNetwork>,
}
impl Model {
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| invalid(format!("network model JSON: {e}")))
    }
    pub fn problem<const D: usize>(self) -> Result<Problem<D>> {
        let mut networks = vec![];
        for input in self.networks {
            let mut graph = Graph::<D>::new(input.vertices, input.resources.len())?;
            graph.source = input.source;
            if let Some(sink) = input.sink {
                graph.sink = sink;
            }
            for (r, resource) in input.resources.into_iter().enumerate() {
                graph.resources[r] = match resource.kind.as_str() {
                    "disposable" => ResourceKind::Disposable,
                    "equality" => ResourceKind::Equality,
                    _ => return Err(invalid("resource kind must be disposable or equality")),
                };
                graph.initial[r] = resource.initial;
                if let Some(bounds) = resource.bounds {
                    if bounds.len() != input.vertices {
                        return Err(invalid("resource bounds must have one pair per vertex"));
                    }
                    for (v, [lower, upper]) in bounds.into_iter().enumerate() {
                        graph.set_bounds(v, r, lower, upper)?;
                    }
                }
            }
            if let Some(ng) = input.ng {
                if ng.len() != input.vertices {
                    return Err(invalid("NG lists must have one entry per vertex"));
                }
                for (v, elements) in ng.iter().enumerate() {
                    graph.set_ng(v, elements)?;
                }
            }
            let mut arc_entries = vec![];
            for input in input.arcs {
                let resources = graph.resources.len();
                if (!input.consumption.is_empty() && input.consumption.len() != resources)
                    || (!input.bounds.is_empty() && input.bounds.len() != resources)
                {
                    return Err(invalid(
                        "arc consumption/bounds must match the resource count",
                    ));
                }
                let id = graph.add_arc(input.tail, input.head, input.cost)?;
                let arc = graph.arc_mut(id)?;
                for (r, value) in input.consumption.into_iter().enumerate() {
                    arc.consumption[r] = value;
                }
                for (r, bounds) in input.bounds.into_iter().enumerate() {
                    arc.bounds[r] = bounds.map(|[l, u]| Window::new(l, u)).transpose()?;
                }
                arc_entries.push(input.entries);
            }
            graph.validate()?;
            networks.push(Network {
                id: input.id,
                graph,
                multiplicity: Window::new(input.multiplicity[0], input.multiplicity[1])?,
                arc_entries,
                route_entries: input.route_entries,
            });
        }
        let problem = Problem {
            rows: self.rows.iter().map(SharedRow::row).collect(),
            networks,
        };
        problem.validate()?;
        Ok(problem)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SolveOptions {
    pub tree: bool,
    pub time_limit: f64,
    pub max_iterations: usize,
    pub node_limit: usize,
    pub stabilize: bool,
    pub bound_mode: String,
    pub artificial_cost: f64,
    pub log_interval: Option<usize>,
    pub pricing_options: BTreeMap<String, f64>,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tree: false,
            time_limit: 120.,
            max_iterations: 100000,
            node_limit: 10000,
            stabilize: true,
            bound_mode: "numerical".into(),
            artificial_cost: 2.76,
            log_interval: None,
            pricing_options: BTreeMap::new(),
        }
    }
}
impl SolveOptions {
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| invalid(format!("network options JSON: {e}")))
    }
    pub fn node_options(&self) -> Result<Options> {
        let mut options = Options::default();
        options.cg.time_limit = Duration::try_from_secs_f64(self.time_limit)
            .map_err(|_| invalid("time_limit must be finite and nonnegative"))?;
        options.cg.max_iterations = self.max_iterations;
        options.cg.stabilize = self.stabilize;
        options.cg.bound_mode = cg::BoundMode::parse(&self.bound_mode)?;
        options.cg.artificial_cost = self.artificial_cost;
        options.cg.log_interval = self.log_interval;
        if self.max_iterations == 0 || self.log_interval == Some(0) {
            return Err(invalid("iteration and log limits must be positive"));
        }
        if self.pricing_options.values().any(|v| !v.is_finite()) {
            return Err(invalid("pricing option values must be finite"));
        }
        options.pricing_options = self
            .pricing_options
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        validate_options(&options)?;
        Ok(options)
    }
    pub fn tree_options(&self) -> Result<AssignmentTreeOptions> {
        let node = self.node_options()?;
        Ok(AssignmentTreeOptions {
            time_limit: node.cg.time_limit,
            node,
            node_limit: self.node_limit,
            ..AssignmentTreeOptions::default()
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddRow {
    pub row: SharedRow,
    pub coefficients: Vec<Coefficients>,
}
impl AddRow {
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| invalid(format!("network row JSON: {e}")))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coefficients {
    pub network: i32,
    pub arcs: Vec<f64>,
    #[serde(default)]
    pub per_route: f64,
}
impl From<Coefficients> for RowCoefficients {
    fn from(c: Coefficients) -> Self {
        Self {
            network: c.network,
            arcs: c.arcs,
            per_route: c.per_route,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RouteResult {
    pub network: i32,
    /// Stable registry index for fix_route; absent on fixed/tree witnesses.
    pub route_index: Option<usize>,
    pub value: f64,
    pub cost: f64,
    pub nodes: Vec<i32>,
    pub arcs: Vec<i32>,
    pub resources: Vec<f64>,
    pub resource_profile: Vec<Vec<f64>>,
}
impl RouteResult {
    fn new(route: &NetworkRoute, value: f64, index: Option<usize>) -> Self {
        Self {
            network: route.network,
            route_index: index,
            value,
            cost: route.route.original_cost,
            nodes: route.route.nodes.clone(),
            arcs: route.route.arcs.clone(),
            resources: route.route.resources.clone(),
            resource_profile: route.route.resource_profile.clone(),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct SolveResult {
    pub mode: &'static str,
    pub status: String,
    pub bound_mode: String,
    pub lp: Option<f64>,
    pub bound: Option<f64>,
    pub numerical_bound: Option<f64>,
    pub certified_bound: Option<f64>,
    pub incumbent: Option<f64>,
    pub routes: Vec<RouteResult>,
    pub fixed_routes: Vec<RouteResult>,
    pub duals: Vec<f64>,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub oracle_calls: Vec<(i32, usize)>,
    pub nodes: usize,
    pub open_nodes: usize,
    pub seconds: f64,
}
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
impl SolveResult {
    pub fn root<const D: usize>(output: &Output<D>, seconds: f64) -> Self {
        Self {
            mode: "root",
            status: format!("{:?}", output.master.status),
            bound_mode: output.master.bound_mode.as_str().into(),
            lp: finite(output.master.lp),
            bound: finite(output.master.bound),
            numerical_bound: finite(output.master.numerical_bound),
            certified_bound: finite(output.master.certified_bound),
            incumbent: None,
            routes: output
                .routes
                .iter()
                .zip(&output.values)
                .enumerate()
                .map(|(i, (route, &value))| RouteResult::new(route, value, Some(i)))
                .collect(),
            fixed_routes: output
                .fixed_routes
                .iter()
                .map(|f| RouteResult::new(&f.route, f.quantity, None))
                .collect(),
            duals: output.duals.clone(),
            iterations: output.master.iterations,
            pricing_calls: output.master.pricing_calls,
            oracle_calls: output.oracle_calls.clone(),
            nodes: 0,
            open_nodes: 0,
            seconds,
        }
    }
    pub fn tree(output: &AssignmentTreeOutput, seconds: f64) -> Self {
        Self {
            mode: "tree",
            status: format!("{:?}", output.status),
            bound_mode: output.bound_mode.as_str().into(),
            lp: None,
            bound: finite(output.bound),
            numerical_bound: None,
            certified_bound: None,
            incumbent: finite(output.incumbent),
            routes: output
                .routes
                .iter()
                .map(|f| RouteResult::new(&f.route, f.quantity, None))
                .collect(),
            fixed_routes: vec![],
            duals: vec![],
            iterations: output.iterations,
            pricing_calls: output.pricing_calls,
            oracle_calls: output.oracle_calls.clone(),
            nodes: output.nodes,
            open_nodes: output.open_nodes,
            seconds,
        }
    }
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| invalid(format!("network result JSON: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_json_preserves_f64_costs_at_representable_boundaries() {
        let fixture = include_str!("../../../examples/networks/triangle.json");
        let mut model = Model::from_json(fixture).unwrap();
        for bits in [
            0x3ff0000000000001,
            0x3fb999999999999a,
            0x4340000000000001,
            0x0000000000000001,
        ] {
            let expected = f64::from_bits(bits);
            model.networks[0].arcs[0].cost = expected;
            let text = serde_json::to_string(&model).unwrap();
            let parsed = Model::from_json(&text).unwrap();
            assert_eq!(parsed.networks[0].arcs[0].cost.to_bits(), bits);
        }
    }
}
