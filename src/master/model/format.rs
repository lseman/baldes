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
    /// Optional disjoint service-start windows per vertex (null uses bounds).
    #[cfg(feature = "multi-time-windows")]
    #[serde(default)]
    pub time_windows: Option<Vec<Option<Vec<[f64; 2]>>>>,
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
    #[serde(deserialize_with = "multiplicity_bounds")]
    pub multiplicity: [f64; 2],
    #[serde(default)]
    pub source: usize,
    #[serde(default)]
    pub sink: Option<usize>,
    #[serde(default)]
    pub resources: Vec<Resource>,
    #[serde(default)]
    pub coupled_resources: Vec<crate::graph::coupled::CoupledResource>,
    /// Graph-level costs, using each descriptor's declared normalization.
    /// A resource bound to an original variable gets its cost from that variable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coupled_objective_coefficients: Vec<(usize, f64)>,
    /// Explicitly use the computed-front program as the feasibility
    /// domain for FrontAlgebra resources; defaults to source controls.
    #[serde(default)]
    pub front_program_domain: bool,
    /// Integer-route domain; bounded walks require an explicit finite arc cap.
    #[serde(default)]
    pub route_domain: crate::graph::RouteDomain,
    #[serde(default)]
    pub elementary: bool,
    pub arcs: Vec<Arc>,
    #[serde(default)]
    pub route_entries: Vec<(usize, f64)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activity_entries: Vec<AccumulatedActivityEntry>,
    #[serde(default)]
    pub ng: Option<Vec<Vec<usize>>>,
}

fn multiplicity_bounds<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<[f64; 2], D::Error> {
    let [lower, upper] = <[Option<f64>; 2]>::deserialize(deserializer)?;
    let lower =
        lower.ok_or_else(|| serde::de::Error::custom("multiplicity lower bound must be finite"))?;
    Ok([lower, upper.unwrap_or(f64::INFINITY)])
}

fn pricing_collection<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<PricingNetwork>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Input {
        One(Box<PricingNetwork>),
        Many(Vec<PricingNetwork>),
    }
    Ok(match Input::deserialize(deserializer)? {
        Input::One(network) => vec![*network],
        Input::Many(networks) => networks,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pricing_variables: Vec<OriginalPricingVariable>,
    #[serde(default)]
    pub rows: Vec<SharedRow>,
    #[serde(deserialize_with = "pricing_collection")]
    pub networks: Vec<PricingNetwork>,
}
impl Model {
    pub fn from_json(text: &str) -> Result<Self> {
        let mut input: serde_json::Value =
            serde_json::from_str(text).map_err(|e| invalid(format!("network model JSON: {e}")))?;
        if input.is_array()
            || input
                .as_object()
                .is_some_and(|object| !object.contains_key("networks"))
        {
            input = serde_json::json!({"networks": input});
        }
        serde_json::from_value(input).map_err(|e| invalid(format!("network model JSON: {e}")))
    }
    pub fn problem<const D: usize>(self) -> Result<Problem<D>> {
        self.compile().map(|(problem, _)| problem)
    }
    /// Compile original variable associations exactly once and retain identity.
    pub fn compile<const D: usize>(mut self) -> Result<(Problem<D>, PricingVariableRegistry)> {
        let variables = std::mem::take(&mut self.pricing_variables);
        for network in &self.networks {
            if network
                .coupled_objective_coefficients
                .iter()
                .any(|&(resource, _)| {
                    variables.iter().any(|variable| {
                        variable.network == network.id
                            && variable.accumulated_resource == Some(resource)
                    })
                })
            {
                return Err(invalid("a coupled resource cost must be declared either on its original pricing variable or on the network, not both"));
            }
        }
        compile_pricing_variables(self.physical_problem()?, &variables)
    }
    fn physical_problem<const D: usize>(self) -> Result<Problem<D>> {
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
                #[cfg(feature = "multi-time-windows")]
                if let Some(unions) = resource.time_windows {
                    if unions.len() != input.vertices {
                        return Err(invalid("time_windows must have one entry per vertex"));
                    }
                    for (vertex, union) in unions.into_iter().enumerate() {
                        if let Some(union) = union {
                            let windows = union
                                .into_iter()
                                .map(|[a, b]| Window::new(a, b))
                                .collect::<Result<Vec<_>>>()?;
                            graph.set_time_windows(vertex, r, &windows)?;
                        }
                    }
                }
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
            graph.elementary = input.elementary;
            graph.set_route_domain(input.route_domain)?;
            graph.set_coupled_resources(&input.coupled_resources)?;
            if input.front_program_domain {
                graph.set_front_program_domain(true)?;
            }
            graph.set_coupled_objective_coefficients(&input.coupled_objective_coefficients)?;
            graph.validate()?;
            networks.push(Network {
                id: input.id,
                graph,
                // Multiplicity is an aggregate count domain, not a resource
                // window. The shared validator permits an infinite upper cap.
                multiplicity: Window {
                    lower: input.multiplicity[0],
                    upper: input.multiplicity[1],
                },
                arc_entries,
                route_entries: input.route_entries,
                activity_entries: input.activity_entries,
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
    pub network_pricing_schedule: NetworkPricingSchedule,
    pub network_unpromising_limit: usize,
    /// Native promising-status visit budget (default one million).
    pub network_promising_limit: usize,
    pub network_branching: NetworkBranchingPolicy,
    pub network_tree_search: TreeSearchPolicy,
    pub network_strategy2_bound_rounding: IntegerBoundRounding,
    /// Primary frontier capacity before nodes use the overflow heap.
    pub network_open_node_limit: usize,
    pub network_scoring_gap_tolerance: f64,
    pub network_share_inactive_columns: bool,
    pub network_search_inactive_columns: bool,
    pub variable_boxes: Vec<VariableBox>,
    pub original_variable_boxes: Vec<OriginalVariableBound>,
    pub network_branch_priorities: NetworkBranchPriorities,
    pub network_branch_directions: BTreeMap<String, NetworkBranchDirection>,
    pub mapped_variables: Vec<MappedBranchVariable>,
    pub packing: Option<PackingBranching>,
    pub mapped_strong_branching: bool,
    pub synchronized_services: Vec<SynchronizedService>,
    pub enumerate: bool,
    pub enumeration_route_limit: usize,
    pub enumeration_work_limit: usize,
    pub enumeration_stored_arc_limit: usize,
    pub enumeration_time_limit: f64,
    pub enumerated_mip: bool,
    pub mip_time_limit: f64,
    pub mip_node_limit: usize,
    pub mip_objective_cutoff: Option<f64>,
    pub time_limit: f64,
    pub max_iterations: usize,
    pub node_limit: usize,
    pub stabilize: bool,
    pub stabilization_max_tree_depth: i64,
    pub proximal: Option<crate::master::stabilization::ProximalParameters>,
    pub bound_mode: String,
    pub artificial_cost: f64,
    pub log_interval: Option<usize>,
    pub pricing_options: BTreeMap<String, f64>,
}
impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tree: false,
            network_pricing_schedule: NetworkPricingSchedule::default(),
            network_unpromising_limit: 1,
            network_promising_limit: 1_000_000,
            network_branching: NetworkBranchingPolicy::default(),
            network_tree_search: TreeSearchPolicy::default(),
            network_strategy2_bound_rounding: IntegerBoundRounding::default(),
            network_open_node_limit: 1000,
            network_scoring_gap_tolerance: 1e-6,
            network_share_inactive_columns: false,
            network_search_inactive_columns: false,
            variable_boxes: vec![],
            original_variable_boxes: vec![],
            network_branch_priorities: NetworkBranchPriorities::default(),
            network_branch_directions: BTreeMap::new(),
            mapped_variables: vec![],
            packing: None,
            mapped_strong_branching: true,
            synchronized_services: vec![],
            enumerate: false,
            enumeration_route_limit: 10000,
            enumeration_work_limit: 10000000,
            enumeration_stored_arc_limit: 1000000,
            enumeration_time_limit: 1.,
            enumerated_mip: false,
            mip_time_limit: 1.,
            mip_node_limit: 100000,
            mip_objective_cutoff: None,
            time_limit: 120.,
            max_iterations: 100000,
            node_limit: 10000,
            stabilize: true,
            stabilization_max_tree_depth: 10_000,
            proximal: None,
            bound_mode: "numerical".into(),
            artificial_cost: 2.76,
            log_interval: None,
            pricing_options: BTreeMap::new(),
        }
    }
}
impl SolveOptions {
    pub fn node_options_with_registry(
        &self,
        registry: &PricingVariableRegistry,
    ) -> Result<Options> {
        let mut unbound = self.clone();
        unbound.original_variable_boxes.clear();
        let mut node = unbound.node_options()?;
        registry.configure_pricing(&mut node);
        registry.configure_boxes(&mut node, &self.original_variable_boxes)?;
        Ok(node)
    }
    pub fn tree_options_with_registry(
        &self,
        registry: &PricingVariableRegistry,
    ) -> Result<ModelTreeOptions> {
        let mut unbound = self.clone();
        unbound.original_variable_boxes.clear();
        let mut tree = unbound.tree_options()?;
        registry.configure_tree(&mut tree)?;
        registry.configure_boxes(&mut tree.node, &self.original_variable_boxes)?;
        Ok(tree)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| invalid(format!("network options JSON: {e}")))
    }
    pub fn node_options(&self) -> Result<Options> {
        if !self.original_variable_boxes.is_empty() {
            return Err(invalid(
                "original-variable boxes require a pricing-variable registry",
            ));
        }
        if !self.network_branch_directions.is_empty() && !self.tree {
            return Err(invalid("network_branch_directions requires tree=true"));
        }
        if self
            .network_branch_directions
            .keys()
            .any(|name| name.trim().is_empty())
        {
            return Err(invalid(
                "branch direction requires a nonempty generator name",
            ));
        }
        if self.packing.is_some() && !self.tree {
            return Err(invalid("packing requires tree=true"));
        }
        if self.network_share_inactive_columns && !self.tree {
            return Err(invalid("network_share_inactive_columns requires tree=true"));
        }
        let mut options = Options::default();
        options.search_inactive_columns = self.network_search_inactive_columns;
        options.variable_boxes = self.variable_boxes.clone();
        options.network_pricing_schedule = self.network_pricing_schedule;
        options.network_unpromising_limit = self.network_unpromising_limit;
        options.network_promising_limit = self.network_promising_limit;
        options.cg.proximal = self.proximal.clone();
        options.synchronized_services = self.synchronized_services.clone();
        if self.enumerated_mip && (!self.tree || !cfg!(feature = "highs")) {
            return Err(invalid(
                "enumerated MIP requires tree=True and the highs feature",
            ));
        }
        if let Some(cutoff) = self.mip_objective_cutoff {
            if !cutoff.is_finite() || !self.enumerated_mip {
                return Err(invalid(
                    "mip_objective_cutoff must be finite and requires enumerated MIP",
                ));
            }
        }
        let enumeration_time = Duration::try_from_secs_f64(self.enumeration_time_limit)
            .map_err(|_| invalid("enumeration_time_limit must be finite and nonnegative"))?;
        Duration::try_from_secs_f64(self.mip_time_limit)
            .map_err(|_| invalid("mip_time_limit must be finite and nonnegative"))?;
        if self.enumerate || self.enumerated_mip {
            options.enumeration = Some(EnumerationOptions {
                max_routes: self.enumeration_route_limit,
                max_work: self.enumeration_work_limit,
                max_stored_arcs: self.enumeration_stored_arc_limit,
                time_limit: enumeration_time,
            });
        }
        options.cg.time_limit = Duration::try_from_secs_f64(self.time_limit)
            .map_err(|_| invalid("time_limit must be finite and nonnegative"))?;
        options.cg.max_iterations = self.max_iterations;
        options.cg.stabilize = self.stabilize;
        options.cg.stabilization_max_tree_depth = self.stabilization_max_tree_depth;
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
    pub fn tree_options(&self) -> Result<ModelTreeOptions> {
        let node = self.node_options()?;
        Ok(ModelTreeOptions {
            branching: self.network_branching,
            tree_search: self.network_tree_search,
            strategy2_bound_rounding: self.network_strategy2_bound_rounding,
            open_node_limit: self.network_open_node_limit,
            scoring_gap_tolerance: self.network_scoring_gap_tolerance,
            share_inactive_columns: self.network_share_inactive_columns,
            branch_priorities: self.network_branch_priorities.clone(),
            branch_directions: self.network_branch_directions.clone(),
            mapped_variables: self.mapped_variables.clone(),
            packing: self.packing.clone(),
            mapped_strong_branching: self
                .mapped_strong_branching
                .then(crate::master::tree::StrongBranchingOptions::default),
            time_limit: node.cg.time_limit,
            node,
            node_limit: self.node_limit,
            enumerated_mip: if self.enumerated_mip {
                Some(crate::master::solvers::mip::MipOptions {
                    time_limit: Duration::try_from_secs_f64(self.mip_time_limit)
                        .map_err(|_| invalid("MIP time limit"))?,
                    node_limit: self.mip_node_limit,
                    objective_cutoff: self.mip_objective_cutoff.unwrap_or(f64::INFINITY),
                })
            } else {
                None
            },
            ..ModelTreeOptions::default()
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddRow {
    #[serde(default)]
    pub dynamic: bool,
    pub row: SharedRow,
    pub coefficients: Vec<Coefficients>,
    #[serde(default)]
    pub activities: Vec<AccumulatedActivityCoefficients>,
}

/// The same owned grouped-resource restriction used by Rust, CLI and Python.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceGroupRestriction {
    pub packing_row: usize,
    pub resource: usize,
    pub threshold: f64,
    pub upper: bool,
    pub members: Vec<ResourceGroupMember>,
}
impl ResourceGroupRestriction {
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).map_err(|e| invalid(format!("resource group JSON: {e}")))
    }
    pub fn apply<const D: usize>(&self, node: &NodeSnapshot<D>) -> Result<NodeSnapshot<D>> {
        node.with_resource_group_branch(
            self.packing_row,
            self.resource,
            self.threshold,
            self.upper,
            &self.members,
        )
    }
}
impl AddRow {
    /// Apply one owned row request through the common node lifecycle.
    pub fn apply<const D: usize>(self, node: &NodeSnapshot<D>) -> Result<NodeSnapshot<D>> {
        let shared = node.state.shared_rows.len();
        let coefficients = self
            .coefficients
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>();
        let mut child = if self.dynamic {
            node.with_dynamic_row(self.row.row(), &coefficients)?
        } else {
            node.with_row(self.row.row(), &coefficients)?
        };
        if self.activities.is_empty() {
            return Ok(child);
        }
        super::activity_rows::install(&mut child.state, shared, &self.activities)?;
        super::nonlinear::refresh_row(child, shared)
    }
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

/// Encode a descriptor-level source-control search for the shared frontends.
pub fn control_optimization_json(
    result: &crate::master::controls::ControlOptimization,
) -> Result<String> {
    serde_json::to_string(result)
        .map_err(|error| invalid(format!("control optimization JSON: {error}")))
}

/// Decode explicit source controls through the shared model JSON boundary.
pub fn parse_paired_controls(text: &str) -> Result<Vec<crate::graph::coupled::PairedControl>> {
    serde_json::from_str(text).map_err(|error| invalid(format!("paired controls JSON: {error}")))
}

#[derive(Debug, Serialize)]
pub struct RouteResult {
    pub coupled_controls: Vec<crate::graph::coupled::PairedControl>,
    pub coupled_values: Vec<(usize, f64)>,
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
            coupled_values: route.route.coupled_values.clone(),
            coupled_controls: route.route.coupled_controls.clone(),
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
pub struct OriginalVariableResult {
    pub network: i32,
    pub id: i32,
    pub lower: f64,
    pub upper: f64,
}
#[derive(Debug, Serialize)]
pub struct SolveResult {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub original_variables: Vec<OriginalVariableResult>,
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
    pub tree_processed_nodes: usize,
    pub open_nodes: usize,
    pub seconds: f64,
    pub enumeration_reports: Vec<EnumerationReport>,
    pub enumerated_mip_calls: usize,
    pub enumerated_mip_nodes: usize,
    pub enumerated_mip_failures: usize,
    pub strong_branching_trials: usize,
    pub mapped_history_ranked: usize,
    pub mapped_branch_evaluations: Vec<MappedBranchEvaluation>,
}
fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}
impl SolveResult {
    /// Original activities at a root LP point or an available integer witness.
    pub fn with_original_variables(mut self, registry: &PricingVariableRegistry) -> Result<Self> {
        if self.mode != "root" && self.incumbent.is_none() {
            return Ok(self);
        }
        self.original_variables = registry
            .variables()
            .iter()
            .map(|v| {
                let bounds = registry.activity_from_columns(
                    v.network,
                    v.id,
                    self.routes
                        .iter()
                        .chain(&self.fixed_routes)
                        .map(|r| OriginalColumn {
                            network: r.network,
                            arcs: &r.arcs,
                            coupled_values: Some(&r.coupled_values),
                            coupled_controls: Some(&r.coupled_controls),
                            quantity: r.value,
                        }),
                )?;
                Ok(OriginalVariableResult {
                    network: v.network,
                    id: v.id,
                    lower: bounds.lower,
                    upper: bounds.upper,
                })
            })
            .collect::<Result<_>>()?;
        Ok(self)
    }

    pub fn root<const D: usize>(output: &Output<D>, seconds: f64) -> Self {
        Self {
            original_variables: vec![],
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
            tree_processed_nodes: 0,
            open_nodes: 0,
            enumeration_reports: vec![output.snapshot.enumeration_report().clone()],
            enumerated_mip_calls: 0,
            enumerated_mip_nodes: 0,
            enumerated_mip_failures: 0,
            strong_branching_trials: 0,
            mapped_history_ranked: 0,
            mapped_branch_evaluations: vec![],
            seconds,
        }
    }
    pub fn tree(output: &ModelTreeOutput, seconds: f64) -> Self {
        Self {
            original_variables: vec![],
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
            tree_processed_nodes: output.processed_nodes,
            open_nodes: output.open_nodes,
            enumeration_reports: output.enumeration_reports.clone(),
            enumerated_mip_calls: output.enumerated_mip_calls,
            enumerated_mip_nodes: output.enumerated_mip_nodes,
            enumerated_mip_failures: output.enumerated_mip_failures,
            strong_branching_trials: output.strong_branching_trials,
            mapped_history_ranked: output.mapped_history_ranked,
            mapped_branch_evaluations: output.mapped_branch_evaluations.clone(),
            seconds,
        }
    }
    pub fn mip(output: &EnumeratedMipOutput, report: &EnumerationReport, seconds: f64) -> Self {
        let proof = matches!(
            output.mip.status,
            crate::master::solvers::mip::MipStatus::Optimal
                | crate::master::solvers::mip::MipStatus::Infeasible
                | crate::master::solvers::mip::MipStatus::ObjectiveCutoff
        );
        Self {
            original_variables: vec![],
            mode: "mip",
            status: format!("{:?}", output.mip.status),
            bound_mode: "numerical".into(),
            lp: None,
            bound: finite(output.mip.bound),
            numerical_bound: finite(output.mip.bound),
            certified_bound: None,
            incumbent: output
                .mip
                .solution
                .as_ref()
                .and_then(|s| finite(s.objective)),
            routes: output
                .route_values
                .iter()
                .map(|f| RouteResult::new(&f.route, f.quantity, None))
                .collect(),
            fixed_routes: vec![],
            duals: vec![],
            iterations: 0,
            pricing_calls: 0,
            oracle_calls: vec![],
            nodes: output.mip.nodes,
            tree_processed_nodes: 0,
            open_nodes: usize::from(!proof),
            seconds,
            enumeration_reports: vec![report.clone()],
            enumerated_mip_calls: 1,
            enumerated_mip_nodes: output.mip.nodes,
            enumerated_mip_failures: 0,
            strong_branching_trials: 0,
            mapped_history_ranked: 0,
            mapped_branch_evaluations: vec![],
        }
    }
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| invalid(format!("network result JSON: {e}")))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/format/tests.rs"]
mod tests;
