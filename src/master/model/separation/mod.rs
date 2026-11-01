//! Automatic cut separation between column-generation rounds.
//!
//! Separation applies to networks whose shared unit rows partition their interior
//! vertices consistently (see [`Problem::set_partitioning`]).
//! Separated cuts become dynamic shared rows: rank-one and strong k-path cuts
//! are full registry cuts priced by the labeling engine, knapsack cuts are
//! resource rows and capacity cuts are arc-linear rows of the network.
//! Separators run by decreasing priority (capacity 3 at the root and 1 below
//! it, rank-one family 1, dynamic NG at its configured priority); a lower
//! level runs only while nothing at a higher level was added in that round.
use super::*;
use crate::cuts::capacity::{CapacityKind, CapacitySeparator};
use crate::cuts::knapsack::{KnapsackColumn, KnapsackKind, ResourceKnapsackSeparator};
use crate::cuts::kpath::{GeneralKPathSeparator, KPathSeparator};
use crate::cuts::FractionalColumn;
use crate::master::LpMaster;
mod options;
pub(crate) use options::KPathSeparation;
pub use options::{
    CapacityCutConfig, CutOptions, GeneralKPathCutConfig, KPathCutConfig, KnapsackCutConfig,
};

const CAPACITY_PRIORITY: f64 = 3.;
const RANK_ONE_PRIORITY: f64 = 1.;
/// Master-variable ID of `Theta_v` is this base plus the vertex index.
pub const THETA_VARIABLE_BASE: i32 = 1 << 30;

/// Exact route cost in excess of its arc costs (Tomazella, Munari and
/// Morabito 2026, eq. 31). Each interior vertex `j` gets a master variable
/// `Theta_j >= 0` of unit cost; an integral route `r` whose lowest vertex is
/// `j` and whose excess is positive receives the Theta optimality cut
/// `Theta_j / excess(r) - sum_{a in A(r)} x_a >= 1 - |A(r)|`,
/// binding only when every arc of `r` is used. Integral routes are separated
/// before any other cut, so an integral master solution pays its exact cost.
pub trait RouteRecourse: Send + Sync {
    /// Nonnegative excess of the route visiting `customers` (graph vertex
    /// indices, in visiting order, source and sink excluded).
    fn excess(&self, customers: &[usize]) -> Result<f64>;
}

/// A shared route-recourse oracle.
#[derive(Clone)]
pub struct Recourse(pub Arc<dyn RouteRecourse>);
impl std::fmt::Debug for Recourse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Recourse(..)")
    }
}

/// The Theta master variables of a one-network model, one per interior vertex.
pub(super) fn theta_variables<const D: usize>(problem: &Problem<D>) -> Result<Vec<MasterVariable>> {
    let [network] = problem.networks.as_slice() else {
        return Err(invalid(
            "route recourse requires exactly one pricing network",
        ));
    };
    let g = &network.graph;
    (0..g.vertices.len())
        .filter(|&v| v != g.source && v != g.sink)
        .map(|v| {
            Ok(MasterVariable {
                id: THETA_VARIABLE_BASE
                    .checked_add(i32::try_from(v).map_err(|_| invalid("Theta vertex"))?)
                    .ok_or_else(|| invalid("Theta vertex"))?,
                cost: 1.,
                bounds: crate::master::preprocessing::Bounds {
                    lower: 0.,
                    upper: f64::INFINITY,
                },
                integer: false,
                entries: vec![],
            })
        })
        .collect()
}

/// Cut families separated automatically during column generation.
#[derive(Clone, Debug, Default)]
pub struct SeparationOptions {
    /// Rank-one cuts (`enabled`) and the tailing-off controls of every family.
    pub rank_one: CutOptions,
    pub capacity: Option<CapacityCutConfig>,
    pub kpath: Option<KPathCutConfig>,
    pub general_kpath: Option<GeneralKPathCutConfig>,
    pub knapsack: Option<KnapsackCutConfig>,
    /// Separator priority of dynamic NG strengthening; 0 disables it.
    pub relaxation_improvement_priority: f64,
    /// Caller-supplied objective upper bound for the separation gap test.
    /// It must bound every integer solution of interest.
    pub upper_bound: Option<f64>,
    /// Route recourse imposed by Theta optimality cuts (fresh solves add the
    /// Theta master variables). Enumeration is disabled with recourse, since
    /// enumerated routes would be selected by their arc costs alone.
    pub route_recourse: Option<Recourse>,
}
impl SeparationOptions {
    pub fn is_active(&self) -> bool {
        self.rank_one.enabled
            || self.capacity.is_some()
            || self.kpath.is_some()
            || self.general_kpath.is_some()
            || self.knapsack.is_some()
            || self.relaxation_improvement_priority > 0.
            || self.route_recourse.is_some()
    }
    /// The master controls that separated cuts require: cut artificials cost
    /// `artificial_cost / 1.2` and the separator owns tailing off. The solver
    /// enables row cleanup only when the model supports row remapping.
    pub(super) fn configure(&self, cg: &mut cg::Options) {
        if !self.is_active() {
            return;
        }
        cg.remove_basic_cuts = false;
        cg.cut_artificial_cost = if cg.artificial_cost > 0. {
            cg.artificial_cost / 1.2
        } else {
            0.
        };
        cg.tailing_off_counter = 1_000_000;
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Separator {
    Capacity,
    KPath,
    Knapsack,
    RankOne,
    Relaxation,
}

/// Separation state retained by a node and inherited by its descendants.
#[derive(Clone)]
pub(super) struct CutState {
    pub(super) options: CutOptions,
    /// Memory mode chosen at the root when automatic rank-one memory search ran.
    pub(super) resolved_memory_type: Option<crate::cuts::MemoryType>,
    /// Whether this node's cut generation stopped at the lowest priority due to tailing off.
    pub(super) stopped_by_tailing_off: bool,
    /// Vertices with a unit cover row, in row order.
    customers: Vec<usize>,
    /// Per network (oracle order): its vertex of each packing row. Several
    /// networks share rank-one cuts over the same packing rows.
    network_customers: Vec<Vec<usize>>,
    /// Per network: each of its vertices in the first network's vertex space
    /// (through the shared packing rows; source and sink map to the depot).
    lead_maps: Vec<Vec<Option<usize>>>,
    /// Registry cut index and shared row of separated rank-one/k-path cuts.
    rows: Vec<(usize, usize)>,
    /// Cut memory when the coefficients of `rows` were last written.
    memories: Vec<Vec<Vec<bool>>>,
    pub lp_before_cuts: f64,
    separation_lp: f64,
    rounds: usize,
    tailing: usize,
    capacity_separator: Option<CapacitySeparator>,
    capacity_priority: f64,
    capacity_rows: Vec<(CapacityKind, usize)>,
    kpath_separator: Option<KPathSeparation>,
    kpath_indices: HashSet<usize>,
    knapsack_separator: Option<ResourceKnapsackSeparator>,
    level: f64,
    relaxation_priority: f64,
    pub(super) upper_bound: f64,
    recourse: Option<Recourse>,
    /// Theta master-variable index by vertex.
    theta: Vec<Option<usize>>,
    /// Sorted, deduplicated arc IDs of each separated Theta cut route.
    theta_cuts: Vec<Vec<i32>>,
    /// Master columns to delete after a relaxation strengthening.
    pub removed_columns: Vec<usize>,
}

impl CutState {
    pub(super) fn rank_one_cuts_present(&self) -> bool {
        self.rows
            .iter()
            .any(|(index, _)| !self.kpath_indices.contains(index))
    }
    fn separators(&self) -> Vec<(f64, Separator)> {
        let mut list = vec![];
        if self.capacity_separator.is_some() {
            list.push((self.capacity_priority, Separator::Capacity));
        }
        if self.kpath_separator.is_some() {
            list.push((RANK_ONE_PRIORITY, Separator::KPath));
        }
        if self.knapsack_separator.is_some() {
            list.push((RANK_ONE_PRIORITY, Separator::Knapsack));
        }
        if self.options.enabled {
            list.push((RANK_ONE_PRIORITY, Separator::RankOne));
        }
        if self.relaxation_priority > 0. {
            list.push((self.relaxation_priority, Separator::Relaxation));
        }
        // On a tie the relaxation improvement sorts after the cut families.
        list.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then((a.1 == Separator::Relaxation).cmp(&(b.1 == Separator::Relaxation)))
        });
        list
    }
    fn rank_one_active(&self) -> bool {
        (self.options.enabled || self.kpath_separator.is_some()) && self.level <= RANK_ONE_PRIORITY
    }
    fn next_level(&self) -> Option<f64> {
        self.separators()
            .into_iter()
            .map(|(p, _)| p)
            .find(|&p| p < self.level)
    }
    pub fn cut_count(&self) -> usize {
        self.rows.len() + self.capacity_rows.len() + self.theta_cuts.len()
    }
}

/// A validated integer point can guide ordinary cut separation at the root.
/// Recourse requires a separate exact objective check before using a witness.
pub(super) fn tracks_integer_witness<const D: usize>(state: &State<D>) -> bool {
    state
        .cuts
        .as_ref()
        .is_some_and(|cuts| cuts.recourse.is_none())
}

/// Compact routing masters with aligned registry rows. More general models
/// retain their rows until all their row-indexed structures support remapping.
pub(super) fn supports_row_cleanup<const D: usize>(state: &State<D>) -> bool {
    state
        .cuts
        .as_ref()
        .is_some_and(|cuts| cuts.recourse.is_none() && cuts.knapsack_separator.is_none())
        && !state.oracles.is_empty()
        && state.master_variables.is_empty()
        && state.variable_boxes.is_empty()
        && state.grouped_variable_boxes.is_empty()
        && state.synchronized_services.is_empty()
        && state.fixed_routes.is_empty()
        && state.universe.is_none()
        && state.oracles.iter().all(|oracle| {
            oracle.activity_rows.is_empty()
                && oracle.presence_rows.is_empty()
                && oracle.knapsack_rows.is_empty()
                && oracle.resource_groups.is_empty()
                && oracle.ryan_foster_rows.is_empty()
        })
}

pub(super) fn protected_rows<const D: usize>(state: &State<D>) -> Vec<usize> {
    if !supports_row_cleanup(state) {
        return (0..state.shared_offset + state.shared_rows.len()).collect();
    }
    let Some(cuts) = &state.cuts else {
        return vec![];
    };
    let eligible: HashSet<_> = cuts.rows.iter().map(|&(_, row)| row).collect();
    (0..state.shared_offset)
        .chain(
            (0..state.shared_rows.len())
                .filter(|row| !eligible.contains(row))
                .map(|row| state.shared_offset + row),
        )
        .collect()
}

pub(super) fn dynamic_rows<const D: usize>(state: &State<D>) -> HashSet<usize> {
    state.cuts.as_ref().map_or_else(HashSet::new, |cuts| {
        cuts.rows
            .iter()
            .map(|&(_, row)| state.shared_offset + row)
            .chain(
                cuts.capacity_rows
                    .iter()
                    .map(|&(_, row)| state.shared_offset + row),
            )
            .collect()
    })
}

pub(super) fn remap_rows<const D: usize>(
    state: &mut State<D>,
    map: &[Option<usize>],
) -> Result<()> {
    let offset = state.shared_offset;
    let shared_map: Vec<_> = (0..state.shared_rows.len())
        .map(|row| {
            map.get(offset + row)
                .copied()
                .flatten()
                .and_then(|row| row.checked_sub(offset))
        })
        .collect();
    for oracle in &mut state.oracles {
        for entries in Arc::make_mut(&mut oracle.network.arc_entries) {
            entries.retain_mut(|(row, _)| match shared_map.get(*row).copied().flatten() {
                Some(next) => {
                    *row = next;
                    true
                }
                None => false,
            });
        }
        Arc::make_mut(&mut oracle.network.route_entries).retain_mut(|(row, _)| {
            match shared_map.get(*row).copied().flatten() {
                Some(next) => {
                    *row = next;
                    true
                }
                None => false,
            }
        });
        oracle.nonlinear_rows.retain_mut(|(cut, row)| {
            match shared_map.get(*row).copied().flatten() {
                Some(next) => {
                    *row = next;
                    true
                }
                None => {
                    oracle.solver.cuts.retire(*cut);
                    false
                }
            }
        });
    }
    let cuts = state
        .cuts
        .as_mut()
        .ok_or_else(|| invalid("missing separation state"))?;
    let mut memories = Vec::with_capacity(cuts.memories.len());
    let mut kept = Vec::with_capacity(cuts.rows.len());
    for ((cut, row), memory) in cuts.rows.drain(..).zip(cuts.memories.drain(..)) {
        if let Some(next) = shared_map.get(row).copied().flatten() {
            kept.push((cut, next));
            memories.push(memory);
        } else {
            cuts.kpath_indices.remove(&cut);
        }
    }
    cuts.rows = kept;
    cuts.memories = memories;
    for (_, row) in &mut cuts.capacity_rows {
        *row = shared_map
            .get(*row)
            .copied()
            .flatten()
            .ok_or_else(|| invalid("removed protected capacity row"))?;
    }
    let mut row = 0;
    state.shared_rows.retain(|_| {
        let keep = shared_map[row].is_some();
        row += 1;
        keep
    });
    let mut row = 0;
    state.curvature_rows.retain(|_| {
        let keep = map.get(row).is_some_and(Option::is_some);
        row += 1;
        keep
    });
    let mut row = 0;
    state.rhs.retain(|_| {
        let keep = map.get(row).is_some_and(Option::is_some);
        row += 1;
        keep
    });
    Ok(())
}

/// Vertices covered by a unit equality row through every arc entering them,
/// in row order, or None when the network rows are not a vertex partition.
fn partition_customers<const D: usize>(
    rows: &[Row],
    network: &OracleNetwork<D>,
) -> Option<Vec<usize>> {
    partition_rows(rows, network).map(|pairs| pairs.into_iter().map(|(_, v)| v).collect())
}

/// The network's (packing row, vertex) pairs in row order.
fn partition_rows<const D: usize>(
    rows: &[Row],
    network: &OracleNetwork<D>,
) -> Option<Vec<(usize, usize)>> {
    let g = &network.graph;
    let mut row_of = vec![None; g.vertices.len()];
    for (arc, entries) in g.arcs.iter().zip(network.arc_entries.iter()) {
        if arc.head == g.sink {
            continue;
        }
        let &[(row, value)] = entries.as_slice() else {
            return None;
        };
        if value != 1. || row_of[arc.head].is_some_and(|r| r != row) {
            return None;
        }
        row_of[arc.head] = Some(row);
    }
    let mut customers: Vec<(usize, usize)> = vec![];
    for v in 0..g.vertices.len() {
        if v == g.source || v == g.sink {
            continue;
        }
        let row = row_of[v]?;
        let r = rows.get(row)?;
        if r.sense != RowSense::Equal || r.rhs != 1. {
            return None;
        }
        customers.push((row, v));
    }
    customers.sort_unstable();
    if customers.windows(2).any(|w| w[0].0 == w[1].0) {
        return None;
    }
    Some(customers)
}

/// Install or refresh the separation state of a node before its solve.
/// A fresh node builds the separators; a resumed node keeps its inherited
/// separators and cuts and restarts the round counters for its depth.
pub(super) fn prepare<const D: usize>(state: &mut State<D>, options: &Options) -> Result<()> {
    let separation = &options.separation;
    for oracle in &state.oracles {
        let graph = &oracle.network.graph;
        if graph.max_route_arcs().is_some() && !graph.bounded_walks_are_vertex_simple() {
            if separation.relaxation_improvement_priority > 0. {
                return Err(invalid(
                    "dynamic NG strengthening is unsupported for bounded-walk solves",
                ));
            }
            if separation.rank_one.enabled
                || separation.capacity.is_some()
                || separation.kpath.is_some()
                || separation.general_kpath.is_some()
                || separation.knapsack.is_some()
            {
                return Err(invalid(
                    "bounded-walk separators require hard elementarity groups on every interior vertex",
                ));
            }
        }
    }
    let capacity_priority = if state.node_depth == 0 {
        CAPACITY_PRIORITY
    } else {
        1.
    };
    if let Some(cuts) = state.cuts.as_mut() {
        cuts.options = separation.rank_one.clone();
        cuts.stopped_by_tailing_off = false;
        cuts.capacity_priority = capacity_priority;
        cuts.relaxation_priority = separation.relaxation_improvement_priority;
        cuts.upper_bound = separation.upper_bound.unwrap_or(f64::INFINITY);
        cuts.rounds = 0;
        cuts.tailing = 0;
        cuts.separation_lp = f64::NAN;
        cuts.lp_before_cuts = f64::NAN;
        cuts.removed_columns.clear();
    } else {
        let relaxed_selection = state
            .oracles
            .iter()
            .any(|oracle| oracle.network.graph.selection_relaxed());
        // Relaxed selections in a multi-network master are handled directly
        // by the common controller, without constructing a cut separator.
        if !separation.is_active() && (!relaxed_selection || state.oracles.len() != 1) {
            return Ok(());
        }
        if state.oracles.len() != 1
            && (separation.route_recourse.is_some()
                || state.oracles.iter().any(|o| o.network.id < 0))
        {
            return Err(invalid(
                "several pricing networks do not support route recourse or negative graph IDs",
            ));
        }
        let mut network_customers = vec![];
        if separation.is_active() {
            let mut shared_rows = None;
            for oracle in &state.oracles {
                let pairs =
                    partition_rows(&state.shared_rows, &oracle.network).ok_or_else(|| {
                        invalid(
                            "cut separation requires unit rows partitioning the network vertices",
                        )
                    })?;
                let rows: Vec<usize> = pairs.iter().map(|&(row, _)| row).collect();
                if shared_rows.get_or_insert_with(|| rows.clone()) != &rows {
                    return Err(invalid(
                        "networks sharing rank-one cuts must cover the same packing rows",
                    ));
                }
                network_customers.push(pairs.into_iter().map(|(_, v)| v).collect());
            }
        } else {
            network_customers = state.oracles.iter().map(|_| vec![]).collect();
        }
        let customers = network_customers[0].clone();
        let lead_maps: Vec<Vec<Option<usize>>> = state
            .oracles
            .iter()
            .zip(&network_customers)
            .map(|(oracle, own)| {
                let g = &oracle.network.graph;
                let mut map = vec![None; g.vertices.len()];
                map[g.source] = Some(state.oracles[0].network.graph.source);
                map[g.sink] = Some(state.oracles[0].network.graph.sink);
                for (row, &v) in own.iter().enumerate() {
                    map[v] = customers.get(row).copied();
                }
                map
            })
            .collect();
        let pricing_graphs: Vec<_> = state
            .oracles
            .iter()
            .map(|oracle| oracle.solver.graph.clone())
            .collect();
        let oracle = &mut state.oracles[0];
        let graph = &oracle.network.graph;
        let routing_layout = |demands: usize| {
            graph.source == 0 && graph.sink == demands && graph.vertices.len() == demands + 1
        };
        let mut separator_options = separation.clone();
        let capacity_separator = match separator_options.capacity.take() {
            Some(config) => {
                if !routing_layout(config.demands.len()) {
                    return Err(invalid(
                        "capacity cuts require depot 0, customers 1..n, sink n+1",
                    ));
                }
                let mut separator = CapacitySeparator::new(config.demands, config.capacity)?;
                separator.parameters = config.parameters;
                if let Some(windows) = config.time_windows {
                    separator.set_time_windows(windows)?;
                }
                Some(separator)
            }
            None => None,
        };
        if separator_options.kpath.is_some() && separator_options.general_kpath.is_some() {
            return Err(invalid("select standalone or native k-path configuration"));
        }
        let kpath_separator = if let Some(config) = separator_options.general_kpath.take() {
            for graph in &pricing_graphs {
                let graph_id = graph.graph_id.max(0);
                let input = config
                    .graphs
                    .iter()
                    .find(|g| g.graph.id == graph_id)
                    .ok_or_else(|| invalid("native k-path configuration has no pricing graph"))?;
                if input.graph.source != graph.source as i32
                    || input.graph.sink != graph.sink as i32
                    || input
                        .graph
                        .vertices
                        .iter()
                        .any(|v| v.id < 0 || v.id as usize >= graph.vertices.len())
                {
                    return Err(invalid("native k-path pricing vertices/terminals differ"));
                }
                for arc in &graph.arcs {
                    if !input.graph.arcs.iter().any(|a| {
                        a.id == arc.id && a.tail == arc.tail as i32 && a.head == arc.head as i32
                    }) {
                        return Err(invalid("native k-path pricing arc incidence differs"));
                    }
                }
            }
            let mut separator = GeneralKPathSeparator::new(config.use_packing_sets);
            separator.parameters = config.parameters;
            separator.prepare(
                &config.graphs,
                &config.demands,
                config.capacity,
                config.two_path_resource,
            )?;
            Some(KPathSeparation::General {
                separator,
                graphs: config.graphs,
            })
        } else if let Some(config) = separator_options.kpath.take() {
            if !routing_layout(config.demands.len()) {
                return Err(invalid(
                    "k-path cuts require depot 0, customers 1..n, sink n+1",
                ));
            }
            let mut separator = KPathSeparator::new(config.demands, config.capacity)?;
            separator.parameters = config.parameters;
            if let Some(windows) = config.time_windows {
                separator.set_time_windows(windows)?;
            }
            Some(KPathSeparation::Standalone(separator))
        } else {
            None
        };
        let knapsack_separator = if let Some(mut config) = separator_options.knapsack.take() {
            let graph_id = graph.graph_id.max(0);
            let resources = graph.resources.len();
            let solver = &mut oracle.solver;
            if config.parameters.fenchel.is_some() {
                solver.knapsack_exact_coefficients = true;
            }
            solver.knapsack_resource_indices = config.resource_indices.clone();
            let indices = solver.knapsack_indices();
            if indices.values().any(|&r| r >= resources) {
                return Err(invalid(
                    "knapsack resource mapping exceeds solver coordinates",
                ));
            }
            for row in &config.rows {
                if let Some(&(resource, scale)) = row.graph_resources.get(&graph_id) {
                    if !indices.contains_key(&resource) || !scale.is_finite() || scale <= 0. {
                        return Err(invalid("knapsack original row resource/scaling"));
                    }
                }
            }
            for (name, value) in &options.pricing_options {
                let name = crate::core::options::Options::canonical_name(name).unwrap_or(name);
                match name {
                    "BALDESresConsKnapsackCutsMode" => {
                        config.parameters.use_rounding = *value > 0.;
                        config.parameters.max_one_k = value.abs() as i32;
                    }
                    "BALDESresConsKnapsackCutsMaxNumPerRound" => {
                        config.parameters.max_cuts_per_round = *value as usize
                    }
                    _ => {}
                }
            }
            crate::cuts::knapsack::KnapsackSeparator::with_parameters(config.parameters.clone())?;
            Some(ResourceKnapsackSeparator {
                rows: config.rows,
                parameters: config.parameters,
                ..Default::default()
            })
        } else {
            None
        };
        state.cuts = Some(CutState {
            options: separation.rank_one.clone(),
            resolved_memory_type: None,
            stopped_by_tailing_off: false,
            customers,
            lead_maps,
            network_customers,
            rows: vec![],
            memories: vec![],
            lp_before_cuts: f64::NAN,
            separation_lp: f64::NAN,
            rounds: 0,
            tailing: 0,
            capacity_separator,
            capacity_priority,
            capacity_rows: vec![],
            kpath_separator,
            kpath_indices: HashSet::new(),
            knapsack_separator,
            level: f64::INFINITY,
            relaxation_priority: separation.relaxation_improvement_priority,
            upper_bound: separation.upper_bound.unwrap_or(f64::INFINITY),
            recourse: separation.route_recourse.clone(),
            theta: vec![],
            theta_cuts: vec![],
            removed_columns: vec![],
        });
        if separation.route_recourse.is_some() {
            let vertices = state.oracles[0].network.graph.vertices.len();
            let mut theta = vec![None; vertices];
            for (index, variable) in state.master_variables.iter().enumerate() {
                if let Some(v) = variable
                    .id
                    .checked_sub(THETA_VARIABLE_BASE)
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|&v| v < vertices)
                {
                    theta[v] = Some(index);
                }
            }
            let g = &state.oracles[0].network.graph;
            if (0..vertices).any(|v| v != g.source && v != g.sink && theta[v].is_none()) {
                return Err(invalid(
                    "route recourse requires a fresh solve with Theta variables",
                ));
            }
            state.cuts.as_mut().unwrap().theta = theta;
        }
    }
    // Resource IDs are original-model IDs; every graph must configure its
    // solver coordinates before graph-specific knapsack separation/pricing.
    if let Some(config) = &options.separation.knapsack {
        for oracle in &mut state.oracles {
            let solver = &mut oracle.solver;
            solver.knapsack_resource_indices = config.resource_indices.clone();
            if config.parameters.fenchel.is_some() {
                solver.knapsack_exact_coefficients = true;
            }
            let indices = solver.knapsack_indices();
            if indices
                .values()
                .any(|&resource| resource >= solver.graph.resources.len())
            {
                return Err(invalid(
                    "knapsack resource mapping exceeds solver coordinates",
                ));
            }
            let graph_id = solver.graph.graph_id.max(0);
            for row in &config.rows {
                if let Some(&(resource, scale)) = row.graph_resources.get(&graph_id) {
                    if !indices.contains_key(&resource) || !scale.is_finite() || scale <= 0. {
                        return Err(invalid("knapsack original row resource/scaling"));
                    }
                }
            }
        }
    }
    let cuts = state.cuts.as_mut().unwrap();
    if let Some(separator) = cuts.capacity_separator.as_mut() {
        for (name, value) in &options.pricing_options {
            // Accept the legacy RCSP aliases of these registered options.
            let name = crate::core::options::Options::canonical_name(name).unwrap_or(name);
            match name {
                "BALDESuseCapacityCuts" => separator.parameters.enabled = *value != 0.,
                "BALDEScapCutsMaxNumPerRound" => {
                    separator.parameters.max_cuts_per_round = *value as usize
                }
                _ => {}
            }
        }
    }
    // Evaluation starts at the highest separator priority.
    cuts.level = cuts
        .separators()
        .first()
        .map_or(0., |&(priority, _)| priority);
    Ok(())
}

/// Each cut round first checks the incumbent gap (absolute 1e-6, relative 1e-9).
pub(super) fn separation_gap_closed(value: f64, incumbent: f64) -> bool {
    if !incumbent.is_finite() {
        return false;
    }
    let gap = incumbent - value;
    gap <= value.abs().max(incumbent.abs()) * 1e-10 + 1e-6 || gap / incumbent.abs() <= 1e-9
}

/// Root automatic rank-one memory decision rule (`BALDESrankOneCutsMemoryType = 0`).
/// Compares the vertex-memory pass (`node_bound`, `node_elapsed_secs`) against the
/// arc-memory pass (`arc_bound`, `arc_elapsed_secs`) relative to `incumbent` (or
/// `1e12` when nonfinite), returning `(prefer_arc, score)`.
pub(super) fn prefer_arc_memory(
    node_bound: f64,
    arc_bound: f64,
    incumbent: f64,
    node_elapsed_secs: f64,
    arc_elapsed_secs: f64,
) -> (bool, f64) {
    let upper = if incumbent.is_finite() {
        incumbent
    } else {
        1e12
    };
    let time_ratio = arc_elapsed_secs / node_elapsed_secs;
    let exponent = 10.0 * ((upper - arc_bound) / (upper - node_bound));
    let raw_score = (1024.0 / 2.0_f64.powf(exponent)) / time_ratio;
    let score = if raw_score.is_nan() { 0.0 } else { raw_score };
    let tol = node_bound.abs().max(arc_bound.abs()) * 1e-10 + 1e-6;
    let strictly_better = arc_bound - node_bound > tol;
    (strictly_better && score > 1.0, score)
}

/// Exact recourse excess of an integer route set, summed per Theta anchor
/// (each route's lowest interior vertex), as master-variable IDs and values;
/// None without route recourse. Visits are recovered from the arc heads, so
/// routes supplied without node lists are evaluated correctly.
pub(super) fn recourse_theta<const D: usize>(
    state: &State<D>,
    routes: &[FixedRoute],
) -> Result<Option<Vec<(i32, f64)>>> {
    let Some(recourse) = state.cuts.as_ref().and_then(|c| c.recourse.as_ref()) else {
        return Ok(None);
    };
    let mut theta = BTreeMap::<i32, f64>::new();
    for fixed in routes {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == fixed.route.network)
            .ok_or_else(|| invalid("recourse route network"))?;
        let graph = &oracle.network.graph;
        let customers: Vec<usize> = fixed
            .route
            .route
            .arcs
            .iter()
            .map(|id| {
                oracle
                    .index
                    .get(id)
                    .map(|&a| graph.arcs[a].head)
                    .ok_or_else(|| invalid("recourse route arc"))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|&v| v != graph.sink)
            .collect();
        let Some(&anchor) = customers.iter().min() else {
            continue;
        };
        let excess = recourse.0.excess(&customers)?;
        if !excess.is_finite() || excess < 0. {
            return Err(invalid(
                "route recourse excess must be finite and nonnegative",
            ));
        }
        let id = THETA_VARIABLE_BASE
            .checked_add(i32::try_from(anchor).map_err(|_| invalid("Theta vertex"))?)
            .ok_or_else(|| invalid("Theta vertex"))?;
        // The stored cut uses a rounded reciprocal. Paying `excess` exactly
        // can leave its exact binary64 row activity just below one. Increase
        // the charge until directed multiplication proves the encoded row.
        let mut charge = excess;
        if excess > 0. {
            let reciprocal = 1. / excess;
            if !reciprocal.is_finite() {
                return Err(invalid("route recourse reciprocal overflow"));
            }
            while crate::core::rounding::multiply_down(charge, reciprocal)
                .map_or(true, |value| value < 1.)
            {
                charge = f64::from_bits(charge.to_bits() + 1);
                if !charge.is_finite() {
                    return Err(invalid("route recourse charge overflow"));
                }
            }
        }
        let contribution = -crate::core::rounding::multiply_down(-charge, fixed.quantity)
            .ok_or_else(|| invalid("route recourse quantity overflow"))?;
        let value = theta.entry(id).or_default();
        *value = crate::core::rounding::add_up(*value, contribution)
            .ok_or_else(|| invalid("route recourse sum overflow"))?;
    }
    Ok(Some(theta.into_iter().collect()))
}

/// Theta has no finite upper bound, so even a one-ulp adverse dual residual
/// prevents a finite Lagrangian certificate. Reduce its exclusive positive
/// row prices until their directed sum is at most the unit objective cost.
/// This changes the pricing point, never the LP matrix or a reported primal.
pub(super) fn repair_theta_duals<const D: usize>(state: &State<D>, duals: &mut [f64]) {
    let Some(cuts) = state.cuts.as_ref().filter(|c| c.recourse.is_some()) else {
        return;
    };
    for &index in cuts.theta.iter().flatten() {
        let variable = &state.master_variables[index];
        if master_variables::reduced_cost_nonnegative(variable, 1., duals, state.shared_offset)
            == Some(true)
        {
            continue;
        }
        if variable.cost <= 0. || variable.entries.iter().any(|&(_, a)| a < 0.) {
            continue;
        }
        // A row shared with another static variable needs a joint projection.
        // The automatically generated Theta cuts have exclusive static support.
        if variable.entries.iter().any(|&(row, a)| {
            a != 0.
                && state.master_variables.iter().enumerate().any(|(other, v)| {
                    other != index && v.entries.iter().any(|&(r, b)| r == row && b != 0.)
                })
        }) {
            continue;
        }
        let upper = variable.entries.iter().try_fold(0., |sum, &(row, a)| {
            let price = duals[state.shared_offset + row].max(0.);
            let term = -crate::core::rounding::multiply_down(-a, price)?;
            crate::core::rounding::add_up(sum, term)
        });
        let Some(upper) = upper.filter(|&upper| upper > variable.cost) else {
            continue;
        };
        let mut factor = variable.cost / upper;
        while -crate::core::rounding::multiply_down(-factor, upper).unwrap_or(f64::NEG_INFINITY)
            > variable.cost
        {
            factor = f64::from_bits(factor.to_bits() - 1);
        }
        for &(row, a) in &variable.entries {
            let price = &mut duals[state.shared_offset + row];
            if a > 0. && *price > 0. {
                if let Some(repaired) = crate::core::rounding::multiply_down(*price, factor) {
                    *price = repaired.max(0.);
                }
            }
        }
    }
}

fn interior(route: &Route) -> Vec<usize> {
    route.nodes[1..route.nodes.len() - 1]
        .iter()
        .map(|&v| v as usize)
        .collect()
}

fn physical<const D: usize>(oracle: &Oracle<D>, route: &Route) -> Result<Vec<(usize, usize, i32)>> {
    route
        .arcs
        .iter()
        .map(|id| {
            let arc = &oracle.network.graph.arcs[*oracle
                .index
                .get(id)
                .ok_or_else(|| invalid("unknown separated route arc"))?];
            Ok((arc.tail, arc.head, arc.id))
        })
        .collect()
}

/// Rows separated in one round, installed together after every separator ran.
struct NewRows {
    base_shared: usize,
    pending: crate::master::formulation::PendingRows,
    rows: Vec<Row>,
}
impl NewRows {
    /// Queue a row with original right-hand side `rhs` and master right-hand
    /// side `rhs - fixed`; returns its shared-row index.
    fn push(&mut self, sense: RowSense, rhs: f64, fixed: f64, entries: Vec<(usize, f64)>) -> usize {
        self.pending.push(sense, rhs - fixed, entries);
        self.rows.push(Row { sense, rhs });
        self.base_shared + self.rows.len() - 1
    }
}

struct Round<'a, const D: usize> {
    state: &'a mut State<D>,
    cuts: &'a mut CutState,
    /// (master column, route tag) of loaded generated columns.
    tagged: Vec<(usize, usize)>,
    values: Vec<f64>,
}

impl<const D: usize> Round<'_, D> {
    fn route(&self, tag: usize) -> &Route {
        &self.state.routes[tag].route
    }
    /// Sum of quantity times coefficient over the fixed routes.
    fn fixed_sum(&self, mut coefficient: impl FnMut(&Route) -> Result<f64>) -> Result<f64> {
        let mut sum = 0.;
        for fixed in &self.state.fixed_routes {
            sum += coefficient(&fixed.route.route)? * fixed.quantity;
        }
        if !sum.is_finite() {
            return Err(invalid("fixed-route contribution overflow"));
        }
        Ok(sum)
    }
    /// Graph ID of a network for separation; one network keeps `-1` (its own).
    fn graph_id(&self, network: i32) -> i32 {
        if self.state.oracles.len() == 1 {
            -1
        } else {
            network
        }
    }

    fn oracle_index(&self, network: i32) -> Result<usize> {
        self.state
            .oracles
            .iter()
            .position(|o| o.network.id == network)
            .ok_or_else(|| invalid("unknown separated route network"))
    }

    /// Vertices of `network` in the first network's vertex space.
    fn to_lead(&self, network: i32, vertices: &[usize]) -> Result<Vec<usize>> {
        let map = &self.cuts.lead_maps[self.oracle_index(network)?];
        vertices
            .iter()
            .map(|&v| {
                map.get(v)
                    .copied()
                    .flatten()
                    .ok_or_else(|| invalid("vertex outside the shared packing rows"))
            })
            .collect()
    }

    /// First-network vertices in network `k`'s own vertex space.
    fn from_lead(&self, k: usize, vertices: &[usize]) -> Result<Vec<usize>> {
        vertices
            .iter()
            .map(|&v| {
                let row = self.cuts.network_customers[0]
                    .iter()
                    .position(|&c| c == v)
                    .ok_or_else(|| invalid("cut vertex outside the shared packing rows"))?;
                Ok(self.cuts.network_customers[k][row])
            })
            .collect()
    }

    /// Interior vertices of a route in the first network's vertex space.
    fn lead_interior(&self, route: &NetworkRoute) -> Result<Vec<usize>> {
        self.to_lead(route.network, &interior(&route.route))
    }

    /// Fractional columns in the first network's vertex space, for the
    /// separators defined over one routing layout (capacity, strong k-path).
    fn lead_fractional(&self, fractional: &[FractionalColumn]) -> Result<Vec<FractionalColumn>> {
        fractional
            .iter()
            .map(|c| {
                let mut c = c.clone();
                if c.graph_id >= 0 {
                    c.vertices = self.to_lead(c.graph_id, &c.vertices)?;
                    c.graph_id = -1;
                }
                Ok(c)
            })
            .collect()
    }

    fn oracle_of(&self, network: i32) -> Result<&Oracle<D>> {
        self.state
            .oracles
            .iter()
            .find(|o| o.network.id == network)
            .ok_or_else(|| invalid("unknown separated route network"))
    }

    /// Coefficient of registry cut `index` for a route of `network`, using
    /// that network's registry and graph.
    fn network_coefficient(
        &self,
        index: usize,
        network: i32,
        path: &[(usize, usize, i32)],
    ) -> Result<f64> {
        let oracle = self.oracle_of(network)?;
        Ok(oracle
            .solver
            .cuts
            .route_coefficient(&oracle.solver.graph, index, path, {
                let mut arcs: Vec<_> = path.iter().map(|arc| arc.2).collect();
                arcs.sort_unstable();
                oracle.enumerated.contains(&arcs)
            })? as f64)
    }

    /// Queue rows for registry cuts `before..` and refresh changed memories
    /// of installed registry rows. Returns whether any installed row changed.
    /// With several networks, new cuts of the first registry are mirrored into
    /// every other registry at the same index; each network prices them on
    /// its own graph support, and every column uses its network's coefficient.
    fn install_registry_cuts(
        &mut self,
        master: &mut ClpMaster,
        rows: &mut NewRows,
        before: usize,
        sense: impl Fn(&crate::cuts::CutRegistry, usize) -> RowSense,
        kpath: bool,
    ) -> Result<bool> {
        let after = self.state.oracles[0].solver.cuts.cuts.len();
        let added: Vec<_> = self.state.oracles[0].solver.cuts.cuts[before..after].to_vec();
        for oracle in self.state.oracles.iter_mut().skip(1) {
            let registry = &mut oracle.solver.cuts;
            // Cuts registered per network (strong k-path) are already there.
            if registry.cuts.len() == before {
                registry.cuts.extend(added.iter().cloned());
            } else if registry.cuts.len() != after {
                return Err(invalid("network cut registries are out of step"));
            }
        }
        // Resolve arc IDs once per route, rather than once per cut and route.
        let paths = self
            .tagged
            .iter()
            .map(|&(_, tag)| {
                let route = &self.state.routes[tag];
                Ok((
                    route.network,
                    physical(self.oracle_of(route.network)?, &route.route)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let fixed_paths = self
            .state
            .fixed_routes
            .iter()
            .map(|fixed| {
                let route = &fixed.route;
                Ok((
                    route.network,
                    physical(self.oracle_of(route.network)?, &route.route)?,
                    fixed.quantity,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        for index in before..after {
            let mut entries = vec![];
            for (&(column, _), (network, path)) in self.tagged.iter().zip(&paths) {
                let c = self.network_coefficient(index, *network, path)?;
                if c != 0. {
                    entries.push((column, c));
                }
            }
            let mut fixed = 0.;
            for (network, path, quantity) in &fixed_paths {
                fixed += self.network_coefficient(index, *network, path)? * quantity;
            }
            if !fixed.is_finite() {
                return Err(invalid("fixed-route contribution overflow"));
            }
            let registry = &self.state.oracles[0].solver.cuts;
            let cut = &registry.cuts[index];
            let shared = rows.push(sense(registry, index), cut.rhs as f64, fixed, entries);
            let memory = self
                .state
                .oracles
                .iter()
                .map(|oracle| {
                    oracle.solver.cuts.cuts[index]
                        .memory_for_graph(oracle.solver.graph.graph_id.max(0))
                })
                .collect();
            for oracle in &mut self.state.oracles {
                oracle.nonlinear_rows.push((index, shared));
            }
            self.cuts.rows.push((index, shared));
            self.cuts.memories.push(memory);
            if kpath {
                self.cuts.kpath_indices.insert(index);
            }
        }
        let mut changed = false;
        for k in 0..self.cuts.rows.len() {
            let (index, shared) = self.cuts.rows[k];
            if shared >= rows.base_shared {
                continue;
            }
            let memory: Vec<_> = self
                .state
                .oracles
                .iter()
                .map(|oracle| {
                    oracle.solver.cuts.cuts[index]
                        .memory_for_graph(oracle.solver.graph.graph_id.max(0))
                })
                .collect();
            if memory == self.cuts.memories[k] {
                continue;
            }
            changed = true;
            self.cuts.memories[k] = memory;
            let row = self.state.shared_offset + shared;
            for (&(column, _), (network, path)) in self.tagged.iter().zip(&paths) {
                let c = self.network_coefficient(index, *network, path)?;
                master.set_coefficient(row, column, c)?;
            }
        }
        Ok(changed)
    }

    fn separate_rank_one(
        &mut self,
        master: &mut ClpMaster,
        rows: &mut NewRows,
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let before = self.state.oracles[0].solver.cuts.cuts.len();
        let (first, rest) = self.state.oracles.split_at_mut(1);
        let solver = &mut first[0].solver;
        let mut graphs = vec![(&solver.graph, self.cuts.network_customers[0].as_slice())];
        for (oracle, customers) in rest.iter().zip(&self.cuts.network_customers[1..]) {
            graphs.push((&oracle.solver.graph, customers.as_slice()));
        }
        solver
            .cuts
            .separate_default_graphs(&graphs, &solver.options, fractional)?;
        let added = solver.cuts.cuts.len() - before;
        let changed = self.install_registry_cuts(
            master,
            rows,
            before,
            |registry, index| {
                if registry.is_covering(index) {
                    RowSense::GreaterEqual
                } else {
                    RowSense::LessEqual
                }
            },
            false,
        )?;
        Ok((added, changed))
    }

    fn separate_kpath(
        &mut self,
        master: &mut ClpMaster,
        rows: &mut NewRows,
        fractional: &[FractionalColumn],
    ) -> Result<(usize, bool)> {
        let mut solutions = vec![];
        let mut values = vec![];
        if matches!(
            self.cuts.kpath_separator,
            Some(KPathSeparation::General { .. })
        ) {
            for &(column, tag) in &self.tagged {
                if self.values[column] <= 1e-9 {
                    continue;
                }
                solutions.push(crate::cuts::separation::Solution {
                    graph_id: self
                        .oracle_of(self.state.routes[tag].network)?
                        .solver
                        .graph
                        .graph_id
                        .max(0),
                    enumerated: self
                        .oracle_of(self.state.routes[tag].network)?
                        .is_enumerated(self.route(tag)),
                    arc_ids: self.route(tag).arcs.clone(),
                });
                values.push(self.values[column]);
            }
            for fixed in &self.state.fixed_routes {
                solutions.push(crate::cuts::separation::Solution {
                    graph_id: self
                        .oracle_of(fixed.route.network)?
                        .solver
                        .graph
                        .graph_id
                        .max(0),
                    enumerated: true,
                    arc_ids: fixed.route.route.arcs.clone(),
                });
                values.push(fixed.quantity);
            }
        }
        let lead = self.lead_fractional(fractional)?;
        let before = self.state.oracles[0].solver.cuts.cuts.len();
        if let Some(KPathSeparation::Standalone(separator)) = self.cuts.kpath_separator.as_mut() {
            let found = separator.separate_fractional(&lead)?;
            // Each network registers the cut on its own vertices, so every
            // registry gains it at the same index.
            for cut in found {
                for k in 0..self.state.oracles.len() {
                    let mut own = cut.clone();
                    if k > 0 {
                        own.customers = self.from_lead(k, &cut.customers)?;
                        own.memory = self.from_lead(k, &cut.memory)?;
                    }
                    let solver = &mut self.state.oracles[k].solver;
                    let vertices = solver.graph.vertices.len();
                    let graph = solver.graph.graph_id.max(0);
                    solver
                        .cuts
                        .register_strong_k_path_cut(vertices, graph, &own, 0.)?;
                }
            }
            let added = self.state.oracles[0].solver.cuts.cuts.len() - before;
            let changed = self.install_registry_cuts(
                master,
                rows,
                before,
                |_, _| RowSense::GreaterEqual,
                true,
            )?;
            return Ok((added, changed));
        }
        let Some(separator) = self.cuts.kpath_separator.as_mut() else {
            return Ok((0, false));
        };
        let solver = &mut self.state.oracles[0].solver;
        match separator {
            KPathSeparation::Standalone(_) => unreachable!(),
            KPathSeparation::General { separator, graphs } => {
                for cut in separator.separate(&solutions, &values)? {
                    solver.cuts.register_general_k_path_cut(
                        graphs,
                        separator.preparation.use_packing_sets,
                        &cut,
                        0.,
                    )?;
                }
            }
        }
        let added = solver.cuts.cuts.len() - before;
        let changed =
            self.install_registry_cuts(master, rows, before, |_, _| RowSense::GreaterEqual, true)?;
        Ok((added, changed))
    }

    fn separate_capacity(
        &mut self,
        rows: &mut NewRows,
        fractional: &[FractionalColumn],
    ) -> Result<usize> {
        let lead = self.lead_fractional(fractional)?;
        let columns: Vec<_> = lead.iter().map(|c| (c.vertices.clone(), c.value)).collect();
        let Some(separator) = self.cuts.capacity_separator.as_mut() else {
            return Ok(0);
        };
        let found = separator.separate(&columns)?;
        let added = found.len();
        for cut in found {
            let mut entries = vec![];
            for &(column, tag) in &self.tagged {
                let coefficient = cut.coefficient(&self.lead_interior(&self.state.routes[tag])?);
                if coefficient != 0 {
                    entries.push((column, coefficient as f64));
                }
            }
            let sense = if cut.greater_equal() {
                RowSense::GreaterEqual
            } else {
                RowSense::LessEqual
            };
            let mut fixed = 0.;
            for route in &self.state.fixed_routes {
                fixed +=
                    cut.coefficient(&self.lead_interior(&route.route)?) as f64 * route.quantity;
            }
            if !fixed.is_finite() {
                return Err(invalid("fixed-route contribution overflow"));
            }
            let shared = rows.push(sense, cut.rhs as f64, fixed, entries);
            let lead_sink = self.state.oracles[0].network.graph.sink;
            for k in 0..self.state.oracles.len() {
                let map = self.cuts.lead_maps[k].clone();
                let network = &mut self.state.oracles[k].network;
                // Source and sink map to the depot (vertex 0 of the layout).
                let lead = |v: usize| match map[v] {
                    Some(l) if l == lead_sink => 0,
                    Some(l) => l,
                    None => usize::MAX,
                };
                let weights: Vec<f64> = network
                    .graph
                    .arcs
                    .iter()
                    .map(|arc| {
                        let (u, v) = (lead(arc.tail), lead(arc.head));
                        if u == usize::MAX || v == usize::MAX {
                            0.
                        } else {
                            cut.edge_coefficient(u, v)
                        }
                    })
                    .collect();
                for (entries, weight) in Arc::make_mut(&mut network.arc_entries)
                    .iter_mut()
                    .zip(weights)
                {
                    if weight != 0. {
                        entries.push((shared, weight));
                    }
                }
            }
            self.cuts.capacity_rows.push((cut.kind, shared));
        }
        Ok(added)
    }

    fn separate_knapsack(&mut self, rows: &mut NewRows) -> Result<usize> {
        let graph_of = |network: i32| {
            self.state
                .oracles
                .iter()
                .find(|o| o.network.id == network)
                .map_or(0, |o| o.solver.graph.graph_id.max(0))
        };
        let mut columns: Vec<_> = self
            .tagged
            .iter()
            .filter(|&&(column, _)| self.values[column] > 1e-9)
            .map(|&(column, tag)| KnapsackColumn {
                graph_id: graph_of(self.state.routes[tag].network),
                resources: self.route(tag).resources.clone(),
                value: self.values[column],
            })
            .collect();
        columns.extend(self.state.fixed_routes.iter().map(|fixed| KnapsackColumn {
            graph_id: graph_of(fixed.route.network),
            resources: fixed.route.route.resources.clone(),
            value: fixed.quantity,
        }));
        // Each network maps the knapsack resources of its own graph.
        let indices: BTreeMap<_, _> = self
            .state
            .oracles
            .iter()
            .map(|o| (o.solver.graph.graph_id.max(0), o.solver.knapsack_indices()))
            .collect();
        let Some(separator) = self.cuts.knapsack_separator.as_mut() else {
            return Ok(0);
        };
        let found = separator.separate_mapped(&columns, &indices)?;
        let count = found.len();
        for cut in found {
            let mut entries = vec![];
            for &(column, tag) in &self.tagged {
                let route = &self.state.routes[tag];
                let solver = &self.oracle_of(route.network)?.solver;
                let coefficient = solver.knapsack_coefficient(&cut, &route.route.resources);
                if coefficient != 0 {
                    entries.push((column, coefficient as f64));
                }
            }
            let mut fixed = 0.;
            for route in &self.state.fixed_routes {
                let solver = &self.oracle_of(route.route.network)?.solver;
                fixed += solver.knapsack_coefficient(&cut, &route.route.route.resources) as f64
                    * route.quantity;
            }
            if !fixed.is_finite() {
                return Err(invalid("fixed-route contribution overflow"));
            }
            let shared = rows.push(RowSense::LessEqual, cut.cut.rhs as f64, fixed, entries);
            for oracle in &mut self.state.oracles {
                oracle
                    .knapsack_rows
                    .push((oracle.solver.cuts.knapsack_cuts.len(), shared));
                oracle.solver.cuts.knapsack_cuts.push(cut.clone());
            }
        }
        if count > 0 {
            for k in 0..self.state.oracles.len() {
                if super::resource_rows::requires_forward(&self.state.oracles[k]) {
                    self.state.oracles[k]
                        .solver
                        .set_option("BALDESuseBidirectionalSearch", 0.)?;
                }
            }
        }
        Ok(count)
    }

    /// Whether the master solution uses a route breaking a relaxed selection.
    fn uses_improper_selection(&self) -> bool {
        self.tagged.iter().any(|&(column, tag)| {
            if self.values[column] <= 1e-9 {
                return false;
            }
            let route = &self.state.routes[tag];
            let Ok(oracle) = self.oracle_of(route.network) else {
                return false;
            };
            let Some(kernel) = oracle
                .network
                .graph
                .coupled_kernel()
                .filter(|kernel| kernel.selection_relaxed())
            else {
                return false;
            };
            route
                .route
                .arcs
                .iter()
                .map(|id| oracle.index.get(id).copied())
                .collect::<Option<Vec<_>>>()
                .is_some_and(|arcs| !kernel.selection_proper(&arcs))
        })
    }

    /// Dynamic NG strengthening and relaxed-selection tightening; marks the
    /// master columns the stronger relaxation forbids for removal.
    fn improve_relaxation(&mut self, rank_one_in_master: bool) -> Result<bool> {
        let mut changed = false;
        for index in 0..self.state.oracles.len() {
            let network = self.state.oracles[index].network.id;
            let paths: Vec<_> = self
                .tagged
                .iter()
                .filter(|&&(column, tag)| {
                    self.values[column] > 1e-9 && self.state.routes[tag].network == network
                })
                .map(|&(column, tag)| crate::graph::relaxation::FractionalPath {
                    arcs: self.route(tag).arcs.clone(),
                    value: self.values[column],
                })
                .collect();
            let solver = &mut self.state.oracles[index].solver;
            let selection = solver.strengthen_selection(&paths)?;
            let option = |name: &str| solver.options.get(name);
            let mode = option("BALDESdynamicNGmode")? as i32;
            let ng = if mode == 0 || (mode < 0 && rank_one_in_master) {
                0
            } else {
                let controls = crate::graph::relaxation::Controls {
                    mode: mode.abs(),
                    max_vertex_size: option("BALDESmaxNGneighbourhoodSize")?.max(0.) as usize,
                    max_average_size: option("BALDESmaxNGaverNeighbourhoodSize")?.max(0.) as usize,
                    max_columns: option("BALDESmaxNumOfColumnsInDynamicNG")?.max(0.) as usize,
                    max_cycle_size: option("BALDESmaxCycleSizeInDynamicNG")? as i32,
                    mirror: false,
                };
                solver.strengthen_ng(&paths, &controls)?.columns_forbidden
            };
            changed |= selection > 0 || ng > 0;
        }
        if !changed {
            return Ok(false);
        }
        self.cuts.removed_columns = self.forbidden_columns()?;
        Ok(true)
    }

    fn forbidden_columns(&self) -> Result<Vec<usize>> {
        let mut removed = vec![];
        for &(column, tag) in &self.tagged {
            let network_route = &self.state.routes[tag];
            let solver = &self.oracle_of(network_route.network)?.solver;
            let route = &network_route.route;
            if !solver.route_satisfies_relaxation(&route.nodes, &route.arcs)? {
                removed.push(column);
            }
        }
        Ok(removed)
    }

    fn excess(&self, route: &Route) -> Result<f64> {
        let Some(recourse) = &self.cuts.recourse else {
            return Ok(0.);
        };
        let excess = recourse.0.excess(&interior(route))?;
        if !excess.is_finite() || excess < 0. {
            return Err(invalid(
                "route recourse excess must be finite and nonnegative",
            ));
        }
        Ok(excess)
    }

    /// Master column of the positive component of master variable `index`.
    fn variable_column(&self, tags: &[i32], index: usize) -> Option<usize> {
        tags.iter()
            .position(|&tag| master_variables::decode(tag) == Some((index, 1.)))
    }

    /// Integral routes (master value one, or fixed) whose anchor Theta is
    /// below their excess and whose cut is not installed yet.
    fn theta_violations(&self, tags: &[i32]) -> Result<Vec<(Vec<i32>, usize, f64, usize)>> {
        if self.cuts.recourse.is_none() {
            return Ok(vec![]);
        }
        let mut integral: Vec<&Route> = self
            .tagged
            .iter()
            .filter(|&&(column, _)| self.values[column] >= 1. - 1e-6)
            .map(|&(_, tag)| self.route(tag))
            .collect();
        integral.extend(self.state.fixed_routes.iter().map(|f| &f.route.route));
        let mut violated: Vec<(Vec<i32>, usize, f64, usize)> = vec![];
        for route in integral {
            let Some(&anchor) = interior(route).iter().min() else {
                continue;
            };
            let variable = self
                .cuts
                .theta
                .get(anchor)
                .copied()
                .flatten()
                .ok_or_else(|| invalid("route recourse anchor has no Theta variable"))?;
            let column = self
                .variable_column(tags, variable)
                .ok_or_else(|| invalid("Theta variable is not in the master"))?;
            let excess = self.excess(route)?;
            if self.values[column] >= excess - 1e-6 * (1. + excess) {
                continue;
            }
            let mut arcs = route.arcs.clone();
            arcs.sort_unstable();
            arcs.dedup();
            if !self.cuts.theta_cuts.contains(&arcs) && !violated.iter().any(|(a, ..)| *a == arcs) {
                violated.push((arcs, variable, excess, route.arcs.len()));
            }
        }
        Ok(violated)
    }

    /// Cut off every integral route whose anchor Theta is below its excess.
    fn separate_theta(&mut self, rows: &mut NewRows, tags: &[i32]) -> Result<usize> {
        let violated = self.theta_violations(tags)?;
        for (arcs, variable, excess, length) in &violated {
            let count = |route: &Route| {
                -(route
                    .arcs
                    .iter()
                    .filter(|id| arcs.binary_search(id).is_ok())
                    .count() as f64)
            };
            let column = self.variable_column(tags, *variable).unwrap();
            let mut entries = vec![(column, 1. / excess)];
            for &(column, tag) in &self.tagged {
                let c = count(self.route(tag));
                if c != 0. {
                    entries.push((column, c));
                }
            }
            let fixed = self.fixed_sum(|route| Ok(count(route)))?;
            let shared = rows.push(RowSense::GreaterEqual, 1. - *length as f64, fixed, entries);
            // Pricing charges every arc of the cut route with its dual.
            let network = &mut self.state.oracles[0].network;
            let ids: Vec<i32> = network.graph.arcs.iter().map(|a| a.id).collect();
            for (entries, id) in Arc::make_mut(&mut network.arc_entries).iter_mut().zip(ids) {
                if arcs.binary_search(&id).is_ok() {
                    entries.push((shared, -1.));
                }
            }
            self.state.master_variables[*variable]
                .entries
                .push((shared, 1. / excess));
            self.cuts.theta_cuts.push(arcs.clone());
        }
        Ok(violated.len())
    }

    fn new_rows(&self, master: &ClpMaster) -> Result<NewRows> {
        let physical_rows = master.rhs().len();
        let base_shared = self.state.shared_rows.len();
        if self.state.shared_offset + base_shared != physical_rows {
            return Err(invalid("cut separation requires contiguous shared rows"));
        }
        Ok(NewRows {
            base_shared,
            pending: crate::master::formulation::PendingRows::new(physical_rows),
            rows: vec![],
        })
    }

    fn install(&mut self, master: &mut ClpMaster, rows: NewRows) -> Result<()> {
        rows.pending.install(master)?;
        let count = rows.rows.len();
        self.state.shared_rows.extend(rows.rows);
        self.state
            .curvature_rows
            .extend((0..count).map(|_| cg::CurvatureRow {
                class: cg::CurvatureRowClass::Dynamic,
                active: true,
            }));
        self.state.rhs = master.rhs.clone();
        Ok(())
    }

    fn run(&mut self, master: &mut ClpMaster, tags: &[i32]) -> Result<usize> {
        let value = master.objective();
        let rank_one_in_master = !self.cuts.rows.is_empty();
        // A solution using an improper route of a relaxed selection never
        // ends the node: tighten (or drop stale columns) before any stop test.
        if self.state.oracles[0].network.graph.selection_relaxed() && self.uses_improper_selection()
        {
            if !self.improve_relaxation(rank_one_in_master)? {
                self.cuts.removed_columns = self.forbidden_columns()?;
            }
            return Ok(1);
        }
        // Integral routes must pay their exact recourse before the node may
        // stop: Theta cuts precede every stopping and tailing-off test.
        if self.cuts.recourse.is_some() {
            let mut rows = self.new_rows(master)?;
            let theta = self.separate_theta(&mut rows, tags)?;
            self.install(master, rows)?;
            if theta > 0 {
                return Ok(theta);
            }
        }
        let depth = i32::try_from(self.state.node_depth).unwrap_or(i32::MAX);
        let stop = self.state.oracles[0]
            .solver
            .column_generation_terminated(0, depth)?;
        let incumbent = self
            .state
            .pricing_witness
            .as_ref()
            .map_or(f64::INFINITY, |w| w.0)
            .min(self.cuts.upper_bound);
        if self.cuts.rounds == 0 {
            self.cuts.lp_before_cuts = value;
        } else {
            // The gain is relative to the gap to the incumbent when that gap
            // is at most a tenth of the bound, else to a tenth of the bound.
            let previous = self.cuts.separation_lp;
            let upper = if incumbent.is_finite() {
                incumbent
            } else {
                1e12
            };
            let gap = upper - previous;
            let tenth = (0.1 * previous.abs()).max(1e-9);
            let denominator = if gap - (gap.abs().max(tenth) * 1e-10 + 1e-6) > tenth {
                tenth
            } else {
                gap
            };
            let gain = (value - previous) / denominator;
            let min_rounds = self.cuts.options.min_rounds_before_stop_by_sp;
            if self.cuts.rank_one_active() && stop && self.cuts.rounds >= min_rounds {
                return Ok(0);
            }
            let threshold = self.cuts.options.tailing_off_threshold as f32 as f64;
            if gain < threshold - (threshold.abs().max(gain.abs()) * 1e-10 + 1e-6) {
                let solver = &mut self.state.oracles[0].solver;
                if self.cuts.options.enabled
                    && self.cuts.level <= RANK_ONE_PRIORITY
                    && !solver.cuts.separation_phase(&solver.options)?.1
                {
                    solver.cuts.increase_phase(&solver.options)?;
                } else {
                    self.cuts.tailing += 1;
                }
                // Tailing off at a level moves to the next lower priority,
                // and stops cut generation at the lowest one.
                if self.cuts.tailing >= self.cuts.options.tailing_off_counter
                    && self.cuts.rounds >= min_rounds
                {
                    match self.cuts.next_level() {
                        Some(level) => {
                            self.cuts.level = level;
                            self.cuts.tailing = 0;
                        }
                        None => {
                            self.cuts.stopped_by_tailing_off = true;
                            return Ok(0);
                        }
                    }
                }
            }
        }
        self.cuts.separation_lp = value;
        if separation_gap_closed(value, incumbent) {
            return Ok(0);
        }
        // Separation sees the complete solution, including fixed routes.
        let mut fractional = vec![];
        for &(column, tag) in &self.tagged {
            if self.values[column] > 1e-9 {
                fractional.push(FractionalColumn {
                    vertices: interior(self.route(tag)),
                    value: self.values[column],
                    graph_id: self.graph_id(self.state.routes[tag].network),
                    enumerated: self
                        .oracle_of(self.state.routes[tag].network)?
                        .is_enumerated(self.route(tag)),
                });
            }
        }
        for fixed in &self.state.fixed_routes {
            fractional.push(FractionalColumn {
                vertices: interior(&fixed.route.route),
                value: fixed.quantity,
                graph_id: self.graph_id(fixed.route.network),
                enumerated: true,
            });
        }
        let mut rows = self.new_rows(master)?;
        let mut level = self.cuts.level;
        let mut added = 0;
        let mut success = false;
        for (priority, separator) in self.cuts.separators() {
            if priority < level {
                if success {
                    break;
                }
                level = priority;
            }
            let (count, changed) = match separator {
                Separator::Capacity => (self.separate_capacity(&mut rows, &fractional)?, false),
                Separator::KPath => self.separate_kpath(master, &mut rows, &fractional)?,
                Separator::Knapsack => (self.separate_knapsack(&mut rows)?, false),
                Separator::RankOne => self.separate_rank_one(master, &mut rows, &fractional)?,
                Separator::Relaxation => (0, self.improve_relaxation(rank_one_in_master)?),
            };
            added += count;
            success |= count > 0 || changed;
        }
        self.cuts.level = level;
        self.install(master, rows)?;
        if success {
            self.cuts.rounds += 1;
        }
        Ok(if added > 0 { added } else { success as usize })
    }
}

/// Whether a positive generated column violates its network's original
/// selection, independent of the current remembered relaxation.
fn has_improper_selection<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    tags: &[i32],
) -> Result<bool> {
    if !state
        .oracles
        .iter()
        .any(|oracle| oracle.network.graph.selection_relaxed())
    {
        return Ok(false);
    }
    for (column, &tag) in tags.iter().enumerate() {
        if tag < 0 || master.values().get(column).copied().unwrap_or(0.) <= 1e-9 {
            continue;
        }
        let route = state
            .routes
            .get(tag as usize)
            .ok_or_else(|| invalid("unknown selection route"))?;
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .ok_or_else(|| invalid("unknown selection network"))?;
        let Some(kernel) = oracle
            .network
            .graph
            .coupled_kernel()
            .filter(|k| k.selection_relaxed())
        else {
            continue;
        };
        let arcs = route
            .route
            .arcs
            .iter()
            .map(|id| oracle.index.get(id).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("unknown selection arc"))?;
        if !kernel.selection_proper(&arcs) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Tighten relaxed selections on each network and remove loaded columns that
/// the strengthened relaxation no longer permits. Used when no single-network
/// cut separator is installed.
pub(super) fn tighten_selections<const D: usize>(
    state: &mut State<D>,
    master: &ClpMaster,
    tags: &[i32],
) -> Result<(usize, Vec<usize>)> {
    let mut removed = Vec::new();
    let mut changed = false;
    for oracle in &mut state.oracles {
        let Some(kernel) = oracle
            .network
            .graph
            .coupled_kernel()
            .filter(|k| k.selection_relaxed())
        else {
            continue;
        };
        let mut paths = Vec::new();
        let mut improper = false;
        let mut loaded = Vec::new();
        for (column, &tag) in tags.iter().enumerate() {
            if tag < 0 {
                continue;
            }
            let route = state
                .routes
                .get(tag as usize)
                .ok_or_else(|| invalid("unknown selection route"))?;
            if route.network != oracle.network.id {
                continue;
            }
            loaded.push((column, &route.route));
            let value = master.values().get(column).copied().unwrap_or(0.);
            if value <= 1e-9 {
                continue;
            }
            let arcs = route
                .route
                .arcs
                .iter()
                .map(|id| oracle.index.get(id).copied())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("unknown selection arc"))?;
            improper |= !kernel.selection_proper(&arcs);
            paths.push(crate::graph::relaxation::FractionalPath {
                arcs: route.route.arcs.clone(),
                value,
            });
        }
        if !improper {
            continue;
        }
        let strengthened = oracle.solver.strengthen_selection(&paths)?;
        let before = removed.len();
        for (column, route) in loaded {
            if !oracle
                .solver
                .route_satisfies_relaxation(&route.nodes, &route.arcs)?
            {
                removed.push(column);
            }
        }
        if strengthened == 0 && removed.len() == before {
            return Err(incomplete("relaxed selection could not be tightened"));
        }
        changed = true;
    }
    removed.sort_unstable();
    removed.dedup();
    Ok((usize::from(changed), removed))
}

/// One separation round on the current master solution. Returns the number
/// of rows added, or 1 when only cut memories or the relaxation changed.
pub(super) fn separate<const D: usize>(
    state: &mut State<D>,
    master: &mut ClpMaster,
    tags: &[i32],
) -> Result<usize> {
    let Some(mut cuts) = state.cuts.take() else {
        return Ok(0);
    };
    let values = master.values().to_vec();
    let tagged = tags
        .iter()
        .enumerate()
        .filter(|(_, &t)| t >= 0)
        .map(|(i, &t)| (i, t as usize))
        .collect();
    let result = Round {
        state,
        cuts: &mut cuts,
        tagged,
        values,
    }
    .run(master, tags);
    state.cuts = Some(cuts);
    result
}

/// Whether an integral route still needs its Theta cut. Such a solution
/// must be separated regardless of cut-round limits and tailing off.
pub(super) fn requires_separation<const D: usize>(
    state: &mut State<D>,
    master: &ClpMaster,
    tags: &[i32],
) -> Result<bool> {
    if master.solved() && has_improper_selection(state, master, tags)? {
        return Ok(true);
    }
    if !master.solved() || state.cuts.as_ref().is_none_or(|c| c.recourse.is_none()) {
        return Ok(false);
    }
    let mut cuts = state.cuts.take().unwrap();
    let values = master.values().to_vec();
    let tagged = tags
        .iter()
        .enumerate()
        .filter(|(_, &t)| t >= 0)
        .map(|(i, &t)| (i, t as usize))
        .collect();
    let result = Round {
        state,
        cuts: &mut cuts,
        tagged,
        values,
    }
    .theta_violations(tags)
    .map(|v| !v.is_empty());
    state.cuts = Some(cuts);
    result
}

/// Whether automatic separation is installed (or a relaxed selection must
/// still be tightened before the node may end).
pub(super) fn separates<const D: usize>(state: &State<D>) -> bool {
    state
        .cuts
        .as_ref()
        .is_some_and(|cuts| !cuts.separators().is_empty() || cuts.recourse.is_some())
        || state
            .oracles
            .iter()
            .any(|o| o.network.graph.selection_relaxed())
}

/// Installed cut rows by enabled family, for progress logging.
pub(super) fn cut_counts<const D: usize>(state: &State<D>) -> Vec<(&'static str, usize)> {
    let Some(cuts) = state.cuts.as_ref() else {
        return vec![];
    };
    let mut counts = vec![];
    if cuts.options.enabled {
        let rank_one = cuts
            .rows
            .iter()
            .filter(|(index, _)| !cuts.kpath_indices.contains(index))
            .count();
        counts.push(("Rank1", rank_one));
    }
    if let Some(separator) = &cuts.capacity_separator {
        if separator.parameters.enabled {
            let two_path = cuts
                .capacity_rows
                .iter()
                .filter(|(kind, _)| *kind == CapacityKind::TwoPath)
                .count();
            counts.push(("Capacity", cuts.capacity_rows.len() - two_path));
            if separator.parameters.use_two_path {
                counts.push(("2Path", two_path));
            }
        }
    }
    if cuts.kpath_separator.is_some() {
        counts.push(("KPath", cuts.kpath_indices.len()));
    }
    if let Some(separator) = &cuts.knapsack_separator {
        let knapsack_cuts = &state.oracles[0].solver.cuts.knapsack_cuts;
        let fenchel = knapsack_cuts
            .iter()
            .filter(|cut| cut.cut.kind == KnapsackKind::Fenchel)
            .count();
        if separator.parameters.use_rounding || separator.parameters.max_one_k > 0 {
            counts.push(("Knapsack", knapsack_cuts.len() - fenchel));
        }
        if separator.parameters.fenchel.is_some() {
            counts.push(("Fenchel", fenchel));
        }
    }
    counts
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/model/separation_tests.rs"]
mod tests;
