//! Explicit original-variable mappings across pricing networks and master IDs.
//! These are model registrations, never inferred from display names or route tags.
use super::*;
mod batch;
use crate::master::preprocessing::Bounds;
pub(super) use batch::{values, values_except};

/// Integer contribution of one network to an original aggregate variable.
/// Arc IDs belong to this network; omitted arcs have zero coefficient.
/// Half-integer arc coefficients require a parity proof that every complete
/// graph path has integer contribution (including repeated-arc walks).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappedNetworkVariable {
    pub network: i32,
    #[serde(default)]
    pub arcs: Vec<(i32, f64)>,
    /// Intrinsic deterministic custom/cumulative terms. Each descriptor must have
    /// a complete exact-integer path proof in this graph domain.
    #[serde(default)]
    pub activities: Vec<(usize, f64)>,
    #[serde(default)]
    pub per_route: f64,
}

/// A named integer model variable assembled from several pricing networks and
/// independent integer master variables. Larger priorities are considered first;
/// equal-priority candidates prefer the most fractional value, then registration
/// order, both compared within the reference's tolerance band.
/// This registration does not assert that a continuous master value is integer.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappedBranchVariable {
    pub name: String,
    #[serde(default)]
    pub priority: f64,
    #[serde(default)]
    pub networks: Vec<MappedNetworkVariable>,
    #[serde(default)]
    pub masters: Vec<(i32, f64)>,
}

/// Candidate selection within a registered branching family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BranchSelectionRule {
    FirstFound,
    HighestPriority,
    #[default]
    MostFractional,
    LeastFractional,
    FractionWeightedPriority,
    ClosestToUpper,
    ClosestToLower,
}

/// Separate family selection from an original variable's priority. The mapped
/// variable's `priority` remains the family level; equal primary keys prefer a
/// smaller `default_factor`, then the earlier registration.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BranchSelection {
    pub family: String,
    pub rule: BranchSelectionRule,
    pub variable_priority: f64,
    pub default_factor: f64,
}

impl Default for BranchSelection {
    fn default() -> Self {
        Self {
            family: "mapped".into(),
            rule: BranchSelectionRule::MostFractional,
            variable_priority: 1.,
            default_factor: 1.,
        }
    }
}

pub(crate) fn validate_selection(
    variables: &[MappedBranchVariable],
    selection: &BTreeMap<String, BranchSelection>,
) -> Result<()> {
    let variables: HashMap<_, _> = variables.iter().map(|v| (v.name.as_str(), v)).collect();
    let mut families = HashMap::new();
    for (name, metadata) in selection {
        let variable = variables
            .get(name.as_str())
            .ok_or_else(|| invalid("unknown branching selection variable"))?;
        if metadata.family.trim().is_empty()
            || !metadata.variable_priority.is_finite()
            || !metadata.default_factor.is_finite()
        {
            return Err(invalid(
                "branching selection requires a family and finite factors",
            ));
        }
        if let Some((priority, rule)) =
            families.insert(metadata.family.as_str(), (variable.priority, metadata.rule))
        {
            if priority != variable.priority || rule != metadata.rule {
                return Err(invalid(
                    "one branching family requires a single level and selection rule",
                ));
            }
        }
    }
    Ok(())
}

/// Validate aggregate-family admission controls independently of selection keys.
pub(crate) fn validate_aggregate_admission<'a>(
    variables: impl IntoIterator<Item = &'a MappedBranchVariable>,
    admission: &BTreeMap<String, f64>,
) -> Result<()> {
    if admission.is_empty() {
        return Ok(());
    }
    let names: HashSet<_> = variables.into_iter().map(|v| v.name.as_str()).collect();
    if admission.iter().any(|(name, &threshold)| {
        !names.contains(name.as_str()) || !valid_aggregate_threshold(threshold)
    }) {
        return Err(invalid(
            "aggregate admission requires known mapped variables and finite thresholds in (0, 0.5]",
        ));
    }
    Ok(())
}

pub(super) fn valid_aggregate_threshold(threshold: f64) -> bool {
    threshold.is_finite() && threshold > 0. && threshold <= 0.5
}

/// Aggregate families admit the signed fractional part between inclusive
/// parameter bounds. This is independent of ordinary variable admission and
/// the tree tolerance; negative noninteger sums are omitted by this family.
pub(crate) fn aggregate_fractional(value: f64, threshold: f64) -> bool {
    if !value.is_finite()
        || value.abs() >= 9_007_199_254_740_992.
        || !valid_aggregate_threshold(threshold)
    {
        return false;
    }
    let fraction = value.fract();
    fraction >= threshold && fraction <= 1. - threshold
}

fn selection_exceeds(a: f64, b: f64) -> bool {
    a - (1e-6 + a.abs().max(b.abs()) * 1e-10) > b
}

fn snap_fraction(value: f64) -> f64 {
    if value.abs() <= 1e-6 {
        0.
    } else {
        value
    }
}

// Directed distances use the same tolerant integer endpoints as the generator,
// including negative activities and values close to an integer.
fn directed_fractions(value: f64) -> (f64, f64) {
    let offset = value * 1e-10;
    let mut upper = (value - offset - 1e-6).ceil();
    if upper >= value + 1. {
        upper -= 1.;
    }
    upper = snap_fraction(upper);
    let mut lower = (value + offset + 1e-6).floor();
    if value - 1. + offset + 1e-6 > lower {
        lower += 1.;
    }
    lower = snap_fraction(lower);
    (snap_fraction(value - lower), snap_fraction(upper - value))
}

/// Compare candidate metadata after distinct families' levels have tied.
/// Different family rules go directly to the common default factor.
pub(crate) fn selection_before(
    a: &BranchSelection,
    a_value: f64,
    a_order: usize,
    b: &BranchSelection,
    b_value: f64,
    b_order: usize,
) -> bool {
    if a.rule == b.rule {
        use BranchSelectionRule::*;
        if a.rule == FirstFound {
            return a_order < b_order;
        }
        let fraction = |value: f64| {
            let (lower, upper) = directed_fractions(value);
            lower.min(upper)
        };
        let (a_key, b_key, descending) = match a.rule {
            FirstFound => unreachable!(),
            HighestPriority => (a.variable_priority, b.variable_priority, true),
            MostFractional => (fraction(a_value), fraction(b_value), true),
            LeastFractional => (fraction(a_value), fraction(b_value), false),
            FractionWeightedPriority => (
                a.variable_priority * fraction(a_value),
                b.variable_priority * fraction(b_value),
                true,
            ),
            ClosestToUpper => (
                directed_fractions(a_value).1,
                directed_fractions(b_value).1,
                false,
            ),
            ClosestToLower => (
                directed_fractions(a_value).0,
                directed_fractions(b_value).0,
                false,
            ),
        };
        if selection_exceeds(a_key, b_key) {
            return descending;
        }
        if selection_exceeds(b_key, a_key) {
            return !descending;
        }
    }
    if selection_exceeds(a.default_factor, b.default_factor) {
        return false;
    }
    if selection_exceeds(b.default_factor, a.default_factor) {
        return true;
    }
    a_order < b_order
}

fn integer_coefficient(value: f64) -> bool {
    value.is_finite() && value.fract() == 0. && value.abs() <= 1_048_576.
}

/// A parity potential certifies half-integer arc expressions on all walks,
/// including cycles. Each arc parity must equal the difference of its endpoint
/// potentials; equal source/sink potentials make every complete path integral.
fn integral_path_parity<const D: usize>(graph: &Graph<D>, arcs: &[(i32, f64)]) -> bool {
    if arcs.iter().all(|(_, value)| value.fract() == 0.) {
        return true;
    }
    // Fresh tree registrations are checked before full problem validation.
    if graph.source >= graph.vertices.len()
        || graph.sink >= graph.vertices.len()
        || graph
            .arcs
            .iter()
            .any(|arc| arc.tail >= graph.vertices.len() || arc.head >= graph.vertices.len())
    {
        return false;
    }
    let values: HashMap<_, _> = arcs.iter().copied().collect();
    let mut outgoing = vec![vec![]; graph.vertices.len()];
    for arc in &graph.arcs {
        let value = values.get(&arc.id).copied().unwrap_or(0.);
        let parity = u8::from((value * 2.).rem_euclid(2.) == 1.);
        outgoing[arc.tail].push((arc.head, parity));
    }
    let mut potentials = vec![None; graph.vertices.len()];
    potentials[graph.source] = Some(0);
    let mut pending = vec![graph.source];
    while let Some(tail) = pending.pop() {
        let potential = potentials[tail].unwrap();
        for &(head, parity) in &outgoing[tail] {
            let expected = potential ^ parity;
            match potentials[head] {
                Some(value) if value != expected => return false,
                Some(_) => {}
                None => {
                    potentials[head] = Some(expected);
                    pending.push(head);
                }
            }
        }
    }
    potentials[graph.sink].map_or(true, |potential| potential == 0)
}

fn exact_coefficient(values: impl Iterator<Item = f64>) -> Result<f64> {
    let mut sum = crate::master::numerics::ExactSum::default();
    for value in values {
        sum.add(value)
            .ok_or_else(|| invalid("mapped coefficient sum overflow"))?;
    }
    let value = sum
        .upper()
        .ok_or_else(|| invalid("mapped coefficient sum overflow"))?;
    sum.add(-value)
        .ok_or_else(|| invalid("mapped coefficient sum overflow"))?;
    if sum.sign() != 0. {
        return Err(invalid(
            "mapped row coefficient must be exactly representable",
        ));
    }
    Ok(value)
}

impl MappedBranchVariable {
    /// Map selected network and static-master contributions of an existing
    /// original shared row. In particular, an edge/vertex packing row can be
    /// restricted to one pricing network to register its assignment variable.
    /// The selected IDs are explicit; no model meaning is inferred from names.
    pub fn from_shared_row<const D: usize>(
        name: impl Into<String>,
        priority: f64,
        node: &NodeSnapshot<D>,
        shared_row: usize,
        network_ids: &[i32],
        master_ids: &[i32],
    ) -> Result<Self> {
        if shared_row >= node.state.shared_rows.len() {
            return Err(invalid("unknown mapped shared row"));
        }
        if node.state.oracles.iter().any(|o| {
            o.nonlinear_rows.iter().any(|&(_, row)| row == shared_row)
                || o.knapsack_rows.iter().any(|&(_, row)| row == shared_row)
                || o.presence_rows.iter().any(|&(_, row)| row == shared_row)
                || o.resource_groups.iter().any(|&(row, _)| row == shared_row)
        }) {
            return Err(invalid(
                "mapped shared row must have additive original coefficients",
            ));
        }
        let mut networks = vec![];
        for &network in network_ids {
            let oracle = node
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == network)
                .ok_or_else(|| invalid("unknown mapped row network"))?;
            let per_route = exact_coefficient(
                oracle
                    .network
                    .route_entries
                    .iter()
                    .filter(|(r, _)| *r == shared_row)
                    .map(|(_, a)| *a),
            )?;
            let mut arcs = vec![];
            for (arc, entries) in oracle
                .network
                .graph
                .arcs
                .iter()
                .zip(oracle.network.arc_entries.iter())
            {
                let value = exact_coefficient(
                    entries
                        .iter()
                        .filter(|(r, _)| *r == shared_row)
                        .map(|(_, a)| *a),
                )?;
                if value != 0. {
                    arcs.push((arc.id, value));
                }
            }
            networks.push(MappedNetworkVariable {
                network,
                activities: {
                    let mut terms = BTreeMap::<usize, Vec<f64>>::new();
                    for entry in oracle.activity_rows.iter().filter(|e| e.row == shared_row) {
                        terms
                            .entry(entry.resource)
                            .or_default()
                            .push(entry.coefficient);
                    }
                    terms
                        .into_iter()
                        .map(|(r, values)| Ok((r, exact_coefficient(values.into_iter())?)))
                        .collect::<Result<Vec<_>>>()?
                },
                arcs,
                per_route,
            });
        }
        let mut masters = vec![];
        for &id in master_ids {
            let variable = node
                .state
                .master_variables
                .iter()
                .find(|v| v.id == id)
                .ok_or_else(|| invalid("unknown mapped row master variable"))?;
            let value = exact_coefficient(
                variable
                    .entries
                    .iter()
                    .filter(|(r, _)| *r == shared_row)
                    .map(|(_, a)| *a),
            )?;
            masters.push((id, value));
        }
        let mapping = Self {
            name: name.into(),
            priority,
            networks,
            masters,
        };
        mapping.validate::<D>(
            node.state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            &node.state.master_variables,
        )?;
        Ok(mapping)
    }
    pub(super) fn validate<'a, const D: usize>(
        &self,
        networks: impl Iterator<Item = (i32, &'a Graph<D>)>,
        masters: &[MasterVariable],
    ) -> Result<()> {
        if self.name.is_empty() || !self.priority.is_finite() {
            return Err(invalid(
                "mapped branch variable requires a name and finite priority",
            ));
        }
        let networks: HashMap<_, _> = networks.collect();
        let mut seen = HashSet::new();
        let mut nonzero = false;
        for contribution in &self.networks {
            let graph = networks
                .get(&contribution.network)
                .ok_or_else(|| invalid("unknown mapped branch network"))?;
            if !seen.insert(contribution.network) || !integer_coefficient(contribution.per_route) {
                return Err(invalid(
                    "unique mapped networks and integer coefficients required",
                ));
            }
            nonzero |= contribution.per_route != 0.;
            let mut seen_arcs = HashSet::new();
            let mut absolute = contribution.per_route.abs();
            for &(id, value) in &contribution.arcs {
                if !graph.arcs.iter().any(|arc| arc.id == id)
                    || !seen_arcs.insert(id)
                    || !value.is_finite()
                    || value.abs() > 1_048_576.
                    || (value * 2.).fract() != 0.
                {
                    return Err(invalid(
                        "mapped arcs must be known, distinct and have integer or half-integer coefficients",
                    ));
                }
                nonzero |= value != 0.;
                absolute += value.abs();
                if absolute > 9_007_199_254_740_992. {
                    return Err(invalid(
                        "mapped route coefficients exceed exact integer range",
                    ));
                }
            }
            if !integral_path_parity(graph, &contribution.arcs) {
                return Err(invalid(
                    "mapped half-arc expression must be integer on every graph path",
                ));
            }
            let hops = graph.original_max_arcs();
            let max_arc = contribution
                .arcs
                .iter()
                .map(|(_, a)| a.abs())
                .fold(0_f64, f64::max);
            let mut magnitude =
                contribution.per_route.abs() as u128 + (max_arc.ceil() as u128) * (hops as u128);
            let mut resources = HashSet::new();
            for &(resource, coefficient) in &contribution.activities {
                if !resources.insert(resource) || !integer_coefficient(coefficient) {
                    return Err(invalid(
                        "mapped activities require distinct descriptors and integer coefficients",
                    ));
                }
                let range = grouped_boxes::integral_activity(graph, resource).ok_or_else(|| {
                    invalid("mapped activity must be exactly integer on every graph path")
                })?;
                let maximum = range.lower.abs().max(range.upper.abs()) as u128;
                magnitude = magnitude
                    .checked_add(maximum * coefficient.abs() as u128)
                    .ok_or_else(|| invalid("mapped activity range overflow"))?;
                nonzero |= coefficient != 0.;
            }
            if magnitude > 9_007_199_254_740_992u128 {
                return Err(invalid(
                    "mapped walk coefficient exceeds exact integer range at its declared arc limit",
                ));
            }
        }
        let mut seen = HashSet::new();
        for &(id, value) in &self.masters {
            let variable = masters
                .iter()
                .find(|v| v.id == id)
                .ok_or_else(|| invalid("unknown mapped master variable"))?;
            if !seen.insert(id) || !integer_coefficient(value) || value != 0. && !variable.integer {
                return Err(invalid(
                    "mapped master terms require distinct integer variables and coefficients",
                ));
            }
            nonzero |= value != 0.;
        }
        if !nonzero {
            return Err(invalid(
                "mapped branch variable requires a nonzero contribution",
            ));
        }
        Ok(())
    }

    fn coefficients<const D: usize>(&self, node: &NodeSnapshot<D>) -> Result<Vec<RowCoefficients>> {
        self.networks
            .iter()
            .map(|c| {
                let oracle = node
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == c.network)
                    .ok_or_else(|| invalid("unknown mapped branch network"))?;
                let mut arcs = vec![0.; oracle.network.graph.arcs.len()];
                for &(id, value) in &c.arcs {
                    let &position = oracle
                        .index
                        .get(&id)
                        .ok_or_else(|| invalid("unknown mapped branch arc"))?;
                    arcs[position] = value;
                }
                Ok(RowCoefficients {
                    network: c.network,
                    arcs,
                    per_route: c.per_route,
                })
            })
            .collect()
    }

    /// Install one side of an original aggregate-variable disjunction. Fixed
    /// routes, static master columns and all future route columns use the same
    /// expression. Mathematical infinite endpoints add no backend sentinel row.
    pub fn apply<const D: usize>(
        &self,
        node: &NodeSnapshot<D>,
        bounds: Bounds,
    ) -> Result<NodeSnapshot<D>> {
        self.apply_coordinates(node, bounds, None)
    }

    /// Install residual aggregate bounds while retaining full-coordinate rows
    /// for witness replay. The fixed offset must be exact and previously checked
    /// against this expression; static-master offsets are unsupported.
    pub(super) fn apply_residual<const D: usize>(
        &self,
        node: &NodeSnapshot<D>,
        residual: Bounds,
        fixed_activity: f64,
    ) -> Result<NodeSnapshot<D>> {
        if !self.masters.is_empty() || !fixed_activity.is_finite() {
            return Err(invalid("unsupported aggregate fixed offset"));
        }
        let translate = |endpoint: f64| -> Result<f64> {
            if endpoint.is_infinite() {
                Ok(endpoint)
            } else {
                exact_coefficient([endpoint, fixed_activity].into_iter())
            }
        };
        let full = Bounds {
            lower: translate(residual.lower)?,
            upper: translate(residual.upper)?,
        };
        self.apply_coordinates(node, full, Some(residual))
    }

    fn apply_coordinates<const D: usize>(
        &self,
        node: &NodeSnapshot<D>,
        bounds: Bounds,
        residual: Option<Bounds>,
    ) -> Result<NodeSnapshot<D>> {
        self.validate::<D>(
            node.state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            &node.state.master_variables,
        )?;
        if !master_variables::valid_bounds(bounds)
            || residual.is_some_and(|bounds| !master_variables::valid_bounds(bounds))
        {
            return Err(invalid("ordered mapped variable bounds required"));
        }
        let coefficients = self.coefficients(node)?;
        let activities: Vec<_> = self
            .networks
            .iter()
            .flat_map(|c| {
                c.activities.iter().map(|&(resource, coefficient)| {
                    AccumulatedActivityCoefficients {
                        network: c.network,
                        resource,
                        coefficient,
                    }
                })
            })
            .collect();
        let mut child = node.fork()?;
        let append = |node: NodeSnapshot<D>, row, residual_rhs| -> Result<NodeSnapshot<D>> {
            let shared = node.state.shared_rows.len();
            let mut node = node.with_row_class_owned(
                row,
                &coefficients,
                cg::CurvatureRowClass::Ordinary,
                residual_rhs,
                &activities,
            )?;
            node.attach_master_coefficients(shared, &self.masters)?;
            Ok(node)
        };
        if bounds.lower == bounds.upper {
            return append(
                child,
                Row {
                    sense: RowSense::Equal,
                    rhs: bounds.lower,
                },
                residual.map(|bounds| bounds.lower),
            );
        }
        for (sense, rhs, residual_rhs) in [
            (
                RowSense::GreaterEqual,
                bounds.lower,
                residual.map(|b| b.lower),
            ),
            (RowSense::LessEqual, bounds.upper, residual.map(|b| b.upper)),
        ] {
            if rhs.is_finite() {
                child = append(child, Row { sense, rhs }, residual_rhs)?;
            }
        }
        Ok(child)
    }

    /// Exact fixed-route offset for translating a residual aggregate row into
    /// the full coordinates accepted by `apply`. Static-master offsets are not
    /// represented by this helper. Callers must also verify the applied row's
    /// residual RHS because row installation subtracts fixed copies in order.
    pub(super) fn fixed_activity<const D: usize>(&self, output: &Output<D>) -> Option<f64> {
        if !self.masters.is_empty() {
            return None;
        }
        let mut sum = crate::master::numerics::ExactSum::default();
        let mut source_work = 1_000_000;
        for fixed in &output.snapshot.state.fixed_routes {
            let quantity = fixed.quantity;
            if !quantity.is_finite() || quantity.fract() != 0. {
                return None;
            }
            let route = &fixed.route;
            let Some(c) = self.networks.iter().find(|c| c.network == route.network) else {
                continue;
            };
            let mut activities = vec![];
            if !c.activities.is_empty() {
                let oracle = output
                    .snapshot
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == c.network)?;
                let path: Vec<_> = route
                    .route
                    .arcs
                    .iter()
                    .map(|id| oracle.index.get(id).copied())
                    .collect::<Option<_>>()?;
                for &(resource, factor) in &c.activities {
                    let activity =
                        activity_rows::column_value(oracle, &route.route, &path, resource).ok()?;
                    if activity_rows::paired(&oracle.network.graph, resource) {
                        let proof = activity_rows::certificate_value(
                            oracle,
                            &route.route,
                            resource,
                            &mut source_work,
                            output.snapshot.state.deadline,
                        )?;
                        if proof != [activity, activity] {
                            return None;
                        }
                    }
                    let product = activity * factor;
                    if !product.is_finite() || activity.mul_add(factor, -product) != 0. {
                        return None;
                    }
                    activities.push(product);
                }
            }
            let coefficient =
                exact_coefficient(
                    std::iter::once(c.per_route)
                        .chain(
                            route.route.arcs.iter().map(|id| {
                                c.arcs.iter().find(|(a, _)| a == id).map_or(0., |(_, a)| *a)
                            }),
                        )
                        .chain(activities),
                )
                .ok()?;
            if coefficient.fract() != 0. || coefficient.abs() > 9_007_199_254_740_992. {
                return None;
            }
            let product = quantity * coefficient;
            if !product.is_finite() || quantity.mul_add(coefficient, -product) != 0. {
                return None;
            }
            sum.add(product)?;
        }
        let fixed = sum.upper()?;
        sum.add(-fixed)?;
        (sum.sign() == 0.).then_some(fixed)
    }

    pub(super) fn value<const D: usize>(&self, output: &Output<D>) -> Option<f64> {
        let mut value = crate::master::numerics::ExactSum::default();
        let mut source_work = 1_000_000;
        for (route, quantity) in output
            .routes
            .iter()
            .zip(&output.values)
            .map(|(r, &q)| (r, q))
            .chain(
                output
                    .snapshot
                    .state
                    .fixed_routes
                    .iter()
                    .map(|f| (&f.route, f.quantity)),
            )
        {
            if quantity == 0. {
                continue;
            }
            if let Some(c) = self.networks.iter().find(|c| c.network == route.network) {
                let mut activity_terms = vec![];
                if !c.activities.is_empty() {
                    let oracle = output
                        .snapshot
                        .state
                        .oracles
                        .iter()
                        .find(|o| o.network.id == c.network)?;
                    let path: Vec<_> = route
                        .route
                        .arcs
                        .iter()
                        .map(|id| oracle.index.get(id).copied())
                        .collect::<Option<_>>()?;
                    for &(resource, coefficient) in &c.activities {
                        let activity =
                            activity_rows::column_value(oracle, &route.route, &path, resource)
                                .ok()?;
                        if activity_rows::paired(&oracle.network.graph, resource) {
                            let proof = activity_rows::certificate_value(
                                oracle,
                                &route.route,
                                resource,
                                &mut source_work,
                                output.snapshot.state.deadline,
                            )?;
                            if proof != [activity, activity] {
                                return None;
                            }
                        }
                        activity_terms.push(activity * coefficient);
                    }
                }
                let coefficient =
                    exact_coefficient(
                        std::iter::once(c.per_route)
                            .chain(route.route.arcs.iter().map(|id| {
                                c.arcs.iter().find(|(a, _)| a == id).map_or(0., |(_, a)| *a)
                            }))
                            .chain(activity_terms),
                    )
                    .ok()?;
                // Route coefficients must remain exact integer values. Large
                // original activity domains are declined as branch candidates.
                if coefficient.fract() != 0. || coefficient.abs() > 9_007_199_254_740_992. {
                    return None;
                }
                // The multiplier is an exact integer, so product() retains
                // both multiplication residuals and cross-column cancellation.
                value.product(quantity, coefficient)?;
            }
        }
        for &(id, a) in &self.masters {
            let &(_, x) = output.master_values.iter().find(|(i, _)| *i == id)?;
            if a.fract() != 0. {
                return None;
            }
            value.product(x, a)?;
        }
        // Activity is numerical candidate/ranking data, not a proof bound.
        // Use one directed collapse after summing all signed original terms.
        value.upper()
    }
}

pub(super) fn validate<'a, const D: usize>(
    variables: &[MappedBranchVariable],
    networks: impl Iterator<Item = (i32, &'a Graph<D>)> + Clone,
    masters: &[MasterVariable],
) -> Result<()> {
    let mut names = HashSet::new();
    for variable in variables {
        if !names.insert(&variable.name) {
            return Err(invalid("distinct mapped branch variable names required"));
        }
        variable.validate::<D>(networks.clone(), masters)?;
    }
    Ok(())
}

/// Order fractional candidates as the reference's ordered candidate set does.
/// The comparison uses a tolerance band (relative 1e-10, absolute 1e-6):
/// larger priority, then larger distance to the nearest integer; values inside
/// the band tie and keep registration order. Candidates are inserted one by one
/// in registration order because the band is not transitive and a sort could
/// reorder near-ties differently.
pub(super) fn rank_candidates<'a>(
    candidates: impl Iterator<Item = (&'a MappedBranchVariable, f64)>,
) -> Vec<(&'a MappedBranchVariable, f64)> {
    fn exceeds(a: f64, b: f64) -> bool {
        a > b + (a.abs().max(b.abs()) * 1e-10 + 1e-6)
    }
    let before = |(a, x): &(&MappedBranchVariable, f64), (b, y): &(&MappedBranchVariable, f64)| {
        if exceeds(a.priority, b.priority) {
            return true;
        }
        if exceeds(b.priority, a.priority) {
            return false;
        }
        exceeds((x - x.round()).abs(), (y - y.round()).abs())
    };
    let mut ranked: Vec<(&MappedBranchVariable, f64)> = vec![];
    for candidate in candidates {
        // A later registration goes after every element it does not precede.
        let at = ranked.partition_point(|c| !before(&candidate, c));
        ranked.insert(at, candidate);
    }
    ranked
}

/// Rank mapped families with independent variable priorities and selection keys.
/// A red-black insertion order preserves the tolerant comparison over chains.
pub(crate) fn rank_candidates_with_selection<'a>(
    candidates: impl Iterator<Item = (&'a MappedBranchVariable, f64)>,
    selection: &BTreeMap<String, BranchSelection>,
) -> Vec<(&'a MappedBranchVariable, f64)> {
    if selection.is_empty() {
        return rank_candidates(candidates);
    }
    let candidates: Vec<_> = candidates.collect();
    let mut set = crate::cuts::shared::rbtree::CandidateSet::new();
    for index in 0..candidates.len() {
        set.insert_by(index, |a, b| {
            let (av, ax) = candidates[a];
            let (bv, bx) = candidates[b];
            let am = selection.get(&av.name);
            let bm = selection.get(&bv.name);
            let same_family = am.zip(bm).is_some_and(|(a, b)| a.family == b.family);
            if !same_family {
                if selection_exceeds(av.priority, bv.priority) {
                    return true;
                }
                if selection_exceeds(bv.priority, av.priority) {
                    return false;
                }
            }
            if let (Some(am), Some(bm)) = (am, bm) {
                return selection_before(am, ax, a, bm, bx, b);
            }
            let af = (ax - ax.round()).abs();
            let bf = (bx - bx.round()).abs();
            if selection_exceeds(af, bf) {
                return true;
            }
            if selection_exceeds(bf, af) {
                return false;
            }
            a < b
        });
    }
    set.ordered()
        .into_iter()
        .map(|(_, index)| candidates[index])
        .collect()
}

pub(super) fn children<const D: usize>(
    output: &Output<D>,
    variables: &[MappedBranchVariable],
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let candidates = rank_candidates(variables.iter().filter_map(|v| {
        let value = v.value(output)?;
        (value.abs() < 9_007_199_254_740_992. && (value - value.round()).abs() > tolerance)
            .then_some((v, value))
    }));
    let Some((variable, value)) = candidates.first() else {
        return Ok(None);
    };
    Ok(Some(vec![
        variable.apply(
            &output.snapshot,
            Bounds {
                lower: f64::NEG_INFINITY,
                upper: value.floor(),
            },
        )?,
        variable.apply(
            &output.snapshot,
            Bounds {
                lower: value.ceil(),
                upper: f64::INFINITY,
            },
        )?,
    ]))
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/model_mapping_tests.rs"]
mod tests;
