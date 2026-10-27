//! Explicit original-variable mappings across pricing networks and master IDs.
//! These are model registrations, never inferred from display names or route tags.
use super::*;
use crate::master::preprocessing::Bounds;

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
    /// Intrinsic custom/cumulative or selected source paired terms. Each descriptor must have
    /// a complete exact-integer path proof in this graph domain.
    #[serde(default)]
    pub activities: Vec<(usize, f64)>,
    #[serde(default)]
    pub per_route: f64,
}

/// A named integer model variable assembled from several pricing networks and
/// independent integer master variables. Larger priorities are considered first;
/// equal-priority candidates prefer the most fractional value, then their name.
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
        self.validate::<D>(
            node.state
                .oracles
                .iter()
                .map(|o| (o.network.id, o.network.graph.as_ref())),
            &node.state.master_variables,
        )?;
        if !master_variables::valid_bounds(bounds) {
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
        let append = |node: &NodeSnapshot<D>, row| {
            if activities.is_empty() {
                node.with_master_row(row, &coefficients, &self.masters)
            } else {
                node.with_accumulated_activity_master_row(
                    row,
                    &activities,
                    &coefficients,
                    &self.masters,
                )
            }
        };
        if bounds.lower == bounds.upper {
            return append(
                &child,
                Row {
                    sense: RowSense::Equal,
                    rhs: bounds.lower,
                },
            );
        }
        for (sense, rhs) in [
            (RowSense::GreaterEqual, bounds.lower),
            (RowSense::LessEqual, bounds.upper),
        ] {
            if rhs.is_finite() {
                child = append(&child, Row { sense, rhs })?;
            }
        }
        Ok(child)
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

pub(super) fn children<const D: usize>(
    output: &Output<D>,
    variables: &[MappedBranchVariable],
    tolerance: f64,
) -> Result<Option<Vec<NodeSnapshot<D>>>> {
    let mut candidates: Vec<_> = variables
        .iter()
        .filter_map(|v| {
            let value = v.value(output)?;
            (value.abs() < 9_007_199_254_740_992. && (value - value.round()).abs() > tolerance)
                .then_some((v, value))
        })
        .collect();
    candidates.sort_by(|(a, x), (b, y)| {
        b.priority
            .total_cmp(&a.priority)
            .then_with(|| {
                ((x - x.floor()) - 0.5)
                    .abs()
                    .total_cmp(&((y - y.floor()) - 0.5).abs())
            })
            .then_with(|| a.name.cmp(&b.name))
    });
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
