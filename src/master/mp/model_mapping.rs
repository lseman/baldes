//! Explicit original-variable mappings across pricing networks and master IDs.
//! These are model registrations, never inferred from display names or route tags.
use super::*;
use crate::master::preprocessing::Bounds;

/// Integer contribution of one network to an original aggregate variable.
/// Arc IDs belong to this network; omitted arcs have zero coefficient.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MappedNetworkVariable {
    pub network: i32,
    #[serde(default)]
    pub arcs: Vec<(i32, f64)>,
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
        let mut networks = vec![];
        for &network in network_ids {
            let oracle = node
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == network)
                .ok_or_else(|| invalid("unknown mapped row network"))?;
            let per_route = oracle
                .network
                .route_entries
                .iter()
                .filter(|(r, _)| *r == shared_row)
                .map(|(_, a)| a)
                .sum();
            let arcs = oracle
                .network
                .graph
                .arcs
                .iter()
                .zip(oracle.network.arc_entries.iter())
                .filter_map(|(arc, entries)| {
                    let value: f64 = entries
                        .iter()
                        .filter(|(r, _)| *r == shared_row)
                        .map(|(_, a)| a)
                        .sum();
                    (value != 0.).then_some((arc.id, value))
                })
                .collect();
            networks.push(MappedNetworkVariable {
                network,
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
            let value = variable
                .entries
                .iter()
                .filter(|(r, _)| *r == shared_row)
                .map(|(_, a)| a)
                .sum();
            masters.push((id, value));
        }
        let mapping = Self {
            name: name.into(),
            priority,
            networks,
            masters,
        };
        mapping.validate::<D>(
            node.state.oracles.iter().map(|o| {
                (
                    o.network.id,
                    o.network.graph.arcs.iter().map(|a| a.id).collect(),
                )
            }),
            &node.state.master_variables,
        )?;
        Ok(mapping)
    }
    pub(super) fn validate<const D: usize>(
        &self,
        networks: impl Iterator<Item = (i32, Vec<i32>)>,
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
            let arcs = networks
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
                if !arcs.contains(&id) || !seen_arcs.insert(id) || !integer_coefficient(value) {
                    return Err(invalid(
                        "mapped arcs must be known, distinct and have integer coefficients",
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
            node.state.oracles.iter().map(|o| {
                (
                    o.network.id,
                    o.network.graph.arcs.iter().map(|a| a.id).collect(),
                )
            }),
            &node.state.master_variables,
        )?;
        if !master_variables::valid_bounds(bounds) {
            return Err(invalid("ordered mapped variable bounds required"));
        }
        let coefficients = self.coefficients(node)?;
        let mut child = node.fork()?;
        if bounds.lower == bounds.upper {
            return child.with_master_row(
                Row {
                    sense: RowSense::Equal,
                    rhs: bounds.lower,
                },
                &coefficients,
                &self.masters,
            );
        }
        for (sense, rhs) in [
            (RowSense::GreaterEqual, bounds.lower),
            (RowSense::LessEqual, bounds.upper),
        ] {
            if rhs.is_finite() {
                child = child.with_master_row(Row { sense, rhs }, &coefficients, &self.masters)?;
            }
        }
        Ok(child)
    }

    pub(super) fn value<const D: usize>(&self, output: &Output<D>) -> Option<f64> {
        let mut value = 0.;
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
                let coefficient = route.route.arcs.iter().fold(c.per_route, |v, id| {
                    v + c.arcs.iter().find(|(a, _)| a == id).map_or(0., |(_, a)| *a)
                });
                // Route coefficients must remain exact integer values. Large
                // original activity domains are declined as branch candidates.
                if !coefficient.is_finite() || coefficient.abs() > 9_007_199_254_740_992. {
                    return None;
                }
                value += coefficient * quantity;
            }
        }
        for &(id, a) in &self.masters {
            let &(_, x) = output.master_values.iter().find(|(i, _)| *i == id)?;
            value += a * x;
        }
        value.is_finite().then_some(value)
    }
}

pub(super) fn validate<const D: usize>(
    variables: &[MappedBranchVariable],
    networks: impl Iterator<Item = (i32, Vec<i32>)> + Clone,
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
#[path = "../../../tests/unit/master/mp/model_mapping_tests.rs"]
mod tests;
