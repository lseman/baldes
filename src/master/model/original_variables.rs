//! Compile original variable associations into the physical pricing model.
use super::*;
use crate::master::preprocessing::Bounds;

/// One original variable, scoped by its pricing network and model ID.
/// Graph resources define its per-pattern feasibility. This registration does
/// not silently add bounds, resource dimensions, or global master restrictions.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalPricingVariable {
    pub network: i32,
    pub id: i32,
    pub cost: f64,
    #[serde(default)]
    pub entries: Vec<(usize, f64)>,
    /// Physical arc IDs and association coefficients. The last setter for an
    /// arc replaces its previous coefficient; different variables add.
    pub arcs: Vec<(i32, f64)>,
    /// Register the original activity for integer branching. This requires a
    /// proof of integer contribution on every complete graph walk.
    #[serde(default)]
    pub integer: bool,
    #[serde(default)]
    pub priority: f64,
}

/// Retained original identities and expressions, independent of column tags.
#[derive(Clone, Debug)]
pub struct PricingVariableRegistry {
    variables: Vec<OriginalPricingVariable>,
    branches: Vec<MappedBranchVariable>,
    graph_arcs: HashMap<i32, HashSet<i32>>,
    costs: Arc<Vec<OriginalVariableCosts>>,
}

/// Expand variable costs and shared-row coefficients once, before solving.
/// Base arc costs/entries remain additive. Expansion uses finite binary64
/// products and sums; the returned physical coefficients define the compiled
/// model. Original variable bounds must be encoded by the caller's graph.
/// The input is owned, so failed compilation never edits a retained snapshot.
pub fn compile_pricing_variables<const D: usize>(
    mut problem: Problem<D>,
    definitions: &[OriginalPricingVariable],
) -> Result<(Problem<D>, PricingVariableRegistry)> {
    problem.validate()?;
    let registered: HashSet<_> = definitions.iter().map(|v| v.network).collect();
    let mut costs: Vec<_> = problem
        .networks
        .iter()
        .filter(|n| registered.contains(&n.id))
        .map(|n| OriginalVariableCosts {
            network: n.id,
            variables: vec![],
            arcs: n
                .graph
                .arcs
                .iter()
                .zip(&n.arc_entries)
                .map(|(arc, entries)| {
                    (
                        arc.id,
                        original_costs::ArcCost {
                            base: arc.cost,
                            entries: entries.clone(),
                            compiled: arc.cost,
                            compiled_entries: entries.clone(),
                        },
                    )
                })
                .collect(),
        })
        .collect();
    let positions: HashMap<_, _> = problem
        .networks
        .iter()
        .enumerate()
        .map(|(p, n)| (n.id, p))
        .collect();
    let mut variables = definitions.to_vec();
    variables.sort_by_key(|v| (v.network, v.id));
    let mut seen = HashSet::new();
    let mut branches = vec![];
    let mut graph_arcs = HashMap::new();
    for variable in &mut variables {
        let &position = positions
            .get(&variable.network)
            .ok_or_else(|| invalid("unknown original-variable network"))?;
        if variable.id < 0
            || !seen.insert((variable.network, variable.id))
            || !variable.cost.is_finite()
            || !variable.priority.is_finite()
            || variable
                .entries
                .iter()
                .any(|&(row, a)| row >= problem.rows.len() || !a.is_finite())
        {
            return Err(invalid("invalid or duplicate original pricing variable"));
        }
        let network = &mut problem.networks[position];
        let arcs: HashMap<_, _> = network
            .graph
            .arcs
            .iter()
            .enumerate()
            .map(|(p, a)| (a.id, p))
            .collect();
        graph_arcs
            .entry(variable.network)
            .or_insert_with(|| arcs.keys().copied().collect());
        let setters: BTreeMap<_, _> = variable.arcs.iter().copied().collect();
        for (&id, &coefficient) in &setters {
            if !arcs.contains_key(&id) || !coefficient.is_finite() {
                return Err(invalid("invalid original-variable arc association"));
            }
        }
        variable.arcs = setters.into_iter().filter(|(_, a)| *a != 0.).collect();
        if variable.integer && !variable.arcs.is_empty() {
            let branch = MappedBranchVariable {
                name: format!("X_{}_{}", variable.network, variable.id),
                priority: variable.priority,
                networks: vec![MappedNetworkVariable {
                    network: variable.network,
                    arcs: variable.arcs.clone(),
                    per_route: 0.,
                }],
                masters: vec![],
            };
            branch.validate(std::iter::once((network.id, &network.graph)), &[])?;
            branches.push(branch);
        }
        for &(id, coefficient) in &variable.arcs {
            let position = arcs[&id];
            let cost = network.graph.arcs[position].cost + coefficient * variable.cost;
            if !cost.is_finite() {
                return Err(invalid("original-variable arc cost overflow"));
            }
            network.graph.arc_mut(id)?.cost = cost;
            for &(row, value) in &variable.entries {
                let value = coefficient * value;
                if !value.is_finite() {
                    return Err(invalid("original-variable row coefficient overflow"));
                }
                network.arc_entries[position].push((row, value));
            }
        }
    }
    problem.validate()?;
    for cost in &mut costs {
        let network = &problem.networks[positions[&cost.network]];
        cost.variables = variables
            .iter()
            .filter(|v| v.network == network.id)
            .cloned()
            .collect();
        for (arc, entries) in network.graph.arcs.iter().zip(&network.arc_entries) {
            let saved = cost.arcs.get_mut(&arc.id).unwrap();
            saved.compiled = arc.cost;
            saved.compiled_entries = entries.clone();
        }
    }
    costs.retain(|c| !c.variables.is_empty());
    Ok((
        problem,
        PricingVariableRegistry {
            variables,
            branches,
            graph_arcs,
            costs: Arc::new(costs),
        },
    ))
}

impl PricingVariableRegistry {
    /// Retain original-space cost rounding for fresh or resumed pricing.
    /// Replaces this option's decomposition with the compiler's canonical one.
    pub fn configure_pricing(&self, options: &mut Options) {
        options.original_variable_costs.clone_from(&self.costs);
    }
    pub fn variables(&self) -> &[OriginalPricingVariable] {
        &self.variables
    }

    /// Resolve original identities into verified grouped-box requests. Bounds
    /// remain envelopes; preparation checks them against the recipient graph.
    pub fn configure_boxes(
        &self,
        options: &mut Options,
        bounds: &[OriginalVariableBound],
    ) -> Result<()> {
        let mut seen: HashSet<_> = options
            .grouped_variable_boxes
            .iter()
            .map(|b| (b.variable.network, b.variable.id))
            .collect();
        let mut additions = vec![];
        for bound in bounds {
            if !seen.insert((bound.network, bound.id)) {
                return Err(invalid("duplicate original-variable box"));
            }
            let variable = self
                .variables
                .iter()
                .find(|v| v.network == bound.network && v.id == bound.id)
                .ok_or_else(|| invalid("unknown original-variable box identity"))?;
            additions.push(GroupedVariableBound {
                variable: variable.clone(),
                lower: bound.lower,
                upper: bound.upper,
            });
        }
        options.grouped_variable_boxes.extend(additions);
        self.configure_pricing(options);
        Ok(())
    }

    pub fn branch_variables(&self) -> &[MappedBranchVariable] {
        &self.branches
    }

    /// Add registered integer activities to a tree without replacing explicit
    /// registrations. Name conflicts fail before publishing any edits.
    pub fn configure_tree(&self, options: &mut ModelTreeOptions) -> Result<()> {
        let names: HashSet<_> = options.mapped_variables.iter().map(|v| &v.name).collect();
        if self.branches.iter().any(|v| names.contains(&v.name)) {
            return Err(invalid(
                "original-variable branch registration name conflict",
            ));
        }
        options.mapped_variables.extend(self.branches.clone());
        self.configure_pricing(&mut options.node);
        Ok(())
    }

    /// Enclose an original activity at the reported LP point, including fixed
    /// route copies. This is a primal expression, never a pricing certificate.
    /// Routes must use the compiled model's original physical arc IDs.
    pub fn activity<const D: usize>(
        &self,
        network: i32,
        id: i32,
        output: &Output<D>,
    ) -> Result<Bounds> {
        self.activity_from_paths(
            network,
            id,
            output
                .routes
                .iter()
                .zip(&output.values)
                .map(|(r, &q)| (r.network, r.route.arcs.as_slice(), q))
                .chain(
                    output
                        .fixed_routes
                        .iter()
                        .map(|f| (f.route.network, f.route.route.arcs.as_slice(), f.quantity)),
                ),
        )
    }

    /// Reconstruct an activity from original physical paths, including restored
    /// tree or enumeration witnesses. Each tuple is (network, arc IDs, quantity).
    pub fn activity_from_paths<'a>(
        &self,
        network: i32,
        id: i32,
        paths: impl IntoIterator<Item = (i32, &'a [i32], f64)>,
    ) -> Result<Bounds> {
        let variable = self
            .variables
            .iter()
            .find(|v| v.network == network && v.id == id)
            .ok_or_else(|| invalid("unknown original pricing variable"))?;
        let coefficients: HashMap<_, _> = variable.arcs.iter().copied().collect();
        let mut upper = crate::master::numerics::ExactSum::default();
        let mut negative_lower = crate::master::numerics::ExactSum::default();
        for (_, arcs, quantity) in paths
            .into_iter()
            .filter(|(n, _, quantity)| *n == network && *quantity != 0.)
        {
            for arc in arcs {
                if !self.graph_arcs[&network].contains(arc) {
                    return Err(invalid(
                        "original-variable activity uses an unknown physical arc",
                    ));
                }
                let coefficient = coefficients.get(arc).copied().unwrap_or(0.);
                upper
                    .product_upper(quantity, coefficient)
                    .ok_or_else(|| invalid("original-variable activity overflow"))?;
                negative_lower
                    .product_upper(-quantity, coefficient)
                    .ok_or_else(|| invalid("original-variable activity overflow"))?;
            }
        }
        Ok(Bounds {
            lower: -negative_lower
                .upper()
                .ok_or_else(|| invalid("original-variable activity overflow"))?,
            upper: upper
                .upper()
                .ok_or_else(|| invalid("original-variable activity overflow"))?,
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/original_variable_tests.rs"]
mod tests;
