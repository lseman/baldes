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
    #[serde(default)]
    pub arcs: Vec<(i32, f64)>,
    /// Intrinsic custom/cumulative activity or a selected absolute paired value
    /// instead of fixed arc associations. Paired bindings retain the owner
    /// coefficient separately from the unit-charge paired pricing program.
    /// Its objective coefficient replaces the descriptor's implicit unit cost;
    /// the JSON network cannot also assign a graph cost to this resource.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accumulated_resource: Option<usize>,
    /// Register the original activity for integer branching. This requires a
    /// proof of integer contribution on every complete graph walk, except for
    /// paired associations: their integer flag is retained as registration
    /// metadata and does not enforce projected activity integrality.
    #[serde(default)]
    pub integer: bool,
    #[serde(default)]
    pub priority: f64,
}

/// Captured intrinsic program and its original physical arc identities.
#[derive(Clone, Debug)]
struct ActivitySource {
    kernel: Arc<crate::graph::coupled::CoupledKernel>,
    arcs: HashMap<i32, usize>,
}

/// Retained original identities and expressions, independent of column tags.
#[derive(Clone, Debug)]
pub struct PricingVariableRegistry {
    variables: Vec<OriginalPricingVariable>,
    branches: Vec<MappedBranchVariable>,
    graph_arcs: HashMap<i32, HashSet<i32>>,
    costs: Arc<Vec<OriginalVariableCosts>>,
    activity_sources: HashMap<i32, ActivitySource>,
}

/// A generated/restored column expression for original-variable reporting.
/// Selected values are required for paired bindings; deterministic activities
/// can be reconstructed from arcs alone. This is not a pricing certificate.
#[derive(Clone, Copy, Debug)]
pub struct OriginalColumn<'a> {
    pub network: i32,
    pub arcs: &'a [i32],
    pub coupled_values: Option<&'a [(usize, f64)]>,
    pub quantity: f64,
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
    let mut activity_bindings = HashSet::new();
    let mut resource_costs = HashMap::<i32, BTreeMap<usize, f64>>::new();
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
        if let Some(resource) = variable.accumulated_resource {
            if !variable.arcs.is_empty() || !activity_bindings.insert((variable.network, resource))
            {
                return Err(invalid(
                    "an accumulated resource needs one original variable and no arc associations",
                ));
            }
            if matches!(
                network
                    .graph
                    .coupled_kernel()
                    .and_then(|kernel| kernel.definitions().get(resource)),
                Some(crate::graph::coupled::CoupledResource::PairedFront {
                    mode: crate::graph::coupled::PairedResourceMode::Feasibility,
                })
            ) {
                return Err(invalid("a feasibility-only paired resource cannot have an associated original variable"));
            }
            let paired = matches!(
                network
                    .graph
                    .coupled_kernel()
                    .and_then(|kernel| kernel.definitions().get(resource)),
                Some(crate::graph::coupled::CoupledResource::PairedFront {
                    mode: crate::graph::coupled::PairedResourceMode::CostBounded
                        | crate::graph::coupled::PairedResourceMode::CostUnbounded,
                })
            );
            if !paired {
                activity_rows::validate(
                    &network.graph,
                    &AccumulatedActivityCoefficients {
                        network: variable.network,
                        resource,
                        coefficient: 1.,
                    },
                )?;
            }
            // Paired owner registration accepts I even for fractional selected
            // values. Preserve that flag without inventing a mapped integer
            // activity or a proof that every selected value is integral.
            if variable.integer && !paired {
                let branch = MappedBranchVariable {
                    name: format!("X_{}_{}", variable.network, variable.id),
                    priority: variable.priority,
                    networks: vec![MappedNetworkVariable {
                        network: variable.network,
                        arcs: vec![],
                        activities: vec![(resource, 1.)],
                        per_route: 0.,
                    }],
                    masters: vec![],
                };
                branch.validate(std::iter::once((network.id, &network.graph)), &[])?;
                branches.push(branch);
            }
            for &(row, coefficient) in &variable.entries {
                network.activity_entries.push(AccumulatedActivityEntry {
                    row,
                    resource,
                    coefficient,
                });
            }
            resource_costs
                .entry(variable.network)
                .or_default()
                .insert(resource, if paired { 1. } else { variable.cost });
            continue;
        }
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
                    activities: vec![],
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
    for (network, assignments) in resource_costs {
        let graph = &mut problem.networks[positions[&network]].graph;
        let kernel = graph.coupled_kernel().unwrap();
        let mut coefficients: BTreeMap<_, _> = (0..kernel.definitions().len())
            .filter_map(|resource| {
                let coefficient = kernel.objective_coefficient(resource).unwrap();
                (coefficient != 1.).then_some((resource, coefficient))
            })
            .collect();
        coefficients.extend(assignments);
        graph.set_coupled_objective_coefficients(&coefficients.into_iter().collect::<Vec<_>>())?;
    }
    for network in &mut problem.networks {
        let paired_costs = variables
            .iter()
            .filter(|variable| variable.network == network.id)
            .filter_map(|variable| {
                variable
                    .accumulated_resource
                    .filter(|&resource| activity_rows::paired(&network.graph, resource))
                    .map(|resource| (resource, variable.cost))
            })
            .collect::<Vec<_>>();
        if !paired_costs.is_empty() {
            network.graph.set_paired_associated_costs(&paired_costs)?;
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
    let activity_sources = problem
        .networks
        .iter()
        .filter(|network| {
            variables.iter().any(|variable| {
                variable.network == network.id && variable.accumulated_resource.is_some()
            })
        })
        .map(|network| {
            (
                network.id,
                ActivitySource {
                    kernel: Arc::new(network.graph.coupled_kernel().unwrap().clone()),
                    arcs: network
                        .graph
                        .arcs
                        .iter()
                        .enumerate()
                        .map(|(position, arc)| (arc.id, position))
                        .collect(),
                },
            )
        })
        .collect();
    Ok((
        problem,
        PricingVariableRegistry {
            variables,
            branches,
            graph_arcs,
            costs: Arc::new(costs),
            activity_sources,
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
        self.activity_from_columns(
            network,
            id,
            output
                .routes
                .iter()
                .zip(&output.values)
                .map(|(r, &q)| OriginalColumn {
                    network: r.network,
                    arcs: &r.route.arcs,
                    coupled_values: Some(&r.route.coupled_values),
                    quantity: q,
                })
                .chain(output.fixed_routes.iter().map(|f| OriginalColumn {
                    network: f.route.network,
                    arcs: &f.route.route.arcs,
                    coupled_values: Some(&f.route.route.coupled_values),
                    quantity: f.quantity,
                })),
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
        self.activity_from_columns(
            network,
            id,
            paths
                .into_iter()
                .map(|(network, arcs, quantity)| OriginalColumn {
                    network,
                    arcs,
                    coupled_values: None,
                    quantity,
                }),
        )
    }

    /// Enclose a primal expression using selected column values where needed.
    pub fn activity_from_columns<'a>(
        &self,
        network: i32,
        id: i32,
        columns: impl IntoIterator<Item = OriginalColumn<'a>>,
    ) -> Result<Bounds> {
        let variable = self
            .variables
            .iter()
            .find(|v| v.network == network && v.id == id)
            .ok_or_else(|| invalid("unknown original pricing variable"))?;
        let coefficients: HashMap<_, _> = variable.arcs.iter().copied().collect();
        let mut upper = crate::master::numerics::ExactSum::default();
        let mut negative_lower = crate::master::numerics::ExactSum::default();
        for OriginalColumn {
            arcs,
            coupled_values,
            quantity,
            ..
        } in columns
            .into_iter()
            .filter(|column| column.network == network && column.quantity != 0.)
        {
            if let Some(resource) = variable.accumulated_resource {
                let source = &self.activity_sources[&network];
                let path = arcs
                    .iter()
                    .map(|id| {
                        source.arcs.get(id).copied().ok_or_else(|| {
                            invalid("original activity uses an unknown physical arc")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let value = if matches!(
                    source.kernel.definitions().get(resource),
                    Some(crate::graph::coupled::CoupledResource::PairedFront { .. })
                ) {
                    let selected = coupled_values.ok_or_else(|| {
                        invalid("paired activity reporting requires selected column values")
                    })?;
                    let matches = if source.kernel.machine_paired_join_complete() {
                        match selected {
                            [(index, value)]
                                if source.kernel.retained_paired_machine_resource()
                                    == Some(*index) =>
                            {
                                source
                                    .kernel
                                    .joined_machine_path_matching(&path, Some(*value))?
                                    .is_some()
                            }
                            _ => false,
                        }
                    } else {
                        source.kernel.selected_paired_values(&path)?.as_deref() == Some(selected)
                    };
                    if !matches {
                        return Err(invalid("paired activity column values do not match the compiled resource program"));
                    }
                    selected
                        .iter()
                        .find(|(index, _)| *index == resource)
                        .map(|(_, value)| *value)
                        .ok_or_else(|| invalid("missing selected paired value"))?
                } else {
                    source.kernel.accumulated_activity(resource, &path)?
                };
                upper
                    .product_upper(quantity, value)
                    .ok_or_else(|| invalid("original activity overflow"))?;
                negative_lower
                    .product_upper(-quantity, value)
                    .ok_or_else(|| invalid("original activity overflow"))?;
                continue;
            }
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
