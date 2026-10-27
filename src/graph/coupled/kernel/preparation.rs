//! Resource preparation, preprocessing diagnostics, and objective configuration.
use super::*;

impl CoupledKernel {
    fn refresh_bounds<const D: usize>(&mut self, graph: &Graph<D>) {
        self.optimization =
            PreparedBounds::new(graph, &self.definitions, &self.slots, &self.initial);
        self.route_domains = domains::RouteDomains::prepare(
            graph,
            &self.transitions,
            &self.optimization.numeric,
            &self.initial,
        );
        self.numeric_cost_order =
            dominance::NumericCostOrder::new(&self.definitions, &self.slots, self.initial.len())
                .with_horizon(self.optimization.continuation_hops, &self.definitions)
                .with_graph_context(graph, &self.definitions);
        for activity in std::sync::Arc::make_mut(&mut self.pricing_activities) {
            activity.refresh_bounds(graph);
        }
        self.identity = std::sync::Arc::new(());
    }

    pub(crate) fn prepare<const D: usize>(
        g: &Graph<D>,
        definitions: &[CoupledResource],
    ) -> Result<Self> {
        let mut initial = Vec::new();
        let mut slots = Vec::new();
        for (i, r) in definitions.iter().enumerate() {
            r.validate(g, i, definitions)?;
            match r {
                CoupledResource::CustomResource { .. } => {
                    slots.push(Some(initial.len()));
                    initial.push(0.);
                }
                CoupledResource::CumulativeCost { initial: x, .. }
                | CoupledResource::Dependent { initial: x, .. }
                | CoupledResource::PiecewiseLinearDependent { initial: x, .. }
                | CoupledResource::PiecewiseLinearResource { initial: x, .. }
                | CoupledResource::EncodedPiecewiseLinearResource { initial: x, .. }
                | CoupledResource::EncodedPiecewiseLinearDependent { initial: x, .. } => {
                    slots.push(Some(initial.len()));
                    initial.push(*x);
                }
                _ => slots.push(None),
            }
        }
        let optimization = PreparedBounds::new(g, definitions, &slots, &initial);
        let numeric_cost_order =
            dominance::NumericCostOrder::new(definitions, &slots, initial.len())
                .with_horizon(optimization.continuation_hops, definitions)
                .with_graph_context(g, definitions);
        let objective_coefficients: std::sync::Arc<[f64]> = vec![1.; definitions.len()].into();
        let transitions = PreparedTransitions::new(
            definitions,
            &slots,
            initial.len(),
            g.arcs.len(),
            1.,
            &objective_coefficients,
        );
        let route_domains =
            domains::RouteDomains::prepare(g, &transitions, &optimization.numeric, &initial);
        let front_indices: Vec<_> = definitions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| matches!(r, CoupledResource::IntervalCost { .. }).then_some(i))
            .collect();
        let slope_indices: Vec<_> = definitions
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                matches!(r, CoupledResource::SlopeIntervalCost { .. }).then_some(i)
            })
            .collect();
        let slope_roots = slope_indices
            .iter()
            .map(|&i| {
                let CoupledResource::SlopeIntervalCost { initial, .. } = &definitions[i] else {
                    unreachable!()
                };
                descriptor::SlopeFront::zero(initial.lower, initial.upper)
                    .map(|front| slope_interval::SlopeLabel::root(&front, true))
                    .map_err(|error| invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let slope_charges = slope_indices
            .iter()
            .map(|&i| {
                let CoupledResource::SlopeIntervalCost { functions, .. } = &definitions[i] else {
                    unreachable!()
                };
                functions
                    .iter()
                    .map(slope_interval::PreparedSlopeCharge::new)
                    .collect::<Result<Vec<_>>>()
                    .map(std::sync::Arc::<[slope_interval::PreparedSlopeCharge]>::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let slope_dominance = slope_indices
            .iter()
            .map(|&i| slope_interval::PreparedSlopeDominance::new(&definitions[i]))
            .collect();
        let frozen = frozen::PreparedFrozen::new(g, definitions, &front_indices);
        let slope_paired = definitions
            .iter()
            .filter_map(|resource| match resource {
                CoupledResource::SlopePaired {
                    initial,
                    initial_base,
                    arcs,
                    mode,
                    normalization,
                    policy,
                } => Some(slope_paired::PreparedSlopePaired::new(
                    initial,
                    *initial_base,
                    arcs,
                    *mode,
                    *normalization,
                    *policy,
                )),
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        let paired = definitions
            .iter()
            .filter_map(|r| match r {
                CoupledResource::PairedPiecewiseLinear {
                    initial,
                    initial_base,
                    arcs,
                    mode,
                    normalization,
                } => Some(paired::PreparedPaired::new(
                    initial,
                    *initial_base,
                    arcs,
                    *mode,
                    *normalization,
                )),
                _ => None,
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            reversal_union: definitions.iter().any(|resource| match resource {
                CoupledResource::SimultaneousPickupDelivery {
                    deliveries,
                    pickups,
                    ..
                } => {
                    deliveries.iter().zip(pickups).any(|(d, p)| p > d)
                        && deliveries.iter().zip(pickups).any(|(d, p)| p < d)
                }
                _ => false,
            }),
            identity: std::sync::Arc::new(()),
            definitions: definitions.to_vec().into(),
            slots,
            initial,
            cost_scale: 1.,
            objective_coefficients,
            weighted_objective: false,
            negative_paired_objective: false,
            pricing_adjustments: Default::default(),
            pricing_activities: Default::default(),
            front_program_domain: false,
            frozen,
            paired,
            plugins: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::Plugin(plugin) => Some(plugin.clone()),
                    _ => None,
                })
                .collect(),
            front_bucket_policy: Default::default(),
            selection_initial: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::Selection {
                        initial, vertices, ..
                    } => Some(initial & vertices.get(g.source).copied().unwrap_or(u64::MAX)),
                    _ => None,
                })
                .collect(),
            selection_heads: g.arcs.iter().map(|a| a.head).collect(),
            selection_source: g.source,
            selection_memory: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::Selection {
                        initial,
                        vertices,
                        relaxed,
                        ..
                    } => Some((*relaxed || !vertices.is_empty()).then(|| {
                        selection::PreparedSelection::new(g, vertices, *initial, *relaxed)
                    })),
                    _ => None,
                })
                .collect(),
            load_capacities: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::SimultaneousPickupDelivery {
                        capacity,
                        tolerance,
                        ..
                    } => Some(capacity + tolerance),
                    _ => None,
                })
                .collect(),
            load_tolerances: definitions
                .iter()
                .filter_map(|r| match r {
                    CoupledResource::SimultaneousPickupDelivery { tolerance, .. } => {
                        Some(*tolerance)
                    }
                    _ => None,
                })
                .collect(),
            budgeted: budgeted::PreparedBudgeted::prepare(g, definitions),
            relation_indices: definitions
                .iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    matches!(r, CoupledResource::PairedPiecewiseLinear { .. }).then_some(i)
                })
                .collect(),
            front_indices,
            slope_indices,
            slope_roots,
            slope_charges,
            slope_dominance,
            slope_paired,
            cost_resource: definitions.iter().any(|r| {
                matches!(
                    r,
                    CoupledResource::CumulativeCost { .. }
                        | CoupledResource::CustomResource { .. }
                        | CoupledResource::IntervalCost { .. }
                        | CoupledResource::SlopeIntervalCost { .. }
                        | CoupledResource::PiecewiseLinearCost { .. }
                        | CoupledResource::EncodedPiecewiseLinearCost { .. }
                        | CoupledResource::SlopePaired {
                            mode: PairedResourceMode::CostUnbounded
                                | PairedResourceMode::CostBounded,
                            ..
                        }
                        | CoupledResource::PairedPiecewiseLinear {
                            mode: PairedResourceMode::CostUnbounded
                                | PairedResourceMode::CostBounded,
                            ..
                        }
                ) || matches!(r, CoupledResource::Plugin(p) if p.0.has_costs())
            }),
            arc_count: g.arcs.len(),
            optimization,
            numeric_cost_order,
            transitions,
            route_domains,
        })
    }
    /// Configure optional borrowed paired-front indexing views. Zero disables
    /// subdivision; no label, path, or objective is split or renormalized.
    pub fn configure_paired_front_views(
        &mut self,
        pieces_per_view: usize,
        max_views: usize,
    ) -> Result<()> {
        self.front_bucket_policy = match (
            std::num::NonZeroUsize::new(pieces_per_view),
            std::num::NonZeroUsize::new(max_views),
        ) {
            (Some(pieces_per_view), Some(max_views)) => bucketing::FrontBucketPolicy::PieceGroups {
                pieces_per_view,
                max_views,
            },
            (None, None) => bucketing::FrontBucketPolicy::Disabled,
            _ => {
                return Err(invalid(
                    "paired view width and budget must both be positive, or both zero",
                ))
            }
        };
        Ok(())
    }
    /// Conservative complete-route preprocessing diagnostics. A work-limit
    /// stop leaves safe, potentially loose domains and does not prove failure.
    pub fn preprocessing(&self) -> CoupledPreprocessing {
        CoupledPreprocessing {
            passes: self.route_domains.passes,
            converged: self.route_domains.converged,
            arcs: self.arc_count,
            usable_arcs: self
                .route_domains
                .possible
                .iter()
                .filter(|&&possible| possible)
                .count(),
        }
    }
    /// Whether preprocessing retained this physical arc for a complete route.
    /// True is a necessary condition, not a feasibility certificate.
    pub fn route_arc_possible(&self, arc: usize) -> Result<bool> {
        self.route_domains
            .possible
            .get(arc)
            .copied()
            .ok_or_else(|| invalid("unknown coupled resource arc"))
    }
    pub(crate) fn cache_identity(&self) -> &std::sync::Arc<()> {
        &self.identity
    }
    pub fn definitions(&self) -> &[CoupledResource] {
        &self.definitions
    }
    pub(crate) fn validate_for<const D: usize>(&self, g: &Graph<D>) -> Result<()> {
        for (index, definition) in self.definitions.iter().enumerate() {
            definition.validate(g, index, &self.definitions)?;
        }
        let optimization = PreparedBounds::new(g, &self.definitions, &self.slots, &self.initial);
        if self.arc_count != g.arcs.len()
            || self.optimization != optimization
            || !self.route_domains.matches(g)
        {
            return Err(invalid("reinstall coupled resources after changing elementarity or physical graph topology"));
        }
        Ok(())
    }
    pub fn has_costs(&self) -> bool {
        (self.cost_scale != 0. && self.cost_resource)
            || self.pricing_activities.iter().any(CoupledKernel::has_costs)
    }
    /// Whether a raw resource selects the native machine-front contract.
    ///
    /// Native raw costs use their ordered binary64 front contract, including
    /// alignment and slope merges. Default feasibility still requires an
    /// independent original source-domain proof. Only the explicit graph
    /// machine-domain model choice also defines feasibility by that program.
    /// Neither scope asserts an exact-real control optimum. Values combine with
    /// other resource charges using outward aggregate bounds; this flag does
    /// not make a rounded global objective an exact MIP coefficient.
    pub fn has_machine_front_contracts(&self) -> bool {
        self.definitions.iter().any(|definition| {
            matches!(
                definition,
                CoupledResource::SlopeIntervalCost {
                    policy: FrontOperationPolicy::FrontAlgebra,
                    ..
                } | CoupledResource::SlopePaired {
                    policy: FrontOperationPolicy::FrontAlgebra,
                    ..
                }
            )
        })
    }
    /// The explicit model choice used for native raw resource feasibility.
    pub fn front_program_domain(&self) -> bool {
        self.front_program_domain
    }
    /// Whether any objective is defined by a computed raw machine front.
    /// Source-control abstract objective bounds cannot be substituted for these
    /// costs, even when default raw feasibility has a separate source witness.
    pub fn has_machine_cost_contracts(&self) -> bool {
        self.definitions.iter().any(|definition| match definition {
            CoupledResource::SlopeIntervalCost { .. } => true,
            CoupledResource::SlopePaired {
                policy: FrontOperationPolicy::FrontAlgebra,
                mode,
                ..
            } => self.front_program_domain && *mode != PairedResourceMode::Feasibility,
            _ => false,
        })
    }
    #[cfg(feature = "clp")]
    pub(crate) fn suppress_costs(&mut self) {
        self.identity = std::sync::Arc::new(());
        self.cost_scale = 0.;
        self.pricing_adjustments = Default::default();
        self.pricing_activities = Default::default();
        self.transitions.suppress_costs(self.initial.len());
    }
}

impl<const D: usize> Graph<D> {
    pub(crate) fn refresh_coupled_bounds(&mut self) {
        if let Some(mut kernel) = self.coupled.take() {
            kernel.refresh_bounds(self);
            self.coupled = Some(kernel);
        }
    }
}
