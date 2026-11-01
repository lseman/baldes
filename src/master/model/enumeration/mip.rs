//! Solve enumerated integer masters and verify original-domain encoding.
use super::*;

// Proof outcomes require the finite LP encoding to preserve original costs,
// coefficients and fixed reductions. Witness replay cannot detect a route
// which the rounded model excluded before returning any solution.
#[cfg(feature = "highs")]
/// Worst mismatch between the MIP model and the exact original encoding of
/// every loaded column: 0 when exact, relative to each cost and absolute per
/// row coefficient, or infinity when it cannot be evaluated.
fn integer_encoding_error<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    columns: &[Option<cg::Column>],
    deadline: Instant,
) -> Result<f64> {
    use crate::master::numerics::ExactSum;
    if !super::super::bounds::exact_fixed_reduction(state, master)? {
        return Ok(f64::INFINITY);
    }
    let mut proof_work = 1_000_000;
    let mut worst = 0f64;
    for (column, metadata) in columns.iter().enumerate() {
        if Instant::now() >= deadline {
            return Ok(f64::INFINITY);
        }
        if master.upper[column] == 0. {
            continue;
        }
        let Some(metadata) = metadata else {
            return Ok(f64::INFINITY);
        };
        let mut cost = ExactSum::default();
        let mut rows = BTreeMap::<usize, ExactSum>::new();
        if let Some((index, sign)) = super::super::master_variables::decode(metadata.tag) {
            let variable = &state.master_variables[index];
            if cost.product(sign, variable.cost).is_none() {
                return Ok(f64::INFINITY);
            }
            for &(row, a) in &variable.entries {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .product(sign, a)
                    .is_none()
                {
                    return Ok(f64::INFINITY);
                }
            }
        } else {
            let Some(route) = usize::try_from(metadata.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            else {
                return Ok(f64::INFINITY);
            };
            let Some(oracle) = state.oracles.iter().find(|o| o.network.id == route.network) else {
                return Ok(f64::INFINITY);
            };
            if !oracle.activity_rows.is_empty()
                && super::super::bounds::columns::exact_activity_entries_using(
                    oracle,
                    &route.route,
                    &mut proof_work,
                    deadline,
                )
                .is_none()
            {
                return Ok(f64::INFINITY);
            }
            if let Some(kernel) = oracle.network.graph.coupled_kernel() {
                let arcs: Vec<_> = route.route.arcs.iter().map(|id| oracle.index[id]).collect();
                let Some((true, bounds)) =
                    kernel.certify_with_budget(&arcs, &mut proof_work, Some(deadline))
                else {
                    return Ok(f64::INFINITY);
                };
                if bounds[0] != bounds[1] || cost.add(bounds[0]).is_none() {
                    return Ok(f64::INFINITY);
                }
            }
            for &arc in &route.route.arcs {
                let position = oracle.index[&arc];
                if cost.add(oracle.network.graph.arcs[position].cost).is_none() {
                    return Ok(f64::INFINITY);
                }
                for &(row, a) in &oracle.network.arc_entries[position] {
                    if rows
                        .entry(state.shared_offset + row)
                        .or_default()
                        .add(a)
                        .is_none()
                    {
                        return Ok(f64::INFINITY);
                    }
                }
            }
            for &(row, a) in &oracle.route_entries {
                if rows.entry(row).or_default().add(a).is_none() {
                    return Ok(f64::INFINITY);
                }
            }
            for &(row, a) in oracle.network.route_entries.iter() {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .add(a)
                    .is_none()
                {
                    return Ok(f64::INFINITY);
                }
            }
            for (row, a) in super::super::nonlinear::entries(oracle, &route.route)? {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .add(a)
                    .is_none()
                {
                    return Ok(f64::INFINITY);
                }
            }
        }
        if cost.add(-master.costs[column]).is_none() {
            return Ok(f64::INFINITY);
        }
        worst = worst.max(cost.sign().abs() / master.costs[column].abs().max(1.));
        for (row, a) in master.column_entries(column)? {
            if rows.entry(row).or_default().add(-a).is_none() {
                return Ok(f64::INFINITY);
            }
        }
        for value in rows.values() {
            worst = worst.max(value.sign().abs());
        }
    }
    Ok(if Instant::now() < deadline {
        worst
    } else {
        f64::INFINITY
    })
}

impl<const D: usize> NodeSnapshot<D> {
    /// Restricted-master integer heuristic (native
    /// `RestrictedMasterIpHeuristic`): the node's current columns, with
    /// artificial columns at zero, solved as an integer program on a copy.
    /// A solution is accepted only after exact replay in the original model;
    /// its objective is that replay's upward bound. The MIP bound is never
    /// used, because absent columns can be cheaper.
    pub fn solve_restricted_mip(
        &self,
        options: &MipOptions,
    ) -> Result<Option<(f64, Vec<FixedRoute>, Vec<(i32, f64)>)>> {
        #[cfg(not(feature = "highs"))]
        {
            let _ = options;
            Err(invalid("restricted master MIP requires the highs feature"))
        }
        #[cfg(feature = "highs")]
        {
            let mut master = self.master.try_clone()?;
            for &column in &self.warm.artificials {
                master.set_upper(column, 0.)?;
            }
            let columns = &self.warm.columns;
            let integers = columns
                .iter()
                .enumerate()
                .filter_map(|(c, meta)| {
                    meta.as_ref().and_then(|m| {
                        let integer = super::super::master_variables::decode(m.tag)
                            .map_or(true, |(index, _)| {
                                self.state.master_variables[index].integer
                            });
                        integer.then_some(c)
                    })
                })
                .collect::<Vec<_>>();
            let mip = solve_pool(&master, &integers, options)?;
            let Some(solution) = mip.solution else {
                return Ok(None);
            };
            let mut routes = self.state.fixed_routes.clone();
            let mut master_values: Vec<_> = self
                .state
                .master_variables
                .iter()
                .map(|v| (v.id, 0.))
                .collect();
            for (column, metadata) in columns.iter().enumerate() {
                let value = solution.values[column];
                if value == 0. {
                    continue;
                }
                let Some(metadata) = metadata else {
                    return Ok(None);
                };
                let static_variable = super::super::master_variables::decode(metadata.tag);
                let integer = static_variable.map_or(true, |(index, _)| {
                    self.state.master_variables[index].integer
                });
                if value < 0.
                    || integer && value.fract() != 0.
                    || value > master.upper[column]
                    || value > 9_007_199_254_740_992.
                    || metadata.tag < 0
                {
                    return Ok(None);
                }
                if let Some((index, sign)) = static_variable {
                    master_values[index].1 += sign * value;
                    continue;
                }
                routes.push(FixedRoute {
                    route: self.state.routes[metadata.tag as usize].as_ref().clone(),
                    quantity: value,
                });
            }
            let Some(routes) = synchronization::witness(&self.state, &routes)? else {
                return Ok(None);
            };
            Ok(super::super::tree_adapter::validate_routes(
                &self.state,
                &routes,
                &mut master_values,
            )?
            .map(|objective| (objective, routes, master_values)))
        }
    }

    /// Native end-of-node MIP route threshold: `ControlParameters::postTreatment`
    /// raises `maxNumOfEnumSolsForEndOfNodeMIP` to `maxNumOfEnumSolutionsForMIP`.
    /// With several networks, the tightest network threshold applies.
    pub(crate) fn end_of_node_mip_route_limit(&self) -> Result<usize> {
        let mut limit = usize::MAX;
        for oracle in &self.state.oracles {
            #[cfg(feature = "highs")]
            let (normal, end) = (
                oracle
                    .solver
                    .options
                    .get("BALDESmaxNumOfEnumSolutionsForMIP")?,
                oracle
                    .solver
                    .options
                    .get("BALDESmaxNumOfEnumSolsForEndOfNodeMIP")?,
            );
            #[cfg(not(feature = "highs"))]
            let (normal, end): (f64, f64) = {
                let _ = oracle;
                (10_000., 0.)
            };
            if !(normal >= 0. && end >= 0.) {
                return Err(invalid("enumerated MIP route limits must be nonnegative"));
            }
            limit = limit.min(normal.max(end) as usize);
        }
        Ok(limit)
    }
    /// Routes in this node's complete universe or retained gap pool.
    pub(crate) fn enumerated_route_count(&self) -> usize {
        if self.state.enumeration_report.complete {
            self.state.enumeration_report.routes
        } else {
            self.state.enumerated_routes
        }
    }

    /// HiGHS solves only fully transferred copies. No source model is mutated.
    pub fn solve_enumerated_mip(
        &self,
        options: &MipOptions,
    ) -> Result<Option<EnumeratedMipOutput>> {
        // Before start domains are fixed, the finite master is only a
        // synchronization relaxation. Leave it to the native time tree.
        if self
            .state
            .synchronized_services
            .iter()
            .any(|s| s.domain[0] != s.domain[1])
        {
            return Ok(None);
        }
        if self.state.universe.is_none() && self.state.gap_pool_cutoff.is_none() {
            return Ok(None);
        }
        #[cfg(not(feature = "highs"))]
        {
            let _ = options;
            Err(invalid("enumerated MIP requires the highs feature"))
        }
        #[cfg(feature = "highs")]
        {
            use crate::master::solvers::mip::MipStatus;
            let started = Instant::now();
            let mut master = self.master.try_clone()?;
            let mut columns = self.warm.columns.clone();
            let mut present: HashSet<_> = columns
                .iter()
                .flatten()
                .filter(|c| c.tag >= 0)
                .map(|c| c.tag as usize)
                .collect();
            // Cleanup may retire a route from the restricted LP without
            // changing complete-domain coverage. Restore every missing path
            // before claiming an integer optimum for that complete domain.
            let gap_routes;
            let universe = if let Some(universe) = &self.state.universe {
                universe
            } else {
                gap_routes = self
                    .state
                    .routes
                    .iter()
                    .enumerate()
                    .filter(|(_, route)| {
                        self.state
                            .oracles
                            .iter()
                            .find(|oracle| oracle.network.id == route.network)
                            .is_some_and(|oracle| oracle.is_enumerated(&route.route))
                    })
                    .map(|(tag, _)| tag)
                    .collect::<Vec<_>>();
                &gap_routes
            };
            for &tag in universe {
                if !present.insert(tag) {
                    continue;
                }
                let route = &self.state.routes[tag];
                let oracle = self
                    .state
                    .oracles
                    .iter()
                    .find(|o| o.network.id == route.network)
                    .unwrap();
                let metadata = cg::Column {
                    cost: route.route.original_cost,
                    reduced_cost: 0.,
                    entries: entries(oracle, &route.route, self.state.shared_offset)?,
                    acceptable: true,
                    tag: tag as i32,
                    identity: format!("{}:{}", route.network, route.route.identity()),
                };
                master.add_column(metadata.cost, &metadata.entries)?;
                columns.push(Some(metadata));
            }
            for (column, metadata) in columns.iter().enumerate() {
                if let Some(metadata) = metadata {
                    let cap = if metadata.tag >= 0 {
                        route_upper(&self.state, metadata.tag as usize)
                    } else {
                        super::super::master_variables::decode(metadata.tag).map(|(index, sign)| {
                            super::super::master_variables::upper(
                                &self.state.master_variables[index],
                                sign,
                            )
                        })
                    };
                    if let Some(cap) = cap {
                        if cap < master.upper[column] {
                            master.set_upper(column, cap)?;
                        }
                    }
                }
            }
            for &column in &self.warm.artificials {
                master.set_upper(column, 0.)?;
            }
            let integers = columns
                .iter()
                .enumerate()
                .filter_map(|(c, meta)| {
                    meta.as_ref().and_then(|m| {
                        let integer = super::super::master_variables::decode(m.tag)
                            .map_or(true, |(index, _)| {
                                self.state.master_variables[index].integer
                            });
                        integer.then_some(c)
                    })
                })
                .collect::<Vec<_>>();
            let encoding_error = integer_encoding_error(
                &self.state,
                &master,
                &columns,
                started
                    .checked_add(options.time_limit)
                    .ok_or_else(|| invalid("MIP deadline"))?,
            )?;
            let mut options = options.clone();
            options.time_limit = options.time_limit.saturating_sub(started.elapsed());
            let mut mip = solve_pool(&master, &integers, &options)?;
            // A gap pool covers only objectives below its cutoff: always
            // weaken the MIP claims accordingly.
            if let Some(cutoff) = self.state.gap_pool_cutoff {
                mip.bound = mip.bound.min(cutoff);
                if mip.status == MipStatus::Infeasible {
                    mip.status = MipStatus::ObjectiveCutoff;
                    mip.bound = cutoff;
                }
            }
            // Certified mode needs the exact original encoding. Numerical mode
            // tolerates floating rounding of costs and coefficients, never a
            // materially different model (e.g. cancellation).
            let encoding_exact = match self.warm.bound_mode {
                cg::BoundMode::Certified => encoding_error == 0.,
                cg::BoundMode::Numerical => encoding_error <= 1e-9,
            };
            if !encoding_exact {
                mip.bound = f64::NEG_INFINITY;
                if matches!(
                    mip.status,
                    MipStatus::Optimal | MipStatus::Infeasible | MipStatus::ObjectiveCutoff
                ) {
                    mip.status = MipStatus::Incomplete;
                }
            }
            let mut routes = vec![];
            let mut master_values: Vec<_> = self
                .state
                .master_variables
                .iter()
                .map(|v| (v.id, 0.))
                .collect();
            if let Some(solution) = mip.solution.clone() {
                for (column, metadata) in columns.iter().enumerate() {
                    let value = solution.values[column];
                    if value == 0. {
                        continue;
                    }
                    let Some(metadata) = metadata else {
                        return Err(invalid("MIP artificial witness"));
                    };
                    let static_variable = super::super::master_variables::decode(metadata.tag);
                    let integer = static_variable.map_or(true, |(index, _)| {
                        self.state.master_variables[index].integer
                    });
                    if value < 0.
                        || integer && value.fract() != 0.
                        || value > master.upper[column]
                        || value > 9_007_199_254_740_992.
                    {
                        mip.solution = None;
                        mip.status = MipStatus::Incomplete;
                        mip.bound = f64::NEG_INFINITY;
                        break;
                    }
                    if let Some((index, sign)) = static_variable {
                        master_values[index].1 += sign * value;
                        continue;
                    }
                    routes.push(FixedRoute {
                        route: self.state.routes[metadata.tag as usize].as_ref().clone(),
                        quantity: value,
                    });
                }
                routes.extend(self.state.fixed_routes.clone());
                if mip.solution.is_some() {
                    if let Some(scheduled) = synchronization::witness(&self.state, &routes)? {
                        routes = scheduled;
                    } else {
                        mip.solution = None;
                        mip.status = MipStatus::Incomplete;
                        mip.bound = f64::NEG_INFINITY;
                    }
                }
                if mip.solution.is_some() {
                    let encoded_master_values = master_values.clone();
                    if let Some(upper) = super::super::tree_adapter::validate_routes(
                        &self.state,
                        &routes,
                        &mut master_values,
                    )? {
                        let objective = mip.solution.as_ref().unwrap().objective;
                        if (upper - objective).abs()
                            > 1e-6 + 1e-10 * upper.abs().max(objective.abs())
                        {
                            // A rounded matrix objective is not a proof about
                            // a materially different original arc-cost sum.
                            mip.status = MipStatus::Incomplete;
                            mip.bound = f64::NEG_INFINITY;
                        }
                        if upper < options.objective_cutoff {
                            let solution = mip.solution.as_mut().unwrap();
                            // Replay can raise Theta to pay exact recourse.
                            // Publish the same repaired static witness in both
                            // original-variable and backend-column coordinates.
                            for (column, metadata) in columns.iter().enumerate() {
                                let Some((index, sign)) = metadata.as_ref().and_then(|metadata| {
                                    super::super::master_variables::decode(metadata.tag)
                                }) else {
                                    continue;
                                };
                                let value = master_values[index].1;
                                if value != encoded_master_values[index].1 {
                                    solution.values[column] = (sign * value).max(0.);
                                }
                            }
                            solution.objective = upper;
                        } else {
                            // Original-unit replay could not establish a strict
                            // improvement, even though the backend rounded down.
                            mip.solution = None;
                            mip.status = MipStatus::Incomplete;
                            mip.bound = f64::NEG_INFINITY;
                        }
                    } else {
                        mip.solution = None;
                        mip.status = MipStatus::Incomplete;
                        mip.bound = f64::NEG_INFINITY;
                    }
                }
                if mip.solution.is_none() {
                    routes.clear();
                    master_values.clear();
                }
            }
            if mip.solution.is_none() {
                routes.clear();
                master_values.clear();
            }
            Ok(Some(EnumeratedMipOutput {
                mip,
                route_values: routes,
                master_values,
                coverage_cutoff: self.state.gap_pool_cutoff.unwrap_or(f64::INFINITY),
            }))
        }
    }
}

/// Complete-pool integer solve. DeLuxing only removes columns that cannot
/// occur in a solution at most the cutoff, so it requires every column to be
/// integer and a finite cutoff; otherwise the plain solve is used.
#[cfg(feature = "highs")]
fn solve_pool<M: crate::master::LpMaster>(
    master: &M,
    integers: &[usize],
    options: &MipOptions,
) -> Result<crate::master::solvers::mip::MipOutput> {
    match &options.deluxing {
        Some(fixing)
            if options.objective_cutoff.is_finite() && integers.len() == master.costs().len() =>
        {
            Ok(
                crate::master::solvers::highs::solve_integer_with_deluxing(
                    master, options, fixing,
                )?
                .0,
            )
        }
        _ => crate::master::solvers::highs::solve_integer(master, integers, options),
    }
}
