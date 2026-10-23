//! Solve enumerated integer masters and verify original-domain encoding.
use super::*;

// Proof outcomes require the finite LP encoding to preserve original costs,
// coefficients and fixed reductions. Witness replay cannot detect a route
// which the rounded model excluded before returning any solution.
#[cfg(feature = "highs")]
fn exact_integer_encoding<const D: usize>(
    state: &State<D>,
    master: &ClpMaster,
    columns: &[Option<cg::Column>],
    deadline: Instant,
) -> Result<bool> {
    use crate::master::root::ExactSum;
    if !super::super::bounds::exact_fixed_reduction(state, master)? {
        return Ok(false);
    }
    for (column, metadata) in columns.iter().enumerate() {
        if Instant::now() >= deadline {
            return Ok(false);
        }
        if master.upper[column] == 0. {
            continue;
        }
        let Some(metadata) = metadata else {
            return Ok(false);
        };
        let mut cost = ExactSum::default();
        let mut rows = BTreeMap::<usize, ExactSum>::new();
        if let Some((index, sign)) = super::super::master_variables::decode(metadata.tag) {
            let variable = &state.master_variables[index];
            if cost.product(sign, variable.cost).is_none() {
                return Ok(false);
            }
            for &(row, a) in &variable.entries {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .product(sign, a)
                    .is_none()
                {
                    return Ok(false);
                }
            }
        } else {
            let Some(route) = usize::try_from(metadata.tag)
                .ok()
                .and_then(|tag| state.routes.get(tag))
            else {
                return Ok(false);
            };
            let Some(oracle) = state.oracles.iter().find(|o| o.network.id == route.network) else {
                return Ok(false);
            };
            if let Some(kernel) = oracle.network.graph.coupled_kernel() {
                let arcs: Vec<_> = route.route.arcs.iter().map(|id| oracle.index[id]).collect();
                let Some((true, bounds)) = kernel.certify(&arcs) else {
                    return Ok(false);
                };
                if bounds[0] != bounds[1] || cost.add(bounds[0]).is_none() {
                    return Ok(false);
                }
            }
            for &arc in &route.route.arcs {
                let position = oracle.index[&arc];
                if cost.add(oracle.network.graph.arcs[position].cost).is_none() {
                    return Ok(false);
                }
                for &(row, a) in &oracle.network.arc_entries[position] {
                    if rows
                        .entry(state.shared_offset + row)
                        .or_default()
                        .add(a)
                        .is_none()
                    {
                        return Ok(false);
                    }
                }
            }
            for &(row, a) in &oracle.route_entries {
                if rows.entry(row).or_default().add(a).is_none() {
                    return Ok(false);
                }
            }
            for &(row, a) in oracle.network.route_entries.iter() {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .add(a)
                    .is_none()
                {
                    return Ok(false);
                }
            }
            for (row, a) in super::super::nonlinear::entries(oracle, &route.route)? {
                if rows
                    .entry(state.shared_offset + row)
                    .or_default()
                    .add(a)
                    .is_none()
                {
                    return Ok(false);
                }
            }
        }
        if cost.add(-master.costs[column]).is_none() || cost.sign() != 0. {
            return Ok(false);
        }
        for (row, a) in master.column_entries(column)? {
            if rows.entry(row).or_default().add(-a).is_none() {
                return Ok(false);
            }
        }
        if rows.values().any(|value| value.sign() != 0.) {
            return Ok(false);
        }
    }
    Ok(true)
}

impl<const D: usize> NodeSnapshot<D> {
    /// HiGHS solves only fully transferred copies. No source model is mutated.
    pub fn solve_enumerated_mip(
        &self,
        options: &MipOptions,
    ) -> Result<Option<EnumeratedMipOutput>> {
        if self.state.universe.is_none() {
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
            for &tag in self.state.universe.as_ref().unwrap() {
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
            let encoding_exact = exact_integer_encoding(
                &self.state,
                &master,
                &columns,
                started
                    .checked_add(options.time_limit)
                    .ok_or_else(|| invalid("MIP deadline"))?,
            )?;
            let mut options = options.clone();
            options.time_limit = options.time_limit.saturating_sub(started.elapsed());
            let mut mip =
                crate::master::solvers::highs::solve_integer(&master, &integers, &options)?;
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
                    if let Some(upper) =
                        super::super::tree::validate_routes(&self.state, &routes, &master_values)?
                    {
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
                            mip.solution.as_mut().unwrap().objective = upper;
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
                coverage_cutoff: f64::INFINITY,
            }))
        }
    }
}
