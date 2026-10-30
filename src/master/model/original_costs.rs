//! Original-space cached pricing, independent of raw physical proof costs.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct ArcCost {
    pub base: f64,
    pub entries: Vec<(usize, f64)>,
    pub compiled: f64,
    pub compiled_entries: Vec<(usize, f64)>,
}

/// Retained cost decomposition produced by the original-variable compiler.
/// Install through `PricingVariableRegistry::configure_pricing`; the raw
/// compiled model remains authoritative for objectives and proof bounds.
#[derive(Clone, Debug)]
pub struct OriginalVariableCosts {
    pub(super) network: i32,
    pub(super) variables: Vec<OriginalPricingVariable>,
    pub(super) arcs: HashMap<i32, ArcCost>,
}

impl OriginalVariableCosts {
    #[cfg(any(feature = "multi-time-windows", test))]
    pub(super) fn copy_arcs(&mut self, copies: impl Iterator<Item = (i32, i32)>) {
        let copies: Vec<_> = copies.collect();
        for &(copy, original) in &copies {
            if let Some(data) = self.arcs.get(&original).cloned() {
                self.arcs.insert(copy, data);
            }
        }
        for variable in &mut self.variables {
            let original = variable.arcs.clone();
            for &(copy, source) in &copies {
                if let Some(&(_, a)) = original.iter().find(|&&(id, _)| id == source) {
                    variable.arcs.push((copy, a));
                }
            }
        }
    }

    /// A changed coefficient prefix is outside this decomposition. Fall back
    /// to ordinary physical caching; appended rows are handled explicitly.
    pub(super) fn cached<const D: usize>(
        &self,
        network: &OracleNetwork<D>,
        index: &HashMap<i32, usize>,
        duals: &[f64],
        shared_offset: usize,
    ) -> Option<Vec<f64>> {
        let row_cost = |entries: &[(usize, f64)], mut cost: f64| {
            for &(row, a) in entries {
                cost -= duals.get(shared_offset.checked_add(row)?)? * a;
            }
            cost.is_finite().then_some(cost)
        };
        let mut costs = Vec::with_capacity(network.graph.arcs.len());
        for (arc, entries) in network.graph.arcs.iter().zip(network.arc_entries.iter()) {
            let saved = self.arcs.get(&arc.id)?;
            if !entries.starts_with(&saved.compiled_entries) {
                return None;
            }
            let base = saved.base + (arc.cost - saved.compiled);
            let cost = row_cost(&saved.entries, base)?;
            costs.push(row_cost(&entries[saved.compiled_entries.len()..], cost)?);
        }
        for variable in &self.variables {
            if variable.accumulated_resource.is_some() {
                continue;
            }
            let reduced = row_cost(&variable.entries, variable.cost)?;
            let rounded = (reduced * 1e8 + 0.5).floor() / 1e8;
            for &(id, a) in &variable.arcs {
                let p = *index.get(&id)?;
                let cost = costs.get_mut(p)?;
                *cost += rounded * a;
                if !cost.is_finite() {
                    return None;
                }
            }
        }
        Some(costs)
    }
}

/// Apply the owner-associated paired variable coefficients after raw resource
/// selection. The raw front program always retains its registered unit charge.
pub(super) fn route_cost<const D: usize>(oracle: &Oracle<D>, route: &Route) -> Result<f64> {
    let mut ids = route.arcs.clone();
    ids.sort_unstable();
    let positions = route
        .arcs
        .iter()
        .map(|id| oracle.index[id])
        .collect::<Vec<_>>();
    let physical = ids
        .iter()
        .map(|id| oracle.network.graph.arcs[oracle.index[id]].cost)
        .sum::<f64>();
    let associated = oracle
        .network
        .graph
        .coupled_kernel()
        .map(|kernel| kernel.paired_associated_costs())
        .unwrap_or_default();
    let resource = if associated.is_empty() {
        route.coupled_cost(&oracle.network.graph, &positions)?
    } else {
        let [(resource, coefficient)] = associated else {
            return Err(crate::incomplete(
                "multiple paired owner associations require multiple paired programs",
            ));
        };
        let selected = activity_rows::column_value(oracle, route, &positions, *resource)?;
        oracle
            .network
            .graph
            .coupled_kernel()
            .unwrap()
            .joined_machine_path_matching_with_paired_coefficient(
                &positions,
                Some(selected),
                *coefficient,
            )?
            .map(|(cost, _)| cost)
    }
    .ok_or_else(|| invalid("network path violates coupled resources"))?;
    let cost = physical + resource;
    if !cost.is_finite() {
        return Err(invalid("owner-associated route cost overflow"));
    }
    Ok(cost)
}

impl<const D: usize> State<D> {
    /// The paired resource operator ignores its associated variable's reduced
    /// coefficient. Such raw pricing cannot certify the owner objective/rows.
    pub(super) fn has_unpriced_paired_terms(&self, duals: &[f64]) -> bool {
        self.universe.is_none()
            && self.oracles.iter().any(|oracle| {
                if oracle.multiplicity_upper() == 0. {
                    return false;
                }
                let Some(kernel) = oracle.network.graph.coupled_kernel() else {
                    return false;
                };
                kernel
                    .definitions()
                    .iter()
                    .enumerate()
                    .any(|(resource, _)| {
                        if !activity_rows::paired(&oracle.network.graph, resource) {
                            return false;
                        }
                        let mut omitted = crate::master::numerics::ExactSum::default();
                        if omitted
                            .add(kernel.objective_coefficient(resource).unwrap_or(1.))
                            .is_none()
                            || omitted.add(-1.).is_none()
                        {
                            return true;
                        }
                        for entry in oracle
                            .activity_rows
                            .iter()
                            .filter(|entry| entry.resource == resource)
                        {
                            let Some(&dual) = duals.get(self.shared_offset + entry.row) else {
                                return true;
                            };
                            if omitted.product(-dual, entry.coefficient).is_none() {
                                return true;
                            }
                        }
                        omitted.sign() != 0.
                    })
            })
    }
}
