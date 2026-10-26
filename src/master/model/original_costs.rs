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
