use super::*;

/// Pricing keys are the integer cut
/// breakpoints; scaling is applied by separation and master coefficients.
#[derive(Clone, Debug, Default)]
pub struct KnapsackPricing {
    pub active: bool,
    pub resources: BTreeMap<usize, BTreeMap<i32, f64>>,
    /// Scaled, unrounded penalties for coefficient-consistent pricing.
    exact_terms: Vec<(usize, f64, RouteLoadKnapsackCut, f64)>,
    source: Option<std::sync::Arc<PricingSource>>,
}
// Empty registries are common. Cache their result, including public map state,
// without adding snapshot copies to the changing-dual path for active cuts.
#[derive(Debug)]
struct PricingSource {
    graph: i32,
    exact: bool,
    resource_count: usize,
    mapping: BTreeMap<usize, usize>,
    output: KnapsackPricing,
}
impl PartialEq for KnapsackPricing {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            active,
            resources,
            exact_terms,
            source: _,
        } = self;
        active == &other.active
            && resources == &other.resources
            && exact_terms == &other.exact_terms
    }
}
impl KnapsackPricing {
    /// Solver refresh preserves the previous committed cache on invalid input.
    /// The empty explicit mapping denotes identity solver coordinates.
    pub(crate) fn refresh_cached(
        &mut self,
        cuts: &[ResourceKnapsackCut],
        graph: i32,
        explicit_mapping: &BTreeMap<usize, usize>,
        resource_count: usize,
        exact: bool,
    ) -> Result<bool> {
        if explicit_mapping
            .values()
            .any(|&slot| slot >= resource_count)
        {
            return Err(invalid(
                "knapsack resource mapping exceeds solver coordinates",
            ));
        }
        if cuts.is_empty()
            && self.source.as_ref().is_some_and(|source| {
                source.graph == graph
                    && source.exact == exact
                    && source.resource_count == resource_count
                    && source.mapping == *explicit_mapping
                    && source.output == *self
            })
        {
            return Ok(false);
        }
        let identity;
        let mapping = if explicit_mapping.is_empty() {
            identity = (0..resource_count)
                .map(|resource| (resource, resource))
                .collect();
            &identity
        } else {
            explicit_mapping
        };
        let mut next = self.clone();
        // Do not retain an obsolete snapshot through the new output snapshot.
        next.source = None;
        if exact {
            next.retrieve_exact(cuts, graph, mapping)?;
        } else {
            next.retrieve(cuts, graph, mapping)?;
        }
        let changed = next != *self;
        if cuts.is_empty() {
            next.source = Some(std::sync::Arc::new(PricingSource {
                graph,
                exact,
                resource_count,
                mapping: explicit_mapping.clone(),
                output: next.clone(),
            }));
        }
        *self = next;
        Ok(changed)
    }
    /// Prepare nonnegative pricing penalties using the master coefficient's
    /// graph scale and unrounded dual. Unlike `retrieve`, no small dual is
    /// discarded. Positive master duals require a separate dominance contract.
    /// Failure leaves the previous cache unchanged.
    pub fn retrieve_exact(
        &mut self,
        cuts: &[ResourceKnapsackCut],
        graph: i32,
        resource_indices: &BTreeMap<usize, usize>,
    ) -> Result<()> {
        let mut next = Self::default();
        for &slot in resource_indices.values() {
            next.resources.entry(slot).or_default();
        }
        for cut in cuts {
            let Some(&(resource, scale)) = cut.graph_resources.get(&graph) else {
                continue;
            };
            if !cut.dual.is_finite() || cut.dual > 0. || !scale.is_finite() || scale <= 0. {
                return Err(invalid("exact knapsack dual or scale"));
            }
            let &slot = resource_indices
                .get(&resource)
                .ok_or_else(|| invalid("exact knapsack resource mapping"))?;
            let mut previous = 0;
            for &coefficient in cut.cut.coefficients.values() {
                if coefficient < previous {
                    return Err(invalid(
                        "exact knapsack coefficients must be nonnegative and monotone",
                    ));
                }
                previous = coefficient;
            }
            if cut.dual != 0. {
                next.active = true;
                next.exact_terms
                    .push((slot, scale, cut.cut.clone(), -cut.dual));
            }
        }
        *self = next;
        Ok(())
    }
    pub fn prepare(
        cuts: &[ResourceKnapsackCut],
        graph: i32,
        resource_count: usize,
    ) -> Result<Self> {
        let mapping = (0..resource_count).map(|r| (r, r)).collect();
        let mut result = Self::default();
        result.retrieve(cuts, graph, &mapping)?;
        Ok(result)
    }
    /// Refresh native cache slots in place. Resource IDs in cut associations
    /// are external IDs; `resource_indices` maps them to solver coordinates.
    /// A failed retrieval leaves the already processed state.
    pub fn retrieve(
        &mut self,
        cuts: &[ResourceKnapsackCut],
        graph: i32,
        resource_indices: &BTreeMap<usize, usize>,
    ) -> Result<()> {
        self.exact_terms.clear();
        for &resource in resource_indices.values() {
            self.resources.entry(resource).or_default();
        }
        for steps in self.resources.values_mut() {
            steps.clear();
        }
        self.active = false;
        for cut in cuts {
            if !cut.dual.is_finite() || cut.dual > 1e-6 {
                return Err(invalid("knapsack master dual must be non-positive"));
            }
            if -cut.dual <= 1e-6 {
                continue;
            }
            let Some(&(resource, _)) = cut.graph_resources.get(&graph) else {
                continue;
            };
            self.active = true;
            let Some(&resource) = resource_indices.get(&resource) else {
                return Err(invalid("knapsack pricing resource index"));
            };
            let dual = (-cut.dual * 1e8 + 0.5).floor() / 1e8;
            let steps = self.resources.entry(resource).or_default();
            let mut previous = 0;
            for (&load, &coefficient) in &cut.cut.coefficients {
                let value = steps.range(..=load).next_back().map_or(0., |(_, &v)| v);
                steps.entry(load).or_insert(value);
                let delta = (coefficient - previous) as f64 * dual;
                for (_, cost) in steps.range_mut(load..) {
                    *cost += delta;
                }
                previous = coefficient;
            }
        }
        Ok(())
    }
    pub fn cost(&self, resources: &[f64]) -> f64 {
        let mut result = 0.;
        for (resource, scale, cut, penalty) in &self.exact_terms {
            let load = (resources[*resource] * scale + 1e-6).floor() as i32;
            result += cut.coefficient(load) as f64 * penalty;
        }
        for (&resource, steps) in &self.resources {
            if steps.is_empty() {
                continue;
            }
            let load = (resources[resource] + 1e-6).floor() as i32;
            if let Some((_, &cost)) = steps.range(..=load).next_back() {
                result += cost;
            }
        }
        result
    }
    /// Lower bound for any complete join, independent of resource values.
    /// Include zero because a load below every breakpoint contributes zero.
    pub(crate) fn global_cost_lower_bound(&self) -> Option<f64> {
        use crate::core::rounding::{add_down, multiply_down};
        let mut lower = 0.0;
        for (_, _, cut, penalty) in &self.exact_terms {
            if !penalty.is_finite() {
                return None;
            }
            let mut term = 0.0f64;
            for &coefficient in cut.coefficients.values() {
                term = term.min(multiply_down(coefficient as f64, *penalty)?);
            }
            lower = add_down(lower, term)?;
        }
        for steps in self.resources.values() {
            let mut term = 0.0f64;
            for &value in steps.values() {
                if !value.is_finite() {
                    return None;
                }
                term = term.min(value);
            }
            lower = add_down(lower, term)?;
        }
        Some(lower)
    }
    /// Resources with a potentially varying cut cost. Label dominance permits
    /// a resource tolerance, which can cross even an increasing breakpoint.
    pub(crate) fn dominance_protected_resources(&self) -> Vec<usize> {
        let mut protected = std::collections::BTreeSet::new();
        for (&resource, steps) in &self.resources {
            for &value in steps.values() {
                if !value.is_finite() || value != 0.0 {
                    protected.insert(resource);
                    break;
                }
            }
        }
        for (resource, scale, cut, penalty) in &self.exact_terms {
            if !scale.is_finite() || *scale <= 0. || !penalty.is_finite() {
                protected.insert(*resource);
                continue;
            }
            for &coefficient in cut.coefficients.values() {
                let value = coefficient as f64 * penalty;
                if !value.is_finite() || value != 0.0 {
                    protected.insert(*resource);
                    break;
                }
            }
        }
        protected.into_iter().collect()
    }
    /// resConsKnapsackCutsDualsCost: forward consumption, or sink upper
    /// minus the backward label's remaining-resource coordinate.
    pub fn partial_cost<const D: usize>(
        &self,
        g: &crate::Graph<D>,
        resources: &[f64],
        forward: bool,
    ) -> f64 {
        if forward {
            return self.cost(resources);
        }
        let mut consumption = crate::graph::ResourceValues::<D>::filled(g.resources.len(), 0.);
        for &r in self.resources.keys() {
            consumption[r] = g.vertices[g.sink].bounds[r].upper - resources[r];
        }
        self.cost(&consumption)
    }
    /// Native join consumption: forward + (sink upper - backward), or
    /// forward + forward - initial usage for symmetric consumption search.
    pub fn join_cost<const D: usize>(
        &self,
        g: &crate::Graph<D>,
        forward: &[f64],
        backward: &[f64],
        symmetric: bool,
    ) -> f64 {
        let mut consumption = crate::graph::ResourceValues::<D>::filled(g.resources.len(), 0.);
        for &r in self.resources.keys() {
            consumption[r] = if symmetric {
                forward[r] + backward[r]
                    - g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower)
            } else {
                (g.vertices[g.sink].bounds[r].upper - backward[r]) + forward[r]
            };
        }
        self.cost(&consumption)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuts/knapsack/pricing_cache_tests.rs"]
mod tests;
