use super::*;

/// Pricing keys are the integer cut
/// breakpoints; scaling is applied by separation and master coefficients.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KnapsackPricing {
    pub active: bool,
    pub resources: BTreeMap<usize, BTreeMap<i32, f64>>,
}
impl KnapsackPricing {
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
    /// resConsKnapsackCutsDualsCost: forward consumption, or sink upper
    /// minus the backward label's remaining-resource coordinate.
    pub fn partial_cost(&self, g: &crate::Graph, resources: &[f64], forward: bool) -> f64 {
        if forward {
            return self.cost(resources);
        }
        let mut consumption = [0.; 2];
        for &r in self.resources.keys() {
            consumption[r] = g.vertices[g.sink].bounds[r].upper - resources[r];
        }
        self.cost(&consumption)
    }
    /// Native join consumption: forward + (sink upper - backward), or
    /// forward + forward - initial usage for symmetric consumption search.
    pub fn join_cost(
        &self,
        g: &crate::Graph,
        forward: &[f64],
        backward: &[f64],
        symmetric: bool,
    ) -> f64 {
        let mut consumption = [0.; 2];
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
