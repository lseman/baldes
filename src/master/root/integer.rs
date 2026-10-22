//! Complete enumerated routing masters, retaining branch and cut rows.
use super::*;
#[cfg(feature = "highs")]
use crate::master::solvers::mip::MipOptions;
use crate::master::solvers::mip::MipOutput;
#[cfg(feature = "highs")]
use std::collections::HashMap;
#[cfg(feature = "highs")]
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumerationMipPolicy {
    pub normal_route_limit: usize,
    /// ControlParameters::postTreatment raises this to the normal limit.
    pub end_of_node_route_limit: usize,
}
impl EnumerationMipPolicy {
    pub fn from_options(options: &crate::core::options::Options) -> Result<Self> {
        #[cfg(feature = "highs")]
        let (normal, end) = (
            options.get("BALDESmaxNumOfEnumSolutionsForMIP")?,
            options.get("BALDESmaxNumOfEnumSolsForEndOfNodeMIP")?,
        );
        #[cfg(not(feature = "highs"))]
        let (normal, end): (f64, f64) = {
            let _ = options;
            (10_000., 0.)
        };
        if normal < 0. || end < 0. {
            return Err(invalid("enumerated MIP route limits must be nonnegative"));
        }
        Ok(Self {
            normal_route_limit: normal as usize,
            end_of_node_route_limit: normal.max(end) as usize,
        })
    }
    /// None means that at least one pricing problem is not fully enumerated.
    /// Equality is accepted, including a completed empty universe at limit 0.
    pub fn eligible(&self, count: Option<usize>, end_of_node: bool) -> bool {
        count.is_some_and(|n| {
            n <= if end_of_node {
                self.end_of_node_route_limit
            } else {
                self.normal_route_limit
            }
        })
    }
    /// Alg4EvalByMip::setupAlgo reserves 2% of the remaining global budget.
    pub fn time_limit(
        remaining: std::time::Duration,
        configured: std::time::Duration,
    ) -> std::time::Duration {
        configured.min(remaining.mul_f64(0.98))
    }
}
#[derive(Debug)]
pub struct EnumeratedMipOutput {
    pub mip: MipOutput,
    /// Positive integer route values; artificial columns are fixed to zero.
    pub route_values: Vec<(Route, f64)>,
    /// Enumeration may cover only solutions improving this incumbent. A proof
    /// about this matrix cannot exclude solutions above this coverage cutoff.
    pub coverage_cutoff: f64,
}
impl<const D: usize> NodeSnapshot<D> {
    pub fn enumeration_mip_policy(&self) -> Result<EnumerationMipPolicy> {
        EnumerationMipPolicy::from_options(&self.pricing.options)
    }
    /// Solve a complete enumerated copy at node completion. Returns None until
    /// enumeration is complete or if its count exceeds the normalized limit.
    /// The source LP, basis, cuts, branch rows and siblings are never mutated.
    #[cfg(feature = "highs")]
    pub fn solve_enumerated_mip(
        &self,
        options: &MipOptions,
    ) -> Result<Option<EnumeratedMipOutput>> {
        self.solve_enumerated_mip_impl(options, None)
    }

    /// Opt-in DeLuxing filtering of the complete enumerated route matrix.
    /// Requires a finite `options.objective_cutoff`; proof outcomes apply to
    /// improvements below that cutoff. The live node is untouched.
    #[cfg(feature = "highs")]
    pub fn solve_enumerated_mip_with_deluxing(
        &self,
        options: &MipOptions,
        fixing: &crate::master::deluxing::DeluxingOptions,
    ) -> Result<Option<EnumeratedMipOutput>> {
        self.solve_enumerated_mip_impl(options, Some(fixing))
    }

    #[cfg(feature = "highs")]
    fn solve_enumerated_mip_impl(
        &self,
        options: &MipOptions,
        fixing: Option<&crate::master::deluxing::DeluxingOptions>,
    ) -> Result<Option<EnumeratedMipOutput>> {
        let started = Instant::now();
        let Some(universe) = self.pricing.enumeration.as_ref().filter(|e| e.completed) else {
            return Ok(None);
        };
        if !self
            .enumeration_mip_policy()?
            .eligible(Some(universe.paths.len), true)
        {
            return Ok(None);
        }
        let mut routes = universe.paths.routes(&self.pricing.graph)?;
        // Enumeration may deliberately exclude incumbent routes at its cutoff.
        // Retain previously priced elementary routes as possible witnesses too;
        // adding valid columns cannot weaken the universe's coverage guarantee.
        let mut seen: std::collections::HashSet<_> =
            routes.iter().map(|r| r.arcs.clone()).collect();
        for route in &self.routes {
            if let Some(checked) = self.problem.elementary_route(&route.arcs) {
                if seen.insert(checked.arcs.clone()) {
                    routes.push(checked);
                }
            }
        }
        let mut model = crate::master::solvers::highs::HighsMaster::new()?;
        // Copy the current physical row order, including inherited branches
        // and every active cut. The source Clp model and its basis stay intact.
        for (&sense, &rhs) in self.master.senses.iter().zip(&self.master.rhs) {
            model.add_row(sense, rhs, &[])?;
        }
        model.objective_offset = self.problem.fixed_cost();
        let mut upper_by_path: HashMap<Vec<i32>, f64> = HashMap::new();
        for (c, metadata) in self.warm.columns.iter().enumerate() {
            if let Some(column) = metadata {
                if column.tag >= 0 {
                    let route = self
                        .routes
                        .get(column.tag as usize)
                        .ok_or_else(|| invalid("snapshot route tag"))?;
                    let upper = *self
                        .master
                        .upper
                        .get(c)
                        .ok_or_else(|| invalid("snapshot column bound"))?;
                    upper_by_path
                        .entry(route.arcs.clone())
                        .and_modify(|x| *x = x.min(upper))
                        .or_insert(upper);
                }
            }
        }
        // Existing route columns may carry limited-memory coefficients. Rebuild
        // all enumerated routes at full memory. Old routes and artificials are
        // omitted (equivalent to fixing them to zero); every row remains.
        let mut route_columns = vec![];
        for route in routes {
            if !self
                .pricing
                .route_satisfies_relaxation(&route.nodes, &route.arcs)?
            {
                continue;
            }
            let mut entries = self.problem.base_entries(&route)?;
            entries.extend(self.cut.entries(&self.pricing, &route, true)?);
            entries.sort_by_key(|&(r, _)| r);
            let (cost, _) = self.problem.route_cost(&route);
            let c = model.add_column(cost, &entries)?;
            if let Some(&upper) = upper_by_path.get(&route.arcs) {
                model.set_upper(c, upper)?;
            }
            route_columns.push((c, route));
        }
        let integers: Vec<_> = route_columns.iter().map(|&(c, _)| c).collect();
        let mut remaining = options.clone();
        remaining.time_limit = options.time_limit.saturating_sub(started.elapsed());
        let mip = if let Some(fixing) = fixing {
            crate::master::solvers::highs::solve_integer_with_deluxing(&model, &remaining, fixing)?
                .0
        } else {
            crate::master::solvers::highs::solve_integer(&model, &integers, &remaining)?
        };
        let route_values = mip.solution.as_ref().map_or_else(Vec::new, |s| {
            route_columns
                .into_iter()
                .filter_map(|(c, r)| (s.values[c] > 0.).then_some((r, s.values[c])))
                .chain(
                    self.problem
                        .fixed_routes
                        .iter()
                        .map(|(r, q)| (r.clone(), f64::from(*q))),
                )
                .collect()
        });
        Ok(Some(EnumeratedMipOutput {
            mip,
            route_values,
            coverage_cutoff: if self.incumbent.is_finite() {
                self.incumbent
            } else {
                1e12
            },
        }))
    }
}

#[cfg(all(test, feature = "highs"))]
#[path = "../../../tests/unit/master/root/integer/tests.rs"]
mod tests;
