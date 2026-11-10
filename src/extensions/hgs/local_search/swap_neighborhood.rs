//! Independent SWAP* candidate lists and periodic unfiltered passes.

use super::Node;
use crate::extensions::hgs::Params;

pub(super) struct SwapNeighborhood {
    neighbors: Vec<Vec<usize>>,
    interval: usize,
    passes: usize,
    broad: bool,
    enabled: bool,
}

impl SwapNeighborhood {
    pub fn new(params: &Params) -> Self {
        let n = params.nb_clients as usize;
        let limit = params.config.swap_star_neighbors;
        let interval = params.config.swap_star_full_pass_interval;
        let enabled =
            limit > 0 && interval > 1 && params.config.intensification_probability_ls != 0;
        let mut neighbors = vec![Vec::new(); n + 1];
        if enabled {
            let percentage = params.config.swap_star_demand_percentage.min(100) as usize;
            let mut ranked = Vec::with_capacity(n.saturating_sub(1));
            for i in 1..=n {
                ranked.clear();
                for j in (1..=n).filter(|&j| j != i) {
                    let difference =
                        (i64::from(params.cli[i].demand) - i64::from(params.cli[j].demand)).abs();
                    ranked.push((difference, params.time_cost.get(i, j), j));
                }
                // Exact ceil(len * percentage / 100), without multiplication overflow.
                let retained = ranked.len() / 100 * percentage
                    + ((ranked.len() % 100) * percentage).div_ceil(100);
                if retained < ranked.len() {
                    ranked
                        .select_nth_unstable_by(retained, |a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
                    ranked.truncate(retained);
                }
                if limit < ranked.len() {
                    ranked.select_nth_unstable_by(limit, |a, b| {
                        a.1.total_cmp(&b.1).then(a.2.cmp(&b.2))
                    });
                    ranked.truncate(limit);
                }
                // An exchange is an unordered pair. Include the union of both
                // directed selections, also for asymmetric distance matrices.
                for &(_, _, j) in &ranked {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
            for list in &mut neighbors {
                list.sort_unstable();
                list.dedup();
            }
        }
        Self {
            neighbors,
            interval,
            passes: 0,
            broad: true,
            enabled,
        }
    }

    /// Route timestamps were reset by load_individual. Keep the cadence across
    /// calls, but the next phase does not inherit a filtered search timestamp.
    pub fn reset_run(&mut self) {
        self.broad = true;
    }

    /// Return whether a broad pass must ignore timestamps from a filtered pass.
    /// Count actual intensification passes, rather than ordinary LS calls.
    pub fn begin_pass(&mut self) -> bool {
        let previous_broad = self.broad;
        self.broad = !self.enabled || self.passes.is_multiple_of(self.interval);
        self.passes = self.passes.wrapping_add(1);
        self.broad && !previous_broad
    }

    pub fn candidates<'a>(
        &'a self,
        customer: usize,
        first_in_route: usize,
        route: usize,
        nodes: &'a [Node],
    ) -> impl Iterator<Item = usize> + 'a {
        let mut next = first_in_route;
        let mut neighbors = self.neighbors[customer].iter();
        std::iter::from_fn(move || {
            if self.broad {
                if nodes[next].is_depot {
                    return None;
                }
                let current = next;
                next = nodes[current].next as usize;
                Some(current)
            } else {
                neighbors
                    .find(|&&v| nodes[v].route_idx as usize == route)
                    .copied()
            }
        })
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/hgs/local_search/swap_neighborhood.rs"]
mod tests;
