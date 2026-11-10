//! Bounded records of completed RI searches, keyed by exact directed plan arcs.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::extensions::hgs::Individual;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct SearchContext {
    pub capacity_penalty: u64,
    pub time_warp_penalty: u64,
    pub granular: i32,
    pub best_move: bool,
}

struct SearchedPlan {
    context: SearchContext,
    /// (first customer, route index during search), sorted by first customer.
    starts: Box<[(usize, usize)]>,
}

/// An entry proves only the ordinary RI neighborhood was searched. SWAP* and
/// empty-route moves remain eligible independently. Exact arc keys avoid hash
/// collisions creating false matches; FIFO eviction affects reuse, not results.
pub(super) struct SearchedPlans {
    plans: HashMap<Rc<[usize]>, SearchedPlan>,
    order: VecDeque<Rc<[usize]>>,
    limit: usize,
}

impl SearchedPlans {
    pub fn new(limit: usize) -> Self {
        Self {
            plans: HashMap::new(),
            order: VecDeque::new(),
            limit: limit.max(1),
        }
    }

    pub fn remember(
        &mut self,
        individual: &Individual<'_>,
        context: SearchContext,
        mut starts: Vec<(usize, usize)>,
    ) {
        starts.sort_unstable_by_key(|&(first, _)| first);
        let plan = SearchedPlan {
            context,
            starts: starts.into_boxed_slice(),
        };
        if let Some(old) = self.plans.get_mut(individual.successors.as_slice()) {
            *old = plan;
            return;
        }
        if self.order.len() == self.limit {
            let expired = self.order.pop_front().unwrap();
            self.plans.remove(expired.as_ref());
        }
        let arcs: Rc<[usize]> = individual.successors.as_slice().into();
        self.plans.insert(Rc::clone(&arcs), plan);
        self.order.push_back(arcs);
    }

    /// Each slot stores a route's search-time index in one of the two parents.
    /// A suffix, reversal, or route assembled from both parents is not inherited.
    pub fn classify(
        &self,
        individual: &Individual<'_>,
        parents: &[&Individual<'_>],
        context: SearchContext,
        inherited: &mut [[Option<usize>; 2]],
    ) {
        inherited.fill([None; 2]);
        for (source, parent) in parents.iter().take(2).enumerate() {
            let Some((arcs, plan)) = self.plans.get_key_value(parent.successors.as_slice()) else {
                continue;
            };
            if plan.context != context {
                continue;
            }
            for (r, route) in individual.chrom_r.iter().enumerate() {
                let Some(&first) = route.first() else {
                    continue;
                };
                let Ok(start) = plan
                    .starts
                    .binary_search_by_key(&first, |&(first, _)| first)
                else {
                    continue;
                };
                if route.iter().enumerate().all(|(i, &client)| {
                    arcs.get(client).copied() == Some(route.get(i + 1).copied().unwrap_or(0))
                }) {
                    inherited[r][source] = Some(plan.starts[start].1);
                }
            }
        }
    }
}
