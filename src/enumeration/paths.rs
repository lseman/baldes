//! Completed route storage and equivalence classes.
use super::*;
use std::{collections::HashMap, sync::Arc};
// Native retained-path export uses this hash domain; dominance storage can
// choose a smaller partition count independently.
const EXPORT_PARTITIONS: u32 = 2_000_000;
#[derive(Clone, Debug)]
pub struct Path {
    pub reduced_cost: f64,
    pub original_cost: f64,
    pub visited: [u64; 16],
    pub element_hash: u32,
    pub arcs: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct PathStorage {
    pub partitions: Vec<Vec<Path>>,
    pub len: usize,
    pub checks: u64,
    /// Arc-flow branching distinguishes orderings of the same customer set.
    pub(super) preserve_arc_paths: bool,
    export_partitions: Vec<u32>,
    export_ranks: std::collections::HashMap<u32, usize>,
    // One index supplies both deduplication and canonical insertion order.
    // Forks share its immutable keys until a child changes membership.
    path_order: Arc<HashMap<Arc<[usize]>, usize>>,
    next_path: usize,
}
impl PathStorage {
    pub fn new(partitions: usize) -> Result<Self> {
        if partitions == 0 {
            return Err(invalid("positive enumeration path partitions required"));
        }
        Ok(Self {
            partitions: vec![vec![]; partitions],
            len: 0,
            checks: 0,
            preserve_arc_paths: false,
            export_partitions: vec![],
            export_ranks: std::collections::HashMap::new(),
            path_order: Arc::default(),
            next_path: 0,
        })
    }
    pub fn insert(&mut self, path: Path) -> bool {
        let export_partition = path.element_hash % EXPORT_PARTITIONS;
        if !self.export_ranks.contains_key(&export_partition) {
            self.export_ranks
                .insert(export_partition, self.export_partitions.len());
            self.export_partitions.push(export_partition);
        }
        let partition = path.element_hash as usize % self.partitions.len();
        if self.preserve_arc_paths {
            if self.path_order.contains_key(path.arcs.as_slice()) {
                return false;
            }
            Arc::make_mut(&mut self.path_order)
                .insert(Arc::from(path.arcs.as_slice()), self.next_path);
            self.next_path += 1;
            self.partitions[partition].push(path);
            self.len += 1;
            return true;
        }
        let row = &mut self.partitions[partition];
        for previous in row.iter() {
            if previous.original_cost > path.original_cost + EPSILON {
                break;
            }
            self.checks += 1;
            if previous.visited == path.visited {
                return false;
            }
        }
        let before = row.len();
        row.retain(|p| {
            self.checks += 1;
            if p.visited == path.visited {
                Arc::make_mut(&mut self.path_order).remove(p.arcs.as_slice());
                false
            } else {
                true
            }
        });
        self.len -= before - row.len();
        let at = row.partition_point(|p| p.original_cost <= path.original_cost + EPSILON);
        Arc::make_mut(&mut self.path_order).insert(Arc::from(path.arcs.as_slice()), self.next_path);
        self.next_path += 1;
        row.insert(at, path);
        self.len += 1;
        true
    }
    /// Path repricing: physical arc costs, then rank-one costs from
    /// physical incidence (no memory).
    pub fn inspect<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
    ) -> Result<usize> {
        self.inspect_with_knapsack(g, cutoff, cuts, kpath, &Default::default())
    }
    pub fn inspect_with_knapsack<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        knapsack: &crate::cuts::knapsack::KnapsackPricing,
    ) -> Result<usize> {
        self.inspect_with_presence(g, cutoff, cuts, kpath, knapsack, &Default::default())
    }

    pub fn inspect_with_presence<const D: usize>(
        &mut self,
        g: &Graph<D>,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        knapsack: &crate::cuts::knapsack::KnapsackPricing,
        presence: &crate::cuts::soft_conflicts::PresencePricing,
    ) -> Result<usize> {
        if cutoff.is_nan() {
            return Err(invalid("invalid enumeration inspection cutoff"));
        }
        let before = self.len;
        for row in &mut self.partitions {
            for path in row.iter_mut() {
                path.reduced_cost = path.arcs.iter().map(|&a| g.arcs[a].reduced_cost).sum();
                path.reduced_cost += g
                    .coupled_path_cost(&path.arcs)?
                    .ok_or_else(|| invalid("enumerated path violates coupled resources"))?;
                if !cuts.is_empty() || !kpath.is_empty() {
                    let mut vertices = vec![g.source];
                    vertices.extend(path.arcs.iter().map(|&a| g.arcs[a].head));
                    let ids: Vec<i32> = path.arcs.iter().map(|&a| g.arcs[a].id).collect();
                    path.reduced_cost = state::enumerated_path_cut_cost(
                        cuts,
                        kpath,
                        &vertices,
                        &ids,
                        path.reduced_cost,
                    );
                }
                if presence.active() {
                    path.reduced_cost += presence.route_cost(&path.arcs)?;
                }
                if knapsack.active {
                    let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
                    for r in 0..g.resources.len() {
                        q[r] = g
                            .root_resource(r, true)
                            .ok_or_else(|| invalid("initial resource outside source windows"))?;
                    }
                    for &arc in &path.arcs {
                        q = extend_resources(g, &g.arcs[arc], &q, true)
                            .ok_or_else(|| invalid("enumerated knapsack resource infeasible"))?;
                    }
                    path.reduced_cost += knapsack.cost(&q);
                }
            }
            row.retain(|p| p.reduced_cost < cutoff);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
        if self.len != before {
            self.retain_path_order();
        }
        Ok(before - self.len)
    }
    pub fn retain(&mut self, mut keep: impl FnMut(&Path) -> bool) {
        let before = self.len;
        for partition in &mut self.partitions {
            partition.retain(&mut keep);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
        if self.len != before {
            self.retain_path_order();
        }
    }
    pub fn routes<const D: usize>(&self, g: &Graph<D>) -> Result<Vec<Route>> {
        self.ordered_paths()
            .into_iter()
            .map(|path| Self::route(g, path))
            .collect()
    }
    /// Rank retained paths without replaying every resource schedule. The
    /// minimum describes the retained domain independently of output limits.
    pub(crate) fn priced_routes<const D: usize>(
        &self,
        g: &Graph<D>,
        maximum: usize,
        negative_only: bool,
        zero_threshold: f64,
    ) -> Result<(Vec<Route>, f64)> {
        let mut paths = self.ordered_paths();
        // Stable sorting preserves canonical export order for equal costs,
        // including the two signed zeros.
        paths.sort_by(|a, b| {
            a.reduced_cost
                .partial_cmp(&b.reduced_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let minimum = paths
            .first()
            .map_or(crate::labeling::NO_ROUTE_COST, |path| path.reduced_cost);
        let routes = paths
            .into_iter()
            .filter(|path| !negative_only || path.reduced_cost < zero_threshold - EPSILON)
            .take(maximum)
            .map(|path| Self::route(g, path))
            .collect::<Result<_>>()?;
        Ok((routes, minimum))
    }
    fn ordered_paths(&self) -> Vec<&Path> {
        // Native export visits hash partitions in activation order, then
        // their candidates in original-cost insertion order. Dominance may
        // use fewer physical partitions without changing this path index.
        let mut paths: Vec<_> = self.partitions.iter().flatten().collect();
        paths.sort_unstable_by_key(|path| self.path_order[path.arcs.as_slice()]);
        let mut ordered: Vec<Vec<&Path>> = vec![vec![]; self.export_partitions.len()];
        for path in paths {
            let partition = self.export_ranks[&(path.element_hash % EXPORT_PARTITIONS)];
            let row = &mut ordered[partition];
            let position = row
                .partition_point(|previous| previous.original_cost <= path.original_cost + EPSILON);
            row.insert(position, path);
        }
        ordered.into_iter().flatten().collect()
    }
    fn route<const D: usize>(g: &Graph<D>, path: &Path) -> Result<Route> {
        g.coupled_path_cost(&path.arcs)?
            .ok_or_else(|| invalid("enumerated route violates coupled resources"))?;
        let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
        for r in 0..g.resources.len() {
            q[r] = g
                .root_resource(r, true)
                .ok_or_else(|| invalid("initial resource outside source windows"))?;
        }
        let mut nodes = vec![g.source as i32];
        let mut resource_profile = vec![q[..g.resources.len()].to_vec()];
        for &id in &path.arcs {
            let a = &g.arcs[id];
            q = extend_resources(g, a, &q, true)
                .ok_or_else(|| invalid("enumerated route is no longer resource feasible"))?;
            nodes.push(a.head as i32);
            resource_profile.push(q[..g.resources.len()].to_vec());
        }
        Ok(Route {
            nodes,
            arcs: path.arcs.iter().map(|&a| g.arcs[a].id).collect(),
            cost: path.reduced_cost,
            original_cost: path.original_cost,
            resources: q[..g.resources.len()].to_vec(),
            resource_profile,
        })
    }
    fn retain_path_order(&mut self) {
        let live: HashSet<_> = self
            .partitions
            .iter()
            .flatten()
            .map(|path| path.arcs.as_slice())
            .collect();
        Arc::make_mut(&mut self.path_order).retain(|arcs, _| live.contains(arcs.as_ref()));
    }
}

#[cfg(test)]
#[path = "../../tests/unit/enumeration/paths/order_tests.rs"]
mod order_tests;
