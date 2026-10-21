//! Completed route storage and equivalence classes.
use super::*;
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
    arc_identities: HashSet<Vec<usize>>,
    export_partitions: Vec<u32>,
    export_ranks: std::collections::HashMap<u32, usize>,
    path_order: std::collections::HashMap<Vec<usize>, usize>,
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
            arc_identities: HashSet::new(),
            export_partitions: vec![],
            export_ranks: std::collections::HashMap::new(),
            path_order: std::collections::HashMap::new(),
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
            if !self.arc_identities.insert(path.arcs.clone()) {
                return false;
            }
            self.path_order.insert(path.arcs.clone(), self.next_path);
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
                self.path_order.remove(&p.arcs);
                false
            } else {
                true
            }
        });
        self.len -= before - row.len();
        let at = row.partition_point(|p| p.original_cost <= path.original_cost + EPSILON);
        self.path_order.insert(path.arcs.clone(), self.next_path);
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
                        q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
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
        self.retain_path_order();
        if self.preserve_arc_paths {
            self.arc_identities = self
                .partitions
                .iter()
                .flatten()
                .map(|p| p.arcs.clone())
                .collect();
        }
        Ok(before - self.len)
    }
    pub fn retain(&mut self, mut keep: impl FnMut(&Path) -> bool) {
        for partition in &mut self.partitions {
            partition.retain(&mut keep);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
        self.retain_path_order();
        if self.preserve_arc_paths {
            self.arc_identities = self
                .partitions
                .iter()
                .flatten()
                .map(|p| p.arcs.clone())
                .collect();
        }
    }
    pub fn routes<const D: usize>(&self, g: &Graph<D>) -> Result<Vec<Route>> {
        // Native export visits hash partitions in activation order, then
        // their candidates in original-cost insertion order. Dominance may
        // use fewer physical partitions without changing this path index.
        let mut paths: Vec<_> = self.partitions.iter().flatten().collect();
        paths.sort_unstable_by_key(|path| self.path_order[&path.arcs]);
        let mut ordered: Vec<Vec<&Path>> = vec![vec![]; self.export_partitions.len()];
        for path in paths {
            let partition = self.export_ranks[&(path.element_hash % EXPORT_PARTITIONS)];
            let row = &mut ordered[partition];
            let position = row
                .partition_point(|previous| previous.original_cost <= path.original_cost + EPSILON);
            row.insert(position, path);
        }
        let mut result = vec![];
        for path in ordered.into_iter().flatten() {
            let mut q = crate::graph::ResourceValues::filled(g.resources.len(), 0.);
            for r in 0..g.resources.len() {
                q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
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
            result.push(Route {
                nodes,
                arcs: path.arcs.iter().map(|&a| g.arcs[a].id).collect(),
                cost: path.reduced_cost,
                original_cost: path.original_cost,
                resources: q[..g.resources.len()].to_vec(),
                resource_profile,
            });
        }
        Ok(result)
    }
    fn retain_path_order(&mut self) {
        let live: HashSet<_> = self
            .partitions
            .iter()
            .flatten()
            .map(|path| path.arcs.as_slice())
            .collect();
        self.path_order
            .retain(|arcs, _| live.contains(arcs.as_slice()));
    }
}

#[cfg(test)]
mod order_tests {
    use super::*;

    #[test]
    fn export_order_survives_partition_tuning_replacement_retention_and_forks() {
        let mut graph = Graph::<2>::new(4, 1).unwrap();
        for (tail, head, cost) in [(0, 1, 1.), (1, 3, 1.), (0, 2, 2.), (2, 3, 2.), (1, 2, 1.)] {
            graph.add_arc(tail, head, cost).unwrap();
        }
        for count in [1, 8] {
            let mut storage = PathStorage::new(count).unwrap();
            let make = |arcs: Vec<usize>, hash, visited, cost| {
                let mut bits = [0; 16];
                bits[0] = visited;
                Path {
                    arcs,
                    element_hash: hash,
                    visited: bits,
                    original_cost: cost,
                    reduced_cost: cost,
                }
            };
            storage.insert(make(vec![0, 1], 1, 2, 2.));
            storage.insert(make(vec![2, 3], 0, 1, 4.));
            storage.insert(make(vec![0, 4, 3], 1, 3, 4.));
            let paths = |s: &PathStorage| {
                s.routes(&graph)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.arcs)
                    .collect::<Vec<_>>()
            };
            assert_eq!(paths(&storage), [vec![0, 1], vec![0, 4, 3], vec![2, 3]]);
            let mut child = storage.clone();
            child.retain(|p| p.arcs != [0, 1]);
            assert_eq!(paths(&child), [vec![0, 4, 3], vec![2, 3]]);
            assert!(child.insert(make(vec![0, 4, 3], 1, 3, 3.5)));
            assert_eq!(paths(&child), [vec![0, 4, 3], vec![2, 3]]);
            assert_eq!(paths(&storage).len(), 3);
            child.inspect(&graph, 3., &[], &[]).unwrap();
            assert!(paths(&child).is_empty());
            assert!(child.path_order.is_empty());
        }
    }
}
