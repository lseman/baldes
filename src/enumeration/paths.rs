//! Completed route storage and equivalence classes.
use super::*;
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
        })
    }
    pub fn insert(&mut self, path: Path) -> bool {
        let partition = path.element_hash as usize % self.partitions.len();
        if self.preserve_arc_paths {
            if !self.arc_identities.insert(path.arcs.clone()) {
                return false;
            }
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
            p.visited != path.visited
        });
        self.len -= before - row.len();
        let at = row.partition_point(|p| p.original_cost <= path.original_cost + EPSILON);
        row.insert(at, path);
        self.len += 1;
        true
    }
    /// Path repricing: physical arc costs, then rank-one costs from
    /// physical incidence (no memory).
    pub fn inspect(
        &mut self,
        g: &Graph,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
    ) -> Result<usize> {
        self.inspect_with_knapsack(g, cutoff, cuts, kpath, &Default::default())
    }
    pub fn inspect_with_knapsack(
        &mut self,
        g: &Graph,
        cutoff: f64,
        cuts: &[LimitedInput],
        kpath: &[LimitedKPathInput],
        knapsack: &crate::cuts::knapsack::KnapsackPricing,
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
                if knapsack.active {
                    let mut q = [0.; 2];
                    for r in 0..g.resources.len() {
                        q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
                    }
                    for &arc in &path.arcs {
                        q = extend_resources(g, &g.arcs[arc], q, true)
                            .ok_or_else(|| invalid("enumerated knapsack resource infeasible"))?;
                    }
                    path.reduced_cost += knapsack.cost(&q);
                }
            }
            row.retain(|p| p.reduced_cost < cutoff);
        }
        self.len = self.partitions.iter().map(Vec::len).sum();
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
        if self.preserve_arc_paths {
            self.arc_identities = self
                .partitions
                .iter()
                .flatten()
                .map(|p| p.arcs.clone())
                .collect();
        }
    }
    pub fn routes(&self, g: &Graph) -> Result<Vec<Route>> {
        let mut result = vec![];
        for path in self.partitions.iter().flatten() {
            let mut q = [0.; 2];
            for r in 0..g.resources.len() {
                q[r] = g.initial[r].unwrap_or(g.vertices[g.source].bounds[r].lower);
            }
            let mut nodes = vec![g.source as i32];
            for &id in &path.arcs {
                let a = &g.arcs[id];
                q = extend_resources(g, a, q, true)
                    .ok_or_else(|| invalid("enumerated route is no longer resource feasible"))?;
                nodes.push(a.head as i32);
            }
            result.push(Route {
                nodes,
                arcs: path.arcs.iter().map(|&a| g.arcs[a].id).collect(),
                cost: path.reduced_cost,
                original_cost: path.original_cost,
                resources: q[..g.resources.len()].to_vec(),
            });
        }
        Ok(result)
    }
}
