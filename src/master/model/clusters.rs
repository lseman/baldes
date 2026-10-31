//! Cluster branching generators from Silva, Uchoa and Subramanian (2024),
//! Sections 3–4, over one network's arc activities.
//!
//! A cluster boundary counts traversals of arcs entering a vertex group from
//! outside it; a cluster pair counts traversals between two disjoint groups,
//! in either direction. The source and sink together form the depot group.
//! Both quantities are integer for integer route quantities, including
//! repeated traversals, so each floor/ceiling split covers every solution.
use super::model_strong::{Proposal, Restriction};
use super::*;

/// Register cluster branching on one network. Without an explicit partition,
/// customers (vertices other than source and sink) are grouped by removing
/// minimum-spanning-tree edges longer than `mean + theta * std`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkClusterBranching {
    pub network: i32,
    #[serde(default)]
    pub partition: Option<Vec<Vec<usize>>>,
    #[serde(default = "default_theta")]
    pub theta: f64,
    #[serde(default = "default_priority")]
    pub priority: f64,
    /// Arc coefficients of each generator, in registration order.
    #[serde(skip)]
    prepared: Vec<(String, Vec<(i32, f64)>)>,
}

fn default_theta() -> f64 {
    1.
}

fn default_priority() -> f64 {
    10.
}

impl NetworkClusterBranching {
    pub fn new(network: i32) -> Self {
        Self {
            network,
            partition: None,
            theta: default_theta(),
            priority: default_priority(),
            prepared: vec![],
        }
    }

    /// Customer groups used by the generators: the explicit partition with
    /// sorted groups, or the automatic MST partition. A disconnected customer
    /// graph has no automatic partition and yields no customer groups.
    pub fn groups<const D: usize>(&self, graph: &Graph<D>) -> Result<Vec<Vec<usize>>> {
        if !self.theta.is_finite() || self.theta < 0. {
            return Err(invalid("cluster MST theta must be finite and nonnegative"));
        }
        let customers: Vec<usize> = (0..graph.vertices.len())
            .filter(|&v| v != graph.source && v != graph.sink)
            .collect();
        if let Some(groups) = &self.partition {
            let mut seen = HashSet::new();
            for group in groups {
                if group.is_empty()
                    || group
                        .iter()
                        .any(|v| !customers.contains(v) || !seen.insert(*v))
                {
                    return Err(invalid(
                        "cluster partition must contain each customer exactly once",
                    ));
                }
            }
            if seen.len() != customers.len() {
                return Err(invalid("cluster partition omits customers"));
            }
            let mut groups = groups.clone();
            for group in &mut groups {
                group.sort_unstable();
            }
            groups.sort();
            return Ok(groups);
        }
        Ok(mst_partition(graph, &customers, self.theta))
    }

    pub(super) fn prepare<'a, const D: usize>(
        &mut self,
        mut graphs: impl Iterator<Item = (i32, &'a Graph<D>)>,
    ) -> Result<()> {
        if !self.priority.is_finite() {
            return Err(invalid("cluster branching requires finite priority"));
        }
        let graph = graphs
            .find(|(id, _)| *id == self.network)
            .map(|(_, g)| g)
            .ok_or_else(|| invalid("unknown cluster branching network"))?;
        let mut all = vec![vec![graph.source, graph.sink]];
        all.extend(self.groups(graph)?);
        let inside = |group: &[usize], v: usize| group.contains(&v);
        let mut prepared = vec![];
        let network = self.network;
        let mut push = |name: String, coefficient: &dyn Fn(usize, usize) -> bool| {
            let arcs: Vec<_> = graph
                .arcs
                .iter()
                .filter(|a| coefficient(a.tail, a.head))
                .map(|a| (a.id, 1.))
                .collect();
            if !arcs.is_empty() {
                prepared.push((name, arcs));
            }
        };
        for (i, group) in all.iter().enumerate() {
            push(format!("cluster:{network}:{i}"), &|tail, head| {
                !inside(group, tail) && inside(group, head)
            });
        }
        for (i, first) in all.iter().enumerate() {
            for (j, second) in all.iter().enumerate().skip(i + 1) {
                push(format!("cluster:{network}:{i}:{j}"), &|tail, head| {
                    inside(first, tail) && inside(second, head)
                        || inside(second, tail) && inside(first, head)
                });
            }
        }
        self.prepared = prepared;
        Ok(())
    }

    pub(super) fn proposals<const D: usize>(
        &self,
        output: &Output<D>,
        tolerance: f64,
    ) -> Result<Vec<Proposal>> {
        let node = &output.snapshot;
        let Some(oracle) = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == self.network)
        else {
            return Err(invalid("unknown cluster branching network"));
        };
        let mut activity = vec![0.; oracle.network.graph.arcs.len()];
        for (route, quantity) in output
            .routes
            .iter()
            .zip(&output.values)
            .map(|(r, &q)| (r, q))
            .chain(
                node.state
                    .fixed_routes
                    .iter()
                    .map(|f| (&f.route, f.quantity)),
            )
        {
            if route.network != self.network || quantity == 0. {
                continue;
            }
            for id in &route.route.arcs {
                activity[oracle.index[id]] += quantity;
            }
        }
        let mut proposals = vec![];
        for (name, arcs) in &self.prepared {
            let mut value = 0.;
            for (id, coefficient) in arcs {
                let &position = oracle
                    .index
                    .get(id)
                    .ok_or_else(|| invalid("unknown cluster branching arc"))?;
                value += coefficient * activity[position];
            }
            if !(0. ..9_007_199_254_740_992.).contains(&value)
                || (value - value.round()).abs() <= tolerance
            {
                continue;
            }
            proposals.push(Proposal {
                history_key: name.clone(),
                name: name.clone(),
                priority: self.priority,
                value,
                restriction: Restriction::Activity {
                    network: self.network,
                    arcs: arcs.clone(),
                },
                history_eligible: true,
                fresh: true,
            });
        }
        Ok(proposals)
    }
}

/// Deterministic MST clustering. Distances use the cheaper finite arc cost in
/// either direction; missing edges are allowed.
fn mst_partition<const D: usize>(
    graph: &Graph<D>,
    customers: &[usize],
    theta: f64,
) -> Vec<Vec<usize>> {
    let n = customers.len();
    if n < 2 {
        return customers.iter().map(|&v| vec![v]).collect();
    }
    let positions: BTreeMap<_, _> = customers.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut weights = vec![f64::INFINITY; n * n];
    for arc in &graph.arcs {
        if let (Some(&i), Some(&j)) = (positions.get(&arc.tail), positions.get(&arc.head)) {
            if i != j && arc.cost.is_finite() {
                let weight = weights[i * n + j].min(arc.cost);
                weights[i * n + j] = weight;
                weights[j * n + i] = weight;
            }
        }
    }
    let mut reached = vec![false; n];
    let mut best = vec![(f64::INFINITY, 0usize); n];
    best[0].0 = 0.;
    let mut edges = vec![];
    for step in 0..n {
        let next = (0..n)
            .filter(|&v| !reached[v])
            .min_by(|&a, &b| best[a].0.total_cmp(&best[b].0).then(a.cmp(&b)))
            .unwrap();
        if !best[next].0.is_finite() {
            return vec![];
        }
        if step > 0 {
            edges.push((best[next].1, next, best[next].0));
        }
        reached[next] = true;
        for v in 0..n {
            if !reached[v] && weights[next * n + v] < best[v].0 {
                best[v] = (weights[next * n + v], next);
            }
        }
    }
    // Scale before moments to avoid overflowing on large finite distances.
    let scale = edges.iter().map(|e| e.2.abs()).fold(1., f64::max);
    let mean = edges.iter().map(|e| e.2 / scale).sum::<f64>() / edges.len() as f64;
    let variance = edges
        .iter()
        .map(|e| (e.2 / scale - mean).powi(2))
        .sum::<f64>()
        / edges.len() as f64;
    let threshold = mean + theta * variance.sqrt();
    let mut labels: Vec<_> = (0..n).collect();
    for &(a, b, weight) in &edges {
        if weight / scale > threshold {
            continue;
        }
        let old = labels[b];
        let new = labels[a];
        for label in &mut labels {
            if *label == old {
                *label = new;
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (i, &v) in customers.iter().enumerate() {
        groups.entry(labels[i]).or_default().push(v);
    }
    let mut groups: Vec<_> = groups.into_values().collect();
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort();
    groups
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/cluster_branching_tests.rs"]
mod tests;
