//! Cluster variables from Silva, Uchoa and Subramanian (2024), Sections 3–4.
use super::*;
use crate::master::branching::ClusterBranchingOptions;

/// Deterministic MST clustering of customers, excluding the split depot.
/// Distances use the cheaper physical arc in either direction. Missing edges
/// are allowed, but a disconnected customer graph has no automatic partition.
pub fn partition<const D: usize>(
    problem: &RoutingContext<D>,
    options: &ClusterBranchingOptions,
) -> Result<Vec<Vec<usize>>> {
    if !options.theta.is_finite() || options.theta < 0. {
        return Err(invalid("cluster MST theta must be finite and nonnegative"));
    }
    let customers = &problem.customers;
    if let Some(groups) = &options.partition {
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
    let n = customers.len();
    if n < 2 {
        return Ok(customers.iter().map(|&v| vec![v]).collect());
    }
    let positions: BTreeMap<_, _> = customers.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut weights = vec![f64::INFINITY; n * n];
    for arc in &problem.model.networks[0].graph.arcs {
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
            return Ok(vec![]);
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
    let threshold = mean + options.theta * variance.sqrt();
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
    Ok(groups)
}

pub(super) fn candidates<const D: usize>(
    problem: &RoutingContext<D>,
    result: &routing::RootOutput<D>,
    groups: &[Vec<usize>],
) -> Vec<super::candidates::BranchCandidate> {
    use super::candidates::BranchCandidate;
    let routes = result.route_values();
    let mut candidates = vec![];
    let mut add = |make: &dyn Fn(RowSense, f64) -> BranchConstraint| {
        let template = make(RowSense::LessEqual, 0.);
        let value = routes
            .iter()
            .map(|(r, x)| template.coefficient(problem, r) * x)
            .sum::<f64>();
        if (value - value.round()).abs() > 1e-6 {
            candidates.push(BranchCandidate {
                down: make(RowSense::LessEqual, value.floor()),
                up: make(RowSense::GreaterEqual, value.ceil()),
                value,
            });
        }
    };
    let mut all = vec![vec![
        problem.model.networks[0].graph.source,
        problem.model.networks[0].graph.sink,
    ]];
    all.extend_from_slice(groups);
    for group in &all {
        add(&|sense, rhs| BranchConstraint::ClusterBoundary {
            vertices: group.clone(),
            sense,
            rhs,
        });
    }
    for (i, first) in all.iter().enumerate() {
        for second in &all[i + 1..] {
            add(&|sense, rhs| BranchConstraint::ClusterPair {
                first: first.clone(),
                second: second.clone(),
                sense,
                rhs,
            });
        }
    }
    candidates
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/clusters/tests.rs"]
mod tests;
