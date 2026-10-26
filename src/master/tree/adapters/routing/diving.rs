//! Strong round-up diving over elementary route columns, with a depth-first
//! agenda, discrepancy budget, and first-solution stopping. A route is fixed
//! through all its arc flows, so pricing remains valid without route-specific
//! dual resources.
use super::*;
#[derive(Clone, Debug)]
pub struct DivingOptions {
    pub frequency: usize,
    pub max_depth: usize,
    pub max_discrepancy: usize,
    pub strong_candidates: usize,
    pub node_limit: usize,
    pub time_limit: Duration,
    pub max_iterations: usize,
    pub stop_after_first_solution: bool,
}
impl Default for DivingOptions {
    fn default() -> Self {
        Self {
            frequency: 10,
            max_depth: 30,
            max_discrepancy: 2,
            strong_candidates: 3,
            node_limit: 100,
            time_limit: Duration::from_secs(2),
            max_iterations: 10000,
            stop_after_first_solution: true,
        }
    }
}
/// Validate an integral LP witness independently of artificial columns,
/// solver incumbent summaries, or the branch being explored.
pub(super) fn record_solution<const D: usize>(
    problem: &RoutingContext<D>,
    result: &routing::RootOutput<D>,
    output: &mut TreeOutput,
) -> bool {
    if result
        .master
        .artificials
        .iter()
        .any(|&i| result.values.get(i).copied().unwrap_or(0.) > 1e-6)
    {
        return false;
    }
    let mut cover = vec![0; problem.model.networks[0].graph.vertices.len()];
    let mut routes = vec![];
    for (&tag, &value) in result.master.column_tags.iter().zip(&result.values) {
        if tag < 0 || value <= 1e-6 {
            continue;
        }
        if (value - 1.).abs() > 1e-6 {
            return false;
        }
        let Some(route) = feasible_route(problem, &result.routes[tag as usize].arcs) else {
            return false;
        };
        for &v in &route.nodes[1..route.nodes.len() - 1] {
            cover[v as usize] += 1;
        }
        routes.push(route);
    }
    if !problem.customers.iter().all(|&v| cover[v] == 1)
        || routes.len() as f64 > problem.model.rows[0].rhs
    {
        return false;
    }
    let Some(cost) = problem.unit_route_incumbent(&routes) else {
        return false;
    };
    if cost < output.incumbent {
        output.incumbent = cost;
        output.routes = routes;
    }
    true
}
struct Dive<const D: usize> {
    branches: Vec<Branch>,
    depth: usize,
    discrepancy: usize,
    result: routing::RootOutput<D>,
}
pub(super) fn run<const D: usize>(
    problem: &RoutingContext<D>,
    node: &Node<D>,
    parent: routing::RootOutput<D>,
    pool: &mut Pool,
    options: &TreeOptions,
    output: &mut TreeOutput,
    deadline: Instant,
) -> Result<()> {
    let Some(config) = &options.diving else {
        return Ok(());
    };
    if config.frequency == 0 || (output.nodes - 1) % config.frequency != 0 {
        return Ok(());
    }
    let end = deadline.min(Instant::now() + config.time_limit);
    let start_count = output.diving_nodes;
    let mut agenda = vec![Dive {
        branches: node.branches.clone(),
        depth: 0,
        discrepancy: 0,
        result: parent,
    }];
    while let Some(dive) = agenda.pop() {
        if dive.depth >= config.max_depth
            || Instant::now() >= end
            || output.diving_nodes - start_count >= config.node_limit
        {
            break;
        }
        let mut candidates: Vec<_> = dive
            .result
            .master
            .column_tags
            .iter()
            .zip(&dive.result.values)
            .filter_map(|(&tag, &x)| {
                if tag < 0 || x <= 1e-6 || x >= 1. - 1e-6 {
                    return None;
                }
                feasible_route(problem, &dive.result.routes[tag as usize].arcs).map(|r| (x, tag, r))
            })
            .collect();
        // Least-fractional upward rounding (SelectionStrategy case 4,
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        candidates.truncate(config.strong_candidates);
        let mut children = vec![];
        for (_, _, route) in candidates {
            if Instant::now() >= end || output.diving_nodes - start_count >= config.node_limit {
                break;
            }
            let mut branches = dive.branches.clone();
            let mut added = 0;
            for &id in &route.arcs {
                let arc = problem.arc_position(id).unwrap();
                if branches
                    .iter()
                    .any(|b| matches!(b,BranchConstraint::ArcFlow {arc:a,sense:RowSense::GreaterEqual,rhs} if *a==arc && *rhs>=1.))
                {
                    continue;
                }
                branches.push(BranchConstraint::ArcFlow {
                    arc,
                    sense: RowSense::GreaterEqual,
                    rhs: 1.,
                });
                added += 1;
            }
            if added == 0 {
                continue;
            }
            output.diving_nodes += 1;
            match evaluate(
                dive.result.snapshot.as_ref(),
                None,
                problem,
                &branches,
                pool,
                options,
                output.incumbent,
                end,
                config.max_iterations,
                0,
                dive.depth + 1 < config.max_depth,
            ) {
                Ok(result) => {
                    output.pricing_calls += result.master.pricing_calls;
                    output.fixing_calls += result.fixing_calls;
                    output.enumerated_routes += result.enumerated_routes;
                    for route in &result.routes {
                        pool.add(problem, &route.arcs);
                    }
                    let feasible = record_solution(problem, &result, output);
                    output.mip_nodes += integer_heuristic(
                        problem,
                        pool,
                        &mut output.incumbent,
                        &mut output.routes,
                        options,
                        end,
                    );
                    if feasible && config.stop_after_first_solution {
                        return Ok(());
                    }
                    if result.master.status == Status::Optimal
                        && !result.master.pruned_by_bound
                        && result
                            .master
                            .artificials
                            .iter()
                            .all(|&i| result.values.get(i).copied().unwrap_or(0.) <= 1e-6)
                    {
                        children.push((result.master.lp, branches, result));
                    }
                }
                Err(e) if budget_error(&e) => {}
                Err(e) => return Err(e),
            }
        }
        // Strong diving prefers the cheapest priced restricted child, then
        // explores alternatives by depth with a bounded number of discrepancies.
        children.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (rank, (_, branches, result)) in children.into_iter().enumerate().rev() {
            let discrepancy = dive.discrepancy + usize::from(rank != 0);
            if discrepancy <= config.max_discrepancy {
                agenda.push(Dive {
                    branches,
                    depth: dive.depth + 1,
                    discrepancy,
                    result,
                });
            }
        }
    }
    // Dive bounds restrict a heuristic subspace. Never publish them as a
    // bound on the original node or consume global branching nodes.
    Ok(())
}
