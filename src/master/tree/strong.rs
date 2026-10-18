//! Phased strong-branching decision rules for arc-flow candidates.
use super::*;
use crate::master::{clp::ClpMaster, LpMaster};

#[derive(Clone, Debug)]
pub struct StrongPhase {
    pub candidates: usize,
    pub pricing_phase_floor: usize,
    pub max_iterations: usize,
    pub tree_depth_score: bool,
    pub restricted_master_only: bool,
}
#[derive(Clone, Debug)]
pub struct StrongBranchingOptions {
    pub phases: Vec<StrongPhase>,
    pub time_limit: Duration,
    pub trial_time_limit: Duration,
}
impl Default for StrongBranchingOptions {
    fn default() -> Self {
        Self {
            phases: vec![
                StrongPhase {
                    candidates: 100,
                    pricing_phase_floor: 0,
                    max_iterations: 1,
                    tree_depth_score: false,
                    restricted_master_only: true,
                },
                StrongPhase {
                    candidates: 3,
                    pricing_phase_floor: 1,
                    max_iterations: 10000,
                    tree_depth_score: false,
                    restricted_master_only: false,
                },
                StrongPhase {
                    candidates: 1,
                    pricing_phase_floor: 0,
                    max_iterations: 10000,
                    tree_depth_score: true,
                    restricted_master_only: false,
                },
            ],
            time_limit: Duration::from_secs(10),
            trial_time_limit: Duration::from_secs(1),
        }
    }
}
/// Binary specialization of CandidateBranchGroup::computeBranchingScoreAsProduct.
/// If just one child exceeds the incumbent gap, compress its excess by 1e-6;
/// if both do, preserve both gains. Always floor a gain at 1e-6.
pub fn product_score(parent: f64, incumbent: f64, children: [f64; 2]) -> f64 {
    let gap = incumbent - parent;
    let mut gains = children.map(|b| b - parent);
    let all_closed = gains.iter().all(|&g| g >= gap);
    for gain in &mut gains {
        if *gain > gap && !all_closed {
            *gain = (*gain - gap) * 1e-6 + gap;
        }
        *gain = gain.max(1e-6);
    }
    gains[0] * gains[1]
}
fn number_of_leaves(gap: f64, gains: &[f64]) -> f64 {
    let mut low = 0.;
    let mut high = 1e20;
    let mut middle: f64 = 5e19;
    for iteration in 0..100 {
        let sum = gains
            .iter()
            .map(|&gain| middle.powf(-gain / gap))
            .sum::<f64>();
        if sum < 1. {
            high = middle;
        } else {
            low = middle;
        }
        if middle > 9.99e19 {
            return -1.;
        }
        if iteration == 99 {
            return middle;
        }
        middle = (low + high) * 0.5;
        if high / 1e6 > high - low {
            return middle;
        }
    }
    middle
}
/// Native binary tree-depth estimate. A fixed normalization gap of one is
/// sufficient for ranking; callers can supply the reference's normalization.
pub fn tree_depth_score(
    parent: f64,
    incumbent: f64,
    children: [f64; 2],
    normalization: f64,
) -> f64 {
    let gap = incumbent - parent;
    let raw = children.map(|b| b - parent);
    let zero = raw.iter().filter(|&&g| g < 1e-6).count();
    if zero == 2 {
        return -1e12;
    }
    let mut gains = raw.map(|g| g.min(gap));
    if zero != 0 {
        for gain in &mut gains {
            *gain = gain.max(gap * 0.0001);
        }
    }
    let leaves = number_of_leaves(normalization, &gains);
    if leaves >= 0. {
        -leaves.ln() / 2f64.ln()
    } else {
        -1e12
    }
}
struct Candidate {
    branch: super::candidates::BranchCandidate,
    order: usize,
    score: f64,
    bounds: [f64; 2],
}
fn restricted_value(
    parent: Option<&Rc<root::NodeSnapshot>>,
    problem: &RootProblem,
    branches: &[Branch],
    pool: &Pool,
    options: &TreeOptions,
) -> Result<f64> {
    let penalty = options.root.cg.artificial_cost;
    if let Some(parent) = parent {
        let depth = parent.branch_depth();
        if depth <= branches.len() {
            return Ok(parent.restricted_value(&branches[depth..], penalty)?.0);
        }
    }
    let child = branch_problem(problem, branches)?;
    let mut master = ClpMaster::new()?;
    master.set_tolerances(1e-7, 1e-7)?;
    for row in &child.rows {
        master.add_row(row.sense, row.rhs, &[])?;
    }
    for route in &pool.routes {
        let mut entries = BTreeMap::new();
        for &id in &route.arcs {
            for &(r, a) in &child.arc_entries[child.arc_position(id).unwrap()] {
                *entries.entry(r).or_insert(0.) += a;
            }
        }
        for &(r, a) in &child.route_entries {
            *entries.entry(r).or_insert(0.) += a;
        }
        for (offset,branch) in branches.iter().enumerate() {
            if matches!(branch,BranchConstraint::RyanFoster(_)) {
                let value=branch.coefficient(problem,route);
                if value!=0. {entries.insert(problem.rows.len()+offset,value);}
            }
        }
        master.add_column(
            route.original_cost,
            &entries.into_iter().collect::<Vec<_>>(),
        )?;
    }
    for (r, row) in child.rows.iter().enumerate() {
        if row.sense != RowSense::LessEqual {
            master.add_column(penalty, &[(r, 1.)])?;
        }
        if row.sense != RowSense::GreaterEqual {
            master.add_column(penalty, &[(r, -1.)])?;
        }
    }
    Ok(if master.solve() {
        master.objective()
    } else {
        1e12
    })
}
/// Return a chosen candidate and its certified child bounds. Restricted LP
/// objectives and heuristic pricing values affect scores only.
pub(super) fn choose(
    parent: Option<&Rc<root::NodeSnapshot>>,
    problem: &RootProblem,
    node: &Node,
    supplied: Vec<super::candidates::BranchCandidate>,
    pool: &mut Pool,
    options: &TreeOptions,
    output: &mut TreeOutput,
    deadline: Instant,
) -> Result<Option<(super::candidates::BranchCandidate, [f64; 2])>> {
    let mut candidates: Vec<_> = supplied.into_iter().enumerate().map(|(order,branch)|
        Candidate {branch,order,score:0.,bounds:[node.bound;2]}).collect();
    let Some(config) = &options.strong_branching else {
        return Ok(candidates.first().map(|c| (c.branch.clone(), c.bounds)));
    };
    let end = deadline.min(Instant::now() + config.time_limit);
    for phase in &config.phases {
        candidates.truncate(phase.candidates);
        if candidates.is_empty() || Instant::now() >= end {
            break;
        }
        for candidate in &mut candidates {
            let mut estimates = [node.bound; 2];
            for (side,constraint) in [candidate.branch.down.clone(),candidate.branch.up.clone()].into_iter().enumerate() {
                if Instant::now() >= end {break;}
                let mut branches = node.branches.clone();
                branches.push(constraint);
                output.strong_branching_trials += 1;
                if phase.restricted_master_only {
                    estimates[side] = restricted_value(parent, problem, &branches, pool, options)?;
                } else {
                    let result = evaluate(
                        parent,
                        problem,
                        &branches,
                        pool,
                        options,
                        output.incumbent,
                        end.min(Instant::now() + config.trial_time_limit),
                        phase.max_iterations,
                        phase.pricing_phase_floor,
                        false,
                    );
                    match result {
                        Ok(result) => {
                            output.pricing_calls += result.master.pricing_calls;
                            output.fixing_calls += result.fixing_calls;
                            output.enumerated_routes += result.enumerated_routes;
                            estimates[side] = result.master.lp.min(1e12);
                            if result.preprocessing.infeasible {
                                candidate.bounds[side] = f64::INFINITY;
                            } else if result.master.bound.is_finite() {
                                candidate.bounds[side] =
                                    candidate.bounds[side].max(result.master.bound);
                            }
                            for route in &result.routes {
                                pool.add(problem, &route.arcs);
                            }
                            output.mip_nodes += integer_heuristic(
                                problem,
                                pool,
                                &mut output.incumbent,
                                &mut output.routes,
                                options,
                                end,
                            );
                        }
                        Err(e) if budget_error(&e) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            candidate.score = if phase.tree_depth_score {
                tree_depth_score(node.bound, output.incumbent.min(1e12), estimates, 1.)
            } else {
                product_score(node.bound, output.incumbent.min(1e12), estimates)
            };
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.order.cmp(&b.order)));
    }
    Ok(candidates.first().map(|c| (c.branch.clone(), c.bounds)))
}
