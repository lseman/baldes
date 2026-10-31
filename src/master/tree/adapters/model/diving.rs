//! Diving primal heuristic over network columns, following the native
//! `DivingHeuristic`/`DiveAlgorithm` control (disabled by default there too).
//!
//! At a selected tree node, a dive repeatedly fixes columns and re-solves the
//! master by column generation. Each dive node first fixes every column whose
//! value is already a positive integer, then rounds one fractional column up
//! to its ceiling (largest value first, ties in column order). With more than
//! one strong candidate, up to that many rounded children are evaluated and
//! sorted by LP value; within the first `lds_depth` dive levels the first
//! `max_discrepancy + 1 - discrepancy` of them are kept (limited-discrepancy
//! search), otherwise only the cheapest. A discrepancy child inherits the
//! columns rounded by its better siblings as tabu, so its subtree does not
//! repeat their roundings. Dive nodes are taken deepest first.
//! A fixed column is a fixed route copy, so pricing remains valid. Dives only
//! supply incumbents; their bounds restrict a heuristic subspace and are never
//! published as tree bounds.
use super::*;

/// Native parameters: `DivingHeurUseDepthLimit`, `CallFrequencyOfDivingHeur`,
/// `MaxLDSdepth`, `MaxLDSbreadth`, `StrongDivingCandidatesNumber`,
/// `DivingHeurStopsWithFirstFeasSol`, `FixIntValBeforeRoundingHeur` and
/// `MaxNbOfCgIteDuringRh`, with their native defaults. `node_limit` and
/// `time_limit` bound the work of one dive.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DivingOptions {
    /// Dive at tree nodes whose depth is at most this value.
    pub depth_limit: usize,
    /// Dive at every `frequency`-th selected node; zero dives at the root only.
    pub frequency: usize,
    /// Dive levels that keep several children (limited-discrepancy search).
    pub lds_depth: usize,
    /// Additional discrepancies allowed along one dive.
    pub max_discrepancy: usize,
    /// Rounded children evaluated per dive node; one is a plain greedy dive.
    pub strong_candidates: usize,
    pub stop_after_first_solution: bool,
    /// Fix columns with positive integer values before rounding.
    pub fix_integer_columns: bool,
    /// Column-generation iterations per dive node.
    pub max_iterations: usize,
    pub node_limit: usize,
    /// Seconds per dive.
    pub time_limit: f64,
}

impl Default for DivingOptions {
    fn default() -> Self {
        Self {
            depth_limit: 0,
            frequency: 0,
            lds_depth: 0,
            max_discrepancy: 0,
            strong_candidates: 1,
            stop_after_first_solution: false,
            fix_integer_columns: true,
            max_iterations: 5000,
            node_limit: 1000,
            time_limit: 10.,
        }
    }
}

impl DivingOptions {
    pub(super) fn validate(&self) -> Result<()> {
        if self.strong_candidates == 0
            || self.max_iterations == 0
            || !self.time_limit.is_finite()
            || self.time_limit < 0.
        {
            return Err(invalid(
                "diving requires positive candidates and iterations and a finite time limit",
            ));
        }
        Ok(())
    }

    /// Whether the tree node at `depth`, selected as the `selected`-th node
    /// (one-based), runs a dive.
    pub(super) fn due(&self, depth: usize, selected: usize) -> bool {
        depth <= self.depth_limit
            && if self.frequency == 0 {
                depth == 0
            } else {
                selected.saturating_sub(1) % self.frequency == 0
            }
    }
}

#[derive(Default)]
pub(super) struct Dive {
    pub incumbent: Option<IntegerWitness>,
    pub nodes: usize,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub oracle_calls: Vec<(i32, usize)>,
}

impl Dive {
    fn count<const D: usize>(&mut self, output: &Output<D>) {
        self.nodes += 1;
        self.iterations += output.master.iterations;
        self.pricing_calls += output.master.pricing_calls;
        for &(id, calls) in &output.oracle_calls {
            match self.oracle_calls.iter_mut().find(|(n, _)| *n == id) {
                Some((_, total)) => *total += calls,
                None => self.oracle_calls.push((id, calls)),
            }
        }
    }
}

enum Pending<const D: usize> {
    Solved(Output<D>),
    Fixed(NodeSnapshot<D>),
}

struct Entry<const D: usize> {
    pending: Pending<D>,
    depth: usize,
    discrepancy: usize,
    order: usize,
    /// Columns this dive node may not round (native `tabuVariables`).
    tabu: Vec<usize>,
}

/// Positive route columns of a solved node with their master values.
fn route_columns<const D: usize>(output: &Output<D>, tolerance: f64) -> Vec<(usize, f64)> {
    let node = &output.snapshot;
    if !node.master.solved() {
        return vec![];
    }
    let values = node.master.values();
    node.warm
        .columns
        .iter()
        .enumerate()
        .filter_map(|(column, metadata)| {
            let tag = metadata.as_ref()?.tag;
            if tag < 0 || master_variables::decode(tag).is_some() {
                return None;
            }
            let value = values[column];
            (value.is_finite() && value > tolerance).then_some((tag as usize, value))
        })
        .collect()
}

fn is_integer(value: f64) -> bool {
    (value - value.round()).abs() <= 1e-6
}

pub(super) fn run<const D: usize>(
    root: &Output<D>,
    options: &ModelTreeOptions,
    config: &DivingOptions,
    incumbent: f64,
    deadline: Instant,
) -> Result<Dive> {
    let end = deadline.min(
        Instant::now()
            .checked_add(Duration::from_secs_f64(config.time_limit))
            .unwrap_or(deadline),
    );
    let tolerance = options.tolerance;
    let mut dive = Dive::default();
    let mut best = incumbent;
    let mut order = 0;
    let mut heap: Vec<Entry<D>> = vec![];
    let mut current: Option<Entry<D>> = None;
    loop {
        let solved;
        let (output, depth, discrepancy, tabu) = match current.take() {
            None if dive.nodes == 0 && order == 0 => {
                order += 1;
                (root, 0, 0, vec![])
            }
            None => break,
            Some(entry) => {
                let output = match entry.pending {
                    Pending::Solved(output) => output,
                    Pending::Fixed(node) => {
                        if Instant::now() >= end || dive.nodes >= config.node_limit {
                            break;
                        }
                        match solve_child(&node, options, config, best, end) {
                            Ok(output) => {
                                dive.count(&output);
                                output
                            }
                            Err(error) if budget_error(&error) => {
                                current = pop_deepest(&mut heap);
                                continue;
                            }
                            Err(error) => return Err(error),
                        }
                    }
                };
                solved = output;
                (&solved, entry.depth, entry.discrepancy, entry.tabu)
            }
        };
        if let Some(witness) = output.snapshot.integer_witness(tolerance)? {
            if witness.0 < best {
                best = witness.0;
                dive.incumbent = Some(witness);
                if config.stop_after_first_solution {
                    break;
                }
            }
        }
        let children = children(
            output,
            options,
            config,
            best,
            depth,
            discrepancy,
            &tabu,
            end,
            &mut dive,
        )?;
        for (pending, discrepancy, tabu) in children {
            heap.push(Entry {
                pending,
                depth: depth + 1,
                discrepancy,
                order,
                tabu,
            });
            order += 1;
        }
        if Instant::now() >= end || dive.nodes >= config.node_limit {
            break;
        }
        current = pop_deepest(&mut heap);
    }
    Ok(dive)
}

/// Deepest first; among equal depths, the earliest created (cheapest strong
/// child first, since children are created in LP order).
fn pop_deepest<const D: usize>(heap: &mut Vec<Entry<D>>) -> Option<Entry<D>> {
    let index = (0..heap.len()).max_by(|&a, &b| {
        heap[a]
            .depth
            .cmp(&heap[b].depth)
            .then(heap[b].order.cmp(&heap[a].order))
    })?;
    Some(heap.swap_remove(index))
}

fn budget_error(error: &crate::core::error::OrError) -> bool {
    error.message.contains("time limit") || error.message.contains("label limit")
}

fn solve_child<const D: usize>(
    node: &NodeSnapshot<D>,
    options: &ModelTreeOptions,
    config: &DivingOptions,
    best: f64,
    end: Instant,
) -> Result<Output<D>> {
    let mut configured = options.node.clone();
    configured.fixing_upper_bound = best.is_finite().then_some(best);
    configured.cg.max_iterations = configured.cg.max_iterations.min(config.max_iterations);
    configured.cg.time_limit = configured
        .cg
        .time_limit
        .min(end.saturating_duration_since(Instant::now()));
    node.solve(configured)
}

#[allow(clippy::too_many_arguments)]
fn children<const D: usize>(
    output: &Output<D>,
    options: &ModelTreeOptions,
    config: &DivingOptions,
    best: f64,
    depth: usize,
    discrepancy: usize,
    tabu: &[usize],
    end: Instant,
    dive: &mut Dive,
) -> Result<Vec<(Pending<D>, usize, Vec<usize>)>> {
    let tolerance = options.tolerance;
    // Native prunes a dive node whose LP does not improve the primal bound.
    if output.master.status != cg::Status::Optimal
        || model_strong::proven_infeasible(output)
        || output.master.lp >= best - (1e-6 + 1e-10 * output.master.lp.abs().max(best.abs()))
    {
        return Ok(vec![]);
    }
    let columns = route_columns(output, tolerance);
    let mut base: Option<NodeSnapshot<D>> = None;
    if config.fix_integer_columns {
        for &(tag, value) in &columns {
            if is_integer(value) && value.round() >= 1. {
                let parent = base.as_ref().unwrap_or(&output.snapshot);
                if let Ok(fixed) = parent.fix_route(tag, value.round()) {
                    base = Some(fixed);
                }
            }
        }
    }
    let mut candidates: Vec<_> = columns
        .into_iter()
        .filter(|&(tag, value)| !is_integer(value) && !tabu.contains(&tag))
        .collect();
    // Round the column closest to its ceiling first.
    candidates.sort_by(|a, b| {
        (a.1.ceil() - a.1)
            .total_cmp(&(b.1.ceil() - b.1))
            .then(a.0.cmp(&b.0))
    });
    let parent = base.as_ref().unwrap_or(&output.snapshot);
    if config.strong_candidates <= 1 {
        for (tag, value) in candidates {
            if let Ok(child) = parent.fix_route(tag, value.ceil()) {
                return Ok(vec![(Pending::Fixed(child), discrepancy, tabu.to_vec())]);
            }
        }
        // Only integer columns remained fractional-free: dive on the fixings.
        return Ok(base
            .map(|node| vec![(Pending::Fixed(node), discrepancy, tabu.to_vec())])
            .unwrap_or_default());
    }
    let mut evaluated = vec![];
    for (tag, value) in candidates {
        if evaluated.len() >= config.strong_candidates
            || Instant::now() >= end
            || dive.nodes >= config.node_limit
        {
            break;
        }
        let Ok(child) = parent.fix_route(tag, value.ceil()) else {
            continue;
        };
        match solve_child(&child, options, config, best, end) {
            Ok(solved) => {
                dive.count(&solved);
                if solved.master.status == cg::Status::Optimal
                    && !model_strong::proven_infeasible(&solved)
                {
                    evaluated.push((tag, solved));
                }
            }
            Err(error) if budget_error(&error) => {}
            Err(error) => return Err(error),
        }
    }
    evaluated.sort_by(|a, b| a.1.master.lp.total_cmp(&b.1.master.lp));
    let allowed = if depth < config.lds_depth {
        (config.max_discrepancy + 1)
            .saturating_sub(discrepancy)
            .max(1)
    } else {
        1
    };
    // Limited-discrepancy siblings: a child that deviates from the better
    // roundings may not round their columns again in its subtree.
    let kept: Vec<_> = evaluated.into_iter().take(allowed).collect();
    let tags: Vec<usize> = kept.iter().map(|(tag, _)| *tag).collect();
    Ok(kept
        .into_iter()
        .enumerate()
        .map(|(rank, (_, solved))| {
            let mut inherited = tabu.to_vec();
            inherited.extend_from_slice(&tags[..rank]);
            (Pending::Solved(solved), discrepancy + rank, inherited)
        })
        .collect())
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/diving_tests.rs"]
mod tests;
