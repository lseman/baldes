//! Column generation with cut rounds and fixing/enumeration master
//! transitions, from the C++ reference `master/ColumnGeneration.h`.
//!
//! Generic over any `LpMaster` backend (Clp, HiGHS).  The controller trait
//! receives a mutable reference to the generic master so it can inspect
//! primal/dual values and add columns.
pub use super::lp_master::LpMaster;
use super::{
    stabilization::{DualPriceSmoothing, Feedback, Parameters},
    RowSense,
};
use crate::pricing::phases::{self, Phase};
use crate::{invalid, Result};
use std::collections::HashSet;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Column {
    pub cost: f64,
    pub reduced_cost: f64,
    pub entries: Vec<(usize, f64)>,
    pub acceptable: bool,
    pub tag: i32,
    pub identity: String,
}

impl Column {
    fn signature(&self) -> String {
        if !self.identity.is_empty() {
            return format!("identity:{}", self.identity);
        }
        let mut s = format!("{:.11e}", self.cost);
        for &(r, v) in &self.entries {
            if v != 0. {
                s.push_str(&format!("|{r}:{v:.11e}"));
            }
        }
        s
    }
}

#[derive(Clone, Debug)]
pub struct Answer {
    pub columns: Vec<Column>,
    pub minimum: f64,
    pub exact: bool,
    pub bound_minimum: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    Stalled,
}

#[derive(Clone, Debug)]
pub struct Iteration {
    pub iteration: usize,
    pub lp: f64,
    pub minimum: f64,
    pub new_columns: usize,
    pub columns: usize,
    pub exact: bool,
    pub alpha: f64,
    pub effective_alpha: f64,
    pub angle: f64,
    pub misprices: i32,
    pub phase: usize,
    pub pricing_calls: usize,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub smoothing: Parameters,
    pub stabilize: bool,
    pub tolerance: f64,
    pub multiplicity: f64,
    pub max_iterations: usize,
    pub time_limit: Duration,
    pub known: Vec<Column>,
    pub artificial_cost: f64,
    pub penalty_factor: f64,
    pub max_penalty_updates: usize,
    pub insert_non_negative: bool,
    pub gap_absolute: f64,
    pub gap_relative: f64,
    pub enumerated_columns_added: usize,
    pub max_cut_rounds: usize,
    /// Stop after this many rounds whose gain
    /// is below the threshold times 0.1|LP|.
    pub tailing_off_threshold: f64,
    pub tailing_off_counter: usize,
    /// Remove cut rows whose slack is basic before separating again.
    pub remove_basic_cuts: bool,
    /// Cost of the local artificials of separated rows.
    pub cut_artificial_cost: f64,
    /// cleanupRestrictedMastColumns: with more generated columns than this
    /// (0 disables), keep `column_cleanup_ratio` of them before pricing.
    pub column_cleanup_threshold: f32,
    pub column_cleanup_ratio: f32,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            smoothing: Parameters::default(),
            stabilize: true,
            tolerance: 1e-6,
            multiplicity: 1.,
            max_iterations: 100000,
            time_limit: Duration::from_secs(3600000),
            known: vec![],
            artificial_cost: 0.,
            penalty_factor: 2.,
            max_penalty_updates: 5,
            insert_non_negative: true,
            gap_absolute: 1e-6,
            gap_relative: 1e-9,
            enumerated_columns_added: 0,
            max_cut_rounds: 50,
            tailing_off_threshold: 0.02,
            tailing_off_counter: 3,
            remove_basic_cuts: false,
            cut_artificial_cost: 0.,
            column_cleanup_threshold: 0.,
            column_cleanup_ratio: 0.66,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Output {
    pub status: Status,
    pub lp: f64,
    pub bound: f64,
    pub iterations: usize,
    pub pricing_calls: usize,
    pub columns: usize,
    pub history: Vec<Iteration>,
    pub column_tags: Vec<i32>,
    pub inserted_tags: Vec<i32>,
    pub artificials: Vec<usize>,
    pub penalty_updates: usize,
    pub pruned_by_bound: bool,
    pub cut_rounds: usize,
}

/// The controller owns incumbent tracking and the pricing route universe.
pub trait Controller<M> {
    fn master_solved(
        &mut self,
        _master: &M,
        _tags: &[i32],
        _artificials: &[usize],
    ) -> Result<()> {
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        f64::INFINITY
    }
    fn enumerated_universe(&mut self, _duals: &[f64], _bound: f64) -> Result<Option<Vec<Column>>> {
        Ok(None)
    }
    fn remap_columns(&mut self, _map: &[Option<usize>]) -> Result<()> {
        Ok(())
    }
    /// indices_remapped after a row (cut) or column deletion.
    fn remap_rows(&mut self, _rows: &[Option<usize>]) -> Result<()> {
        Ok(())
    }
    /// Whether a separator is installed.
    fn separates(&self) -> bool {
        false
    }
    /// Adds a cut family once pricing has converged on an exact phase;
    /// returns how many rows were added (or changed). Zero ends the rounds.
    fn separate(&mut self, _master: &mut M, _tags: &[i32]) -> Result<usize> {
        Ok(0)
    }
    /// Master columns to delete after `separate`, ascending: columns that a
    /// strengthened pricing relaxation no longer allows
    /// (removeColumnsNotSatisfyingColGenSpRelaxation).
    fn take_removed_columns(&mut self) -> Vec<usize> {
        vec![]
    }
}

pub struct NoController;
impl<M> Controller<M> for NoController {}

pub fn gap_smaller_than_tolerance(dual: f64, primal: f64, absolute: f64, relative: f64) -> bool {
    let largest = dual.abs().max(primal.abs());
    if !(dual < primal - (largest * 1e-10 + 1e-6)) {
        return true;
    }
    if (primal - dual).abs() <= absolute {
        return true;
    }
    if primal.abs() <= 1e-6 && dual.abs() <= 1e-6 {
        return relative >= 0.;
    }
    if dual * primal < -1e-6 {
        return relative >= 1.;
    }
    (primal - dual) / primal.abs() <= relative
}

fn lagrangian<M: LpMaster>(master: &M, duals: &[f64], minimum: f64, multiplicity: f64) -> f64 {
    let mut value = 0.;
    for (pi, rhs) in duals.iter().zip(master.rhs().iter()) {
        value += pi * rhs;
    }
    if minimum < 0. {
        value += multiplicity * minimum;
    }
    value
}

fn master_reduced_cost(column: &Column, duals: &[f64]) -> f64 {
    let mut value = column.cost;
    for &(r, a) in &column.entries {
        if let Some(pi) = duals.get(r) {
            value -= pi * a;
        }
    }
    value
}

/// Generic column generation loop that works with any `LpMaster` backend.
pub fn optimize<M: LpMaster, P: FnMut(usize, &[f64]) -> Result<Answer>, C: Controller<M>>(
    master: &mut M,
    supplied: &[Phase],
    mut price: P,
    controller: &mut C,
    options: Options,
) -> Result<Output> {
    let schedule = phases::initialize(0, supplied)?;
    if !options.multiplicity.is_finite() || options.multiplicity < 0. || options.max_iterations == 0
    {
        return Err(invalid("column generation options"));
    }
    let started = Instant::now();
    let mut smoothing = DualPriceSmoothing::new(options.smoothing.clone());
    let mut seen: HashSet<_> = options.known.iter().map(Column::signature).collect();
    let mut loaded: Vec<Option<Column>> = options.known.iter().cloned().map(Some).collect();
    loaded.resize(master.costs().len(), None);
    let mut out = Output {
        status: Status::IterationLimit,
        lp: 0.,
        bound: f64::NEG_INFINITY,
        iterations: 0,
        pricing_calls: 0,
        columns: master.costs().len(),
        history: vec![],
        column_tags: options.known.iter().map(|c| c.tag).collect(),
        inserted_tags: vec![],
        artificials: vec![],
        penalty_updates: 0,
        pruned_by_bound: false,
        cut_rounds: 0,
    };
    out.column_tags.resize(master.costs().len(), -1);
    if options.artificial_cost > 0. {
        for r in 0..master.rhs().len() {
            if master.senses()[r] != RowSense::LessEqual {
                out.artificials
                    .push(master.add_column(options.artificial_cost, &[(r, 1.)])?);
                loaded.push(None);
                out.column_tags.push(-1);
            }
            if master.senses()[r] != RowSense::GreaterEqual {
                out.artificials
                    .push(master.add_column(options.artificial_cost, &[(r, -1.)])?);
                loaded.push(None);
                out.column_tags.push(-1);
            }
        }
    }
    let mut stage = schedule.len() - 1;
    let mut pass_bound: f64;
    // BaPCod Alg4Master +0x28 (ELF 0x12b8e0): phase-local stopping
    // value, including heuristic passes; never expose it as a global bound.
    let mut stage_bound = -1e12_f64;
    let mut round_penalty_updates = 0;
    let mut tailing = 0;
    let mut lp_at_separation = 0.0f64;
    let mut separated_once = false;
    let initial_rows = master.rhs().len();
    let mut cut_artificials: Vec<(usize, usize)> = vec![];
    for iteration in 1..=options.max_iterations {
        if started.elapsed() > options.time_limit {
            out.status = Status::TimeLimit;
            return Ok(out);
        }
        if !master.solve() {
            out.status = if master.status() == 2 {
                Status::Unbounded
            } else {
                Status::Infeasible
            };
            out.lp = if out.status == Status::Infeasible {
                1e12
            } else {
                master.objective()
            };
            out.iterations = iteration;
            return Ok(out);
        }
        out.lp = master.objective();
        out.iterations = iteration;
        let raw = master.duals().to_vec();
        let positive = out.artificials.iter().any(|&i| master.values()[i] > 1e-6);
        controller.master_solved(master, &out.column_tags, &out.artificials)?;
        let gap = if out.bound.is_finite() { out.lp - out.bound } else { 0. };
        if options.stabilize {
            smoothing.after_master(&raw, master.senses(), gap, 0, iteration == 1);
        }
        // earlyCGtermType1 (ELF 0xdd560) uses the phase value for stage > 0
        // and the certified bound for stage zero, before another pricing call.
        let stopping_bound = if stage > 0 { stage_bound } else { out.bound };
        let converged = !positive
            && stopping_bound.is_finite()
            && gap_smaller_than_tolerance(
                stopping_bound,
                out.lp,
                options.gap_absolute,
                options.gap_relative,
            );
        let mut minimum = 0.;
        let mut fresh = 0;
        let mut exact = stage == 0;
        let mut answer_phase = stage;
        let mut inserted = 0;
        let mut first_angle = 0.;
        let mut advanced = false;
        let mut penalties = false;
        if converged && stage > 0 {
            stage -= 1;
            // eval (ELF 0x107740) clears both values on each phase change.
            stage_bound = -1e12;
            // Each phase is a new solveMastLpPrimPh1A2 (ELF 0x106ea0) call,
            // whose penalty-update count starts at zero.
            round_penalty_updates = 0;
            advanced = true;
            if options.stabilize {
                smoothing.after_iteration();
            }
        }
        if !converged {
            if options.column_cleanup_threshold > 0. {
                let removed = cleanup_columns(master, &out.column_tags, &options);
                if !removed.is_empty() {
                    for &i in &removed {
                        if let Some(column) = &loaded[i] {
                            seen.remove(&column.signature());
                        }
                    }
                    let (_, map) = master.remove_cuts(&[], &removed)?;
                    loaded = loaded.into_iter().zip(&map).filter_map(|(c, m)| m.map(|_| c)).collect();
                    out.column_tags =
                        out.column_tags.iter().zip(&map).filter_map(|(&t, m)| m.map(|_| t)).collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    cut_artificials = cut_artificials
                        .iter()
                        .filter_map(|&(r, c)| Some((r, map[c]?)))
                        .collect();
                    controller.remap_columns(&map)?;
                }
            }
            let mut current = if options.stabilize {
                smoothing.pricing.clone()
            } else {
                raw.clone()
            };
            let mut pending = vec![];
            loop {
                let mut answer = price(stage, &current)?;
                answer.exact |= schedule[stage].exact();
                out.pricing_calls += 1;
                fresh = answer
                    .columns
                    .iter()
                    .filter(|c| {
                        c.acceptable
                            && master_reduced_cost(c, &raw) < -options.tolerance
                            && !seen.contains(&c.signature())
                    })
                    .count();
                minimum = answer
                    .columns
                    .iter()
                    .filter(|c| c.acceptable)
                    .fold(answer.minimum, |best, c| best.min(c.reduced_cost));
                let corrected = answer.bound_minimum.unwrap_or(minimum);
                pass_bound = lagrangian(master, &current, corrected, options.multiplicity);
                let slack = stage_bound.abs().max(pass_bound.abs()) * 1e-10 + 1e-6;
                if !(stage_bound - slack > pass_bound) {
                    stage_bound = pass_bound;
                }
                if answer.exact && pass_bound > out.bound {
                    out.bound = pass_bound;
                }
                exact = answer.exact;
                answer_phase = stage;
                let mut gradient = master.rhs().to_vec();
                if let Some(best) = answer
                    .columns
                    .iter()
                    .filter(|c| c.acceptable)
                    .min_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost))
                {
                    if best.reduced_cost.abs() > 1e-6 * best.reduced_cost.abs() + 1e-6 {
                        for &(r, a) in &best.entries {
                            if let Some(g) = gradient.get_mut(r) {
                                *g -= options.multiplicity * a;
                            }
                        }
                    }
                }
                pending.push(answer.columns);
                if !options.stabilize {
                    break;
                }
                let first = smoothing.misprices == 0;
                let again = smoothing.after_pricing(&Feedback {
                    negative_columns: fresh,
                    subgradient: gradient,
                    lagrangian: pass_bound,
                    phase: stage as i32,
                });
                if first {
                    first_angle = smoothing.angle;
                }
                if !again {
                    break;
                }
                current = smoothing.pricing.clone();
            }
            if options.stabilize {
                smoothing.after_iteration();
            }
            let due = round_penalty_updates < options.max_penalty_updates && positive;
            penalties = fresh == 0
                && due
                && !(stage > 0
                    && controller.incumbent().is_finite()
                    && pass_bound.is_finite()
                    && gap_smaller_than_tolerance(
                        pass_bound,
                        controller.incumbent(),
                        options.gap_absolute,
                        options.gap_relative,
                    ));
            advanced = fresh == 0 && !exact && !penalties && stage > 0;
            if advanced {
                stage -= 1;
                stage_bound = -1e12;
                round_penalty_updates = 0;
            }
            let first = loaded.len();
            for batch in pending {
                let mut identities = vec![];
                for column in batch {
                    if !column.acceptable
                        || (!options.insert_non_negative
                            && master_reduced_cost(&column, &raw) >= -options.tolerance)
                    {
                        continue;
                    }
                    let identity = column.signature();
                    if seen.contains(&identity) {
                        continue;
                    }
                    out.column_tags.push(column.tag);
                    out.inserted_tags.push(column.tag);
                    loaded.push(Some(column));
                    identities.push(identity);
                    inserted += 1;
                }
                seen.extend(identities);
            }
            // One master append for the round, in insertion order.
            let added: Vec<(f64, &[(usize, f64)])> = loaded[first..]
                .iter()
                .map(|c| c.as_ref().map(|c| (c.cost, c.entries.as_slice())).unwrap())
                .collect();
            master.add_columns(&added)?;
        }
        out.columns = master.costs().len();
        out.history.push(Iteration {
            iteration,
            lp: out.lp,
            minimum,
            new_columns: inserted,
            columns: out.columns,
            exact,
            alpha: smoothing.alpha,
            effective_alpha: smoothing.effective_alpha,
            angle: first_angle,
            misprices: smoothing.misprices,
            phase: answer_phase,
            pricing_calls: out.pricing_calls,
        });
        if !converged
            && controller.incumbent().is_finite()
            && out.bound.is_finite()
            && gap_smaller_than_tolerance(
                out.bound,
                controller.incumbent(),
                options.gap_absolute,
                options.gap_relative,
            )
        {
            out.status = Status::Optimal;
            out.pruned_by_bound = true;
            return Ok(out);
        }
        if penalties {
            for &id in &out.artificials {
                master.set_cost(id, master.costs()[id] * options.penalty_factor)?;
            }
            out.penalty_updates += 1;
            round_penalty_updates += 1;
            continue;
        }
        if fresh == 0 && exact {
            if !positive && !converged {
                if let Some(universe) = controller.enumerated_universe(&raw, out.bound)? {
                    // Every route column leaves the master; those of the
                    // enumerated set return as new columns after the others,
                    // in their previous order, with their current entries.
                    let allowed: HashSet<_> = universe.iter().map(Column::signature).collect();
                    let mut loaded_set = seen.clone();
                    let removed: Vec<usize> = (0..loaded.len()).filter(|&i| loaded[i].is_some()).collect();
                    let mut regenerated = vec![];
                    for &i in &removed {
                        let column = loaded[i].as_ref().unwrap();
                        if allowed.contains(&column.signature()) {
                            let mut c = column.clone();
                            c.cost = master.costs()[i];
                            c.entries = master.column_entries(i)?;
                            c.tag = out.column_tags[i];
                            regenerated.push(c);
                        }
                    }
                    let (_, map) = master.remove(&[], &removed)?;
                    loaded = loaded.into_iter().zip(&map).filter_map(|(c, m)| m.map(|_| c)).collect();
                    out.column_tags = out.column_tags.iter().zip(&map).filter_map(|(&t, m)| m.map(|_| t)).collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    for (_, column) in &mut cut_artificials {
                        *column = map[*column].unwrap();
                    }
                    controller.remap_columns(&map)?;
                    for column in regenerated {
                        master.add_column(column.cost, &column.entries)?;
                        out.column_tags.push(column.tag);
                        loaded.push(Some(column));
                    }
                    let mut missing: Vec<&Column> =
                        universe.iter().filter(|c| !loaded_set.contains(&c.signature())).collect();
                    missing.sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
                    missing.truncate(options.enumerated_columns_added);
                    for column in missing {
                        let identity = column.signature();
                        if !loaded_set.insert(identity.clone()) {
                            continue;
                        }
                        master.add_column(column.cost, &column.entries)?;
                        out.column_tags.push(column.tag);
                        out.inserted_tags.push(column.tag);
                        seen.insert(identity);
                        loaded.push(Some(column.clone()));
                    }
                    // eval restarts the phase walk whether or not the master changed.
                    out.columns = master.costs().len();
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    round_penalty_updates = 0;
                    continue;
                }
            }
            let mut stop_cuts = !controller.separates() || out.cut_rounds >= options.max_cut_rounds;
            if !stop_cuts && separated_once {
                let denominator = (0.1 * lp_at_separation.abs()).max(1e-9);
                let gain = (out.lp - lp_at_separation) / denominator;
                if gain < options.tailing_off_threshold - 1e-6 {
                    tailing += 1;
                } else {
                    tailing = 0;
                }
                if tailing >= options.tailing_off_counter {
                    stop_cuts = true;
                }
            }
            lp_at_separation = out.lp;
            if !stop_cuts {
                if options.remove_basic_cuts && separated_once {
                    let rows = master.cleanup_cut_rows(initial_rows)?;
                    let columns: Vec<usize> = cut_artificials
                        .iter()
                        .filter(|(r, _)| rows.binary_search(r).is_ok())
                        .map(|&(_, c)| c)
                        .collect();
                    if !rows.is_empty() {
                        let (row_map, map) = master.remove_cuts(&rows, &columns)?;
                        loaded = loaded.into_iter().zip(&map).filter_map(|(c, m)| m.map(|_| c)).collect();
                        out.column_tags =
                            out.column_tags.iter().zip(&map).filter_map(|(&t, m)| m.map(|_| t)).collect();
                        out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                        smoothing.remap_rows(&row_map);
                        cut_artificials = cut_artificials
                            .iter()
                            .filter_map(|&(r, c)| Some((row_map[r]?, map[c]?)))
                            .collect();
                        controller.remap_columns(&map)?;
                        controller.remap_rows(&row_map)?;
                    }
                }
                if !master.solved() && !master.solve() {
                    return Err(invalid("master could not restore its solution before separation"));
                }
                let previous_rows = master.rhs().len();
                let added = controller.separate(master, &out.column_tags)?;
                let removed = controller.take_removed_columns();
                if !removed.is_empty() {
                    let (_, map) = master.remove_cuts(&[], &removed)?;
                    loaded = loaded.into_iter().zip(&map).filter_map(|(c, m)| m.map(|_| c)).collect();
                    out.column_tags = out.column_tags.iter().zip(&map).filter_map(|(&t, m)| m.map(|_| t)).collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    cut_artificials = cut_artificials.iter().filter_map(|&(r, c)| Some((r, map[c]?))).collect();
                    controller.remap_columns(&map)?;
                }
                if added > 0 {
                    if options.cut_artificial_cost > 0. {
                        for row in previous_rows..master.rhs().len() {
                            if master.senses()[row] != RowSense::LessEqual {
                                let c = master.add_column(options.cut_artificial_cost, &[(row, 1.)])?;
                                out.artificials.push(c);
                                out.column_tags.push(-1);
                                loaded.push(None);
                                cut_artificials.push((row, c));
                            }
                            if master.senses()[row] != RowSense::GreaterEqual {
                                let c = master.add_column(options.cut_artificial_cost, &[(row, -1.)])?;
                                out.artificials.push(c);
                                out.column_tags.push(-1);
                                loaded.push(None);
                                cut_artificials.push((row, c));
                            }
                        }
                    }
                    out.cut_rounds += 1;
                    round_penalty_updates = 0;
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    separated_once = true;
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    out.columns = master.costs().len();
                    continue;
                }
            }
            out.status = if positive {
                Status::Infeasible
            } else {
                Status::Optimal
            };
            if positive {
                out.lp = 1e12;
            }
            return Ok(out);
        }
        if advanced {
            continue;
        }
        if fresh == 0 {
            out.status = Status::Stalled;
            return Ok(out);
        }
    }
    Ok(out)
}

/// Alg4EvalByLagrangianDuality::cleanupRestrictedMastColumns (ELF 0xe0380),
/// run before pricing. With N generated columns above the threshold, columns
/// in the primal solution stay; the others are stably sorted by reduced cost
/// (Double tolerance 1e-10 relative + 1e-6, creation order on ties) and only
/// the first (int)(N * ratio) minus the solution columns are kept.
fn cleanup_columns<M: LpMaster>(master: &M, tags: &[i32], options: &Options) -> Vec<usize> {
    let threshold = options.column_cleanup_threshold as i32;
    let generated: Vec<usize> = (0..tags.len()).filter(|&i| tags[i] >= 0).collect();
    let count = generated.len() as i32;
    if threshold >= count {
        return vec![];
    }
    let (solution, mut candidates): (Vec<usize>, Vec<usize>) =
        generated.into_iter().partition(|&i| master.values()[i] != 0.);
    let in_solution = solution.len() as i32;
    if candidates.len() as i32 + in_solution <= threshold {
        return vec![];
    }
    let keep = (count as f32 * options.column_cleanup_ratio) as i32;
    if in_solution >= keep {
        return vec![];
    }
    let rc = master.reduced_costs();
    crate::cuts::stdsort::stable_sort(&mut candidates, |&a, &b| {
        let tolerance = rc[b].abs().max(rc[a].abs()) * 1e-10 + 1e-6;
        if rc[b] - tolerance > rc[a] {
            return true;
        }
        if rc[a] - tolerance > rc[b] {
            return false;
        }
        a < b
    });
    let mut removed = candidates.split_off((keep - in_solution) as usize);
    removed.sort_unstable();
    removed
}
