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
    LpSolveOutcome,
};
use crate::pricing::phases::{self, Phase};
use crate::{invalid, Result};
use std::collections::{HashMap, HashSet};
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
    /// Gap supplied to stabilization before this iteration's LP is accepted.
    pub stabilization_gap: f64,
}

#[derive(Clone, Debug)]
pub struct WarmStart {
    /// Certified lower bound inherited by restrictions of this node.
    /// Negative infinity denotes an unknown bound.
    pub bound: f64,
    /// Accepted parent center; node schedules restart independently.
    pub stabilization: Option<DualPriceSmoothing>,
    /// Generated columns in creation order, including those outside the
    /// restricted master and duplicates created in a single oracle batch.
    pub generated_pool: Vec<Column>,
    pub columns: Vec<Option<Column>>,
    pub artificials: Vec<usize>,
    pub cut_artificials: Vec<(usize, usize)>,
    pub initial_rows: usize,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Metadata of an already populated, solved master. No duplicate base
    /// artificials are added when resuming a node.
    pub warm_start: Option<WarmStart>,
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
            warm_start: None,
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
    pub warm_start: WarmStart,
}

/// The controller owns incumbent tracking and the pricing route universe.
pub trait Controller<M> {
    /// Optional Lagrangian estimate for the most recent batch at these prices.
    /// Only exact batches publish a certified bound. Multiple pricing domains
    /// can contribute different multiplicities.
    fn pricing_bound(&self, _master: &M, _duals: &[f64]) -> Result<Option<f64>> {
        Ok(None)
    }
    /// Override publication of an exact batch's estimate with a conservative
    /// certificate. Negative infinity means its lower bound is still unknown.
    fn certified_pricing_bound(&self, _master: &M, _duals: &[f64]) -> Result<Option<f64>> {
        Ok(None)
    }
    /// Optional aggregate subgradient for the most recent pricing batch.
    fn pricing_subgradient(&self, _rhs: &[f64]) -> Option<Vec<f64>> {
        None
    }
    /// Logical base/branch rows can follow inherited cut rows. Cleanup must
    /// never delete these rows, even when their slacks are basic.
    fn protected_rows(&self) -> Vec<usize> {
        vec![]
    }
    fn unstabilized_rows(&self) -> Vec<usize> {
        vec![]
    }
    fn master_solved(&mut self, _master: &M, _tags: &[i32], _artificials: &[usize]) -> Result<()> {
        Ok(())
    }
    fn incumbent(&self) -> f64 {
        f64::INFINITY
    }
    fn enumerated_universe(&mut self, _duals: &[f64], _bound: f64) -> Result<Option<Vec<Column>>> {
        Ok(None)
    }
    /// Update cached, possibly unloaded columns after a cut-row change.
    fn refresh_generated_columns(&mut self, _pool: &mut [Column]) -> Result<()> {
        Ok(())
    }
    /// Refresh the generated column pool after successful enumeration. A
    /// representative universe need not contain every admissible cached path.
    fn regenerate_enumerated_columns(
        &mut self,
        pool: &[Column],
        _duals: &[f64],
        _bound: f64,
        universe: &[Column],
    ) -> Result<Vec<Column>> {
        let allowed: HashMap<_, _> = universe.iter().map(|c| (c.signature(), c)).collect();
        Ok(pool
            .iter()
            .filter_map(|column| {
                let current = allowed.get(&column.signature())?;
                let mut column = column.clone();
                column.entries = current.entries.clone();
                Some(column)
            })
            .collect())
    }
    fn remap_columns(&mut self, _map: &[Option<usize>]) -> Result<()> {
        Ok(())
    }
    /// indices_remapped after a row (cut) or column deletion.
    fn remap_rows(&mut self, _rows: &[Option<usize>]) -> Result<()> {
        Ok(())
    }
    /// Retain a completed feasible pricing round before cut mutations.
    fn save_pricing_checkpoint(&mut self, _master: &M, _output: &Output) -> Result<()> {
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
    let mut value = master.objective_offset();
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

fn optimality_gap(primal: f64, dual: f64) -> f64 {
    if primal.abs() <= 1e-6 && dual.abs() <= 1e-6 {
        0.
    } else if primal * dual < -1e-6 {
        1.
    } else if primal - (primal.abs().max(dual.abs()) * 1e-10 + 1e-6) > dual {
        (primal - dual) / primal.abs()
    } else {
        (dual - primal) / dual.abs()
    }
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
    // A zero alpha factor disables the reference's smoothing workflow, not
    // only publication of smoothed duals.
    let mut smoothing = DualPriceSmoothing::new(options.smoothing.clone());
    let stabilize = options.stabilize && smoothing.parameters.alpha_factor > 0.;
    let inherited_center = options
        .warm_start
        .as_ref()
        .and_then(|warm| warm.stabilization.as_ref());
    if let Some(previous) = inherited_center {
        smoothing.inherit_center(previous, (schedule.len() - 1) as i32);
    }
    smoothing.exclude_rows(controller.unstabilized_rows());
    if let Some(warm) = &options.warm_start {
        if (!warm.bound.is_finite() && warm.bound != f64::NEG_INFINITY)
            || warm.columns.len() != master.costs().len()
            || warm.initial_rows > master.rhs().len()
            || warm.artificials.iter().any(|&c| c >= master.costs().len())
            || warm
                .cut_artificials
                .iter()
                .any(|&(r, c)| r >= master.rhs().len() || c >= master.costs().len())
            || warm
                .columns
                .iter()
                .flatten()
                .chain(&warm.generated_pool)
                .any(|column| column.entries.iter().any(|&(r, _)| r >= master.rhs().len()))
        {
            return Err(invalid("warm master metadata dimensions"));
        }
    }
    let inherited_bound = options
        .warm_start
        .as_ref()
        .map_or(f64::NEG_INFINITY, |w| w.bound);
    let mut round_lower_bound = if inherited_bound.is_finite() {
        inherited_bound
    } else {
        -1e12
    };
    let mut round_primal_bound = 1e12_f64;
    let mut loaded: Vec<Option<Column>> = options.warm_start.as_ref().map_or_else(
        || options.known.iter().cloned().map(Some).collect(),
        |w| w.columns.clone(),
    );
    loaded.resize(master.costs().len(), None);
    let mut seen: HashSet<_> = loaded.iter().flatten().map(Column::signature).collect();
    // Generated columns exist independently of membership in the restricted
    // master. Columns omitted by a pricing insertion test can become
    // eligible under the strict regeneration cutoff.
    let mut pool: Vec<Column> = options
        .warm_start
        .as_ref()
        .map_or_else(Vec::new, |warm| warm.generated_pool.clone());
    let mut pool_seen: HashSet<_> = pool.iter().map(Column::signature).collect();
    // Include newly supplied initial routes without losing same-batch
    // duplicates when the caller starts with only a populated master.
    let missing: Vec<_> = loaded
        .iter()
        .flatten()
        .filter(|column| !pool_seen.contains(&column.signature()))
        .cloned()
        .collect();
    pool_seen.extend(missing.iter().map(Column::signature));
    pool.extend(missing);
    let mut out = Output {
        status: Status::IterationLimit,
        lp: 0.,
        bound: f64::NEG_INFINITY,
        iterations: 0,
        pricing_calls: 0,
        columns: master.costs().len(),
        history: vec![],
        column_tags: loaded
            .iter()
            .map(|c| c.as_ref().map_or(-1, |c| c.tag))
            .collect(),
        inserted_tags: vec![],
        artificials: options
            .warm_start
            .as_ref()
            .map_or_else(Vec::new, |w| w.artificials.clone()),
        penalty_updates: 0,
        pruned_by_bound: false,
        cut_rounds: 0,
        warm_start: WarmStart {
            bound: f64::NEG_INFINITY,
            stabilization: None,
            generated_pool: vec![],
            columns: vec![],
            artificials: vec![],
            cut_artificials: vec![],
            initial_rows: 0,
        },
    };
    out.column_tags.resize(master.costs().len(), -1);
    if options.artificial_cost > 0. && options.warm_start.is_none() {
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
    // Phase-local stopping
    // value, including heuristic passes; never expose it as a global bound.
    let mut stage_bound = -1e12_f64;
    let mut round_penalty_updates = 0;
    let mut tailing = 0;
    let mut lp_at_separation = 0.0f64;
    let mut separated_once = false;
    let initial_rows = options
        .warm_start
        .as_ref()
        .map_or(master.rhs().len(), |w| w.initial_rows);
    let mut cut_artificials: Vec<(usize, usize)> = options
        .warm_start
        .as_ref()
        .map_or_else(Vec::new, |w| w.cut_artificials.clone());
    let finish = |mut out: Output,
                  loaded: Vec<Option<Column>>,
                  cut_artificials: Vec<(usize, usize)>,
                  pool: Vec<Column>,
                  smoothing: DualPriceSmoothing| {
        out.warm_start = WarmStart {
            bound: out.bound.max(inherited_bound),
            stabilization: stabilize.then_some(smoothing),
            generated_pool: pool,
            columns: loaded,
            artificials: out.artificials.clone(),
            cut_artificials,
            initial_rows,
        };
        Ok(out)
    };
    for iteration in 1..=options.max_iterations {
        if started.elapsed() >= options.time_limit {
            out.status = Status::TimeLimit;
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        let outcome = master.solve_with_time_limit(options.time_limit.saturating_sub(started.elapsed()))?;
        if outcome != LpSolveOutcome::Optimal {
            out.status = match outcome {
                LpSolveOutcome::PrimalInfeasible => Status::Infeasible,
                LpSolveOutcome::DualInfeasible => Status::Unbounded,
                LpSolveOutcome::TimeLimit => Status::TimeLimit,
                LpSolveOutcome::IterationLimit => Status::IterationLimit,
                _ => Status::Stalled,
            };
            if out.status == Status::Infeasible { out.lp = 1e12; }
            if out.status == Status::Unbounded { out.lp = f64::NEG_INFINITY; }
            out.iterations = iteration;
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        out.lp = master.objective();
        out.iterations = iteration;
        let raw = master.duals().to_vec();
        let positive = out.artificials.iter().any(|&i| master.values()[i] > 1e-6);
        controller.master_solved(master, &out.column_tags, &out.artificials)?;
        // Stabilization compares this round's inherited lower bound with
        // its preceding artificial-free restricted-LP objective. The global
        // integer incumbent is a separate stopping/enumeration control.
        let gap = optimality_gap(round_lower_bound, round_primal_bound);
        let primal_slack = round_primal_bound.abs().max(out.lp.abs()) * 1e-10 + options.tolerance;
        if !positive && round_primal_bound - primal_slack > out.lp {
            round_primal_bound = out.lp;
        }
        if stabilize {
            smoothing.after_master(
                &raw,
                master.senses(),
                gap,
                stage as i32,
                iteration == 1 && inherited_center.is_none(),
            );
        }
        // earlyCGtermType1 uses the phase value for stage > 0
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
            // eval clears both values on each phase change.
            stage_bound = -1e12;
            // Each phase is a new solveMastLpPrimPh1A2 call,
            // whose penalty-update count starts at zero.
            round_penalty_updates = 0;
            advanced = true;
            if stabilize {
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
                    loaded = loaded
                        .into_iter()
                        .zip(&map)
                        .filter_map(|(c, m)| m.map(|_| c))
                        .collect();
                    out.column_tags = out
                        .column_tags
                        .iter()
                        .zip(&map)
                        .filter_map(|(&t, m)| m.map(|_| t))
                        .collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    cut_artificials = cut_artificials
                        .iter()
                        .filter_map(|&(r, c)| Some((r, map[c]?)))
                        .collect();
                    controller.remap_columns(&map)?;
                }
            }
            let mut current = if stabilize {
                smoothing.pricing.clone()
            } else {
                raw.clone()
            };
            let mut pending = vec![];
            loop {
                let mut answer = price(stage, &current)?;
                let mut created = vec![];
                for column in &answer.columns {
                    let identity = column.signature();
                    if column.acceptable && !pool_seen.contains(&identity) {
                        pool.push(column.clone());
                        created.push(identity);
                    }
                }
                // Preserve duplicates produced within one oracle batch, as
                // with restricted-master insertion below.
                pool_seen.extend(created);
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
                pass_bound = controller
                    .pricing_bound(master, &current)?
                    .unwrap_or_else(|| lagrangian(master, &current, corrected, options.multiplicity));
                if !pass_bound.is_finite() && pass_bound != f64::NEG_INFINITY {
                    return Err(invalid("invalid pricing lower bound"));
                }
                let slack = stage_bound.abs().max(pass_bound.abs()) * 1e-10 + 1e-6;
                if !(stage_bound - slack > pass_bound) {
                    stage_bound = pass_bound;
                }
                if answer.exact {
                    let certified = controller.certified_pricing_bound(master, &current)?
                        .unwrap_or(pass_bound);
                    if !certified.is_finite() && certified != f64::NEG_INFINITY {
                        return Err(invalid("invalid certified pricing lower bound"));
                    }
                    if certified > out.bound { out.bound = certified; }
                }
                exact = answer.exact;
                answer_phase = stage;
                let mut gradient = master.rhs().to_vec();
                let best = answer
                    .columns
                    .iter()
                    .filter(|c| c.acceptable)
                    .min_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
                let supplied_gradient = controller.pricing_subgradient(master.rhs());
                let subgradient_available = supplied_gradient.is_some() || best.is_some();
                if let Some(supplied) = supplied_gradient {
                    if supplied.len() != gradient.len() || supplied.iter().any(|g| !g.is_finite()) {
                        return Err(invalid("invalid pricing subgradient"));
                    }
                    gradient = supplied;
                } else if let Some(best) = best {
                    // A returned pricing route contributes to the subgradient
                    // even when its rounded reduced cost is zero. Filtering it
                    // by that cost changes the reference's angle inputs.
                    for &(r, a) in &best.entries {
                        if let Some(g) = gradient.get_mut(r) {
                            *g -= options.multiplicity * a;
                        }
                    }
                }
                pending.push(answer.columns);
                if !stabilize {
                    break;
                }
                let first = smoothing.misprices == 0;
                let again = smoothing.after_pricing(&Feedback {
                    negative_columns: fresh,
                    subgradient_available,
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
            if stabilize {
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
            stabilization_gap: gap,
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
            // Fixing/enumeration can retain only routes capable of improving
            // the incumbent. A bound on that restricted problem may exceed
            // the incumbent; the original problem's global bound cannot.
            out.bound = out.bound.min(controller.incumbent());
            return finish(out, loaded, cut_artificials, pool, smoothing);
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
                    // Reconsider the generated pool, including columns that
                    // never entered the master, rather than restricting old
                    // paths to the representative universe's exact identities.
                    let regenerated = controller
                        .regenerate_enumerated_columns(&pool, &raw, out.bound, &universe)?;
                    let mut loaded_set = seen.clone();
                    let removed: Vec<usize> =
                        (0..loaded.len()).filter(|&i| loaded[i].is_some()).collect();
                    let (_, map) = master.remove(&[], &removed)?;
                    loaded = loaded
                        .into_iter()
                        .zip(&map)
                        .filter_map(|(c, m)| m.map(|_| c))
                        .collect();
                    out.column_tags = out
                        .column_tags
                        .iter()
                        .zip(&map)
                        .filter_map(|(&t, m)| m.map(|_| t))
                        .collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    for (_, column) in &mut cut_artificials {
                        *column = map[*column].unwrap();
                    }
                    controller.remap_columns(&map)?;
                    for column in regenerated {
                        master.add_column(column.cost, &column.entries)?;
                        loaded_set.insert(column.signature());
                        seen.insert(column.signature());
                        out.column_tags.push(column.tag);
                        loaded.push(Some(column));
                    }
                    let mut missing: Vec<&Column> = universe
                        .iter()
                        .filter(|c| !loaded_set.contains(&c.signature()))
                        .collect();
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
                        if pool_seen.insert(column.signature()) {
                            pool.push(column.clone());
                        }
                        loaded.push(Some(column.clone()));
                    }
                    // eval restarts the phase walk whether or not the master changed.
                    out.columns = master.costs().len();
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    if out.bound.is_finite() {
                        round_lower_bound = out.bound.max(round_lower_bound);
                        // A completed CG round accepts its LP objective at the
                        // configured convergence tolerance for gap bookkeeping.
                        // The published pricing certificate stays unchanged.
                        if !positive
                            && gap_smaller_than_tolerance(
                                out.bound,
                                out.lp,
                                options.gap_absolute,
                                options.gap_relative,
                            )
                        {
                            round_lower_bound = round_lower_bound.max(out.lp);
                        }
                    }
                    round_primal_bound = 1e12;
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    round_penalty_updates = 0;
                    continue;
                }
            }
            if !positive && controller.separates() {
                let mut checkpoint = finish(
                    out.clone(), loaded.clone(), cut_artificials.clone(),
                    pool.clone(), smoothing.clone(),
                )?;
                checkpoint.status = Status::Optimal;
                controller.save_pricing_checkpoint(master, &checkpoint)?;
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
                // A pass closed by its existing bound has not run pricing.
                // Keep its solved model; cleanup is enabled by a completed
                // exact pricing pass, not merely by entering separation.
                if options.remove_basic_cuts && separated_once && !converged {
                    let protected = controller.protected_rows();
                    let rows: Vec<_> = master
                        .cleanup_cut_rows(initial_rows)?
                        .into_iter()
                        .filter(|r| !protected.contains(r))
                        .collect();
                    let columns: Vec<usize> = cut_artificials
                        .iter()
                        .filter(|(r, _)| rows.binary_search(r).is_ok())
                        .map(|&(_, c)| c)
                        .collect();
                    if !rows.is_empty() {
                        let (row_map, map) = master.remove_cuts(&rows, &columns)?;
                        loaded = loaded
                            .into_iter()
                            .zip(&map)
                            .filter_map(|(c, m)| m.map(|_| c))
                            .collect();
                        out.column_tags = out
                            .column_tags
                            .iter()
                            .zip(&map)
                            .filter_map(|(&t, m)| m.map(|_| t))
                            .collect();
                        out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                        for column in loaded.iter_mut().flatten().chain(&mut pool) {
                            column.entries = column
                                .entries
                                .iter()
                                .filter_map(|&(row, value)| row_map[row].map(|r| (r, value)))
                                .collect();
                        }
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
                    return Err(invalid(
                        "master could not restore its solution before separation",
                    ));
                }
                let previous_rows = master.rhs().len();
                let added = controller.separate(master, &out.column_tags)?;
                // Add the cut artificials in one call, then delete
                // the columns a strengthened relaxation forbids.
                if added > 0 && options.cut_artificial_cost > 0. {
                    let mut artificials: Vec<(usize, f64)> = vec![];
                    for row in previous_rows..master.rhs().len() {
                        if master.senses()[row] != RowSense::LessEqual {
                            artificials.push((row, 1.));
                        }
                        if master.senses()[row] != RowSense::GreaterEqual {
                            artificials.push((row, -1.));
                        }
                    }
                    let entries: Vec<[(usize, f64); 1]> =
                        artificials.iter().map(|&e| [e]).collect();
                    let batch: Vec<_> = entries
                        .iter()
                        .map(|e| (options.cut_artificial_cost, e.as_slice()))
                        .collect();
                    let columns = master.add_columns(&batch)?;
                    for (c, &(row, _)) in columns.zip(&artificials) {
                        out.artificials.push(c);
                        out.column_tags.push(-1);
                        loaded.push(None);
                        cut_artificials.push((row, c));
                    }
                }
                let removed = controller.take_removed_columns();
                if !removed.is_empty() {
                    let (_, map) = master.remove_cuts(&[], &removed)?;
                    loaded = loaded
                        .into_iter()
                        .zip(&map)
                        .filter_map(|(c, m)| m.map(|_| c))
                        .collect();
                    out.column_tags = out
                        .column_tags
                        .iter()
                        .zip(&map)
                        .filter_map(|(&t, m)| m.map(|_| t))
                        .collect();
                    out.artificials = out.artificials.iter().filter_map(|&i| map[i]).collect();
                    cut_artificials = cut_artificials
                        .iter()
                        .filter_map(|&(r, c)| Some((r, map[c]?)))
                        .collect();
                    controller.remap_columns(&map)?;
                }
                // Separation may add rows or change existing cut memories.
                // Keep warm metadata synchronized with the actual LP, and
                // let the controller rebuild coefficients for unloaded paths.
                for (index, column) in loaded.iter_mut().enumerate() {
                    if let Some(column) = column {
                        column.entries = master.column_entries(index)?;
                    }
                }
                controller.refresh_generated_columns(&mut pool)?;
                seen = loaded.iter().flatten().map(Column::signature).collect();
                pool_seen = pool.iter().map(Column::signature).collect();
                if added > 0 {
                    out.cut_rounds += 1;
                    round_penalty_updates = 0;
                    smoothing.restart_schedule((schedule.len() - 1) as i32);
                    if out.bound.is_finite() {
                        round_lower_bound = out.bound.max(round_lower_bound);
                        // A completed CG round accepts its LP objective at the
                        // configured convergence tolerance for gap bookkeeping.
                        // The published pricing certificate stays unchanged.
                        if !positive
                            && gap_smaller_than_tolerance(
                                out.bound,
                                out.lp,
                                options.gap_absolute,
                                options.gap_relative,
                            )
                        {
                            round_lower_bound = round_lower_bound.max(out.lp);
                        }
                    }
                    round_primal_bound = 1e12;
                    separated_once = true;
                    stage = schedule.len() - 1;
                    stage_bound = -1e12;
                    out.columns = master.costs().len();
                    continue;
                }
            }
            out.status = if positive {
                // Exact pricing proves optimality for these finite artificial
                // penalties, not infeasibility after removing the artificials.
                // A feasible real column can simply cost more than the last
                // allowed penalty. No phase-I/Farkas certificate exists here.
                Status::Stalled
            } else {
                Status::Optimal
            };
            if positive {
                out.lp = 1e12;
            }
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
        if advanced {
            continue;
        }
        if fresh == 0 {
            out.status = Status::Stalled;
            return finish(out, loaded, cut_artificials, pool, smoothing);
        }
    }
    finish(out, loaded, cut_artificials, pool, smoothing)
}

/// Alg4EvalByLagrangianDuality::cleanupRestrictedMastColumns,
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
    let (solution, mut candidates): (Vec<usize>, Vec<usize>) = generated
        .into_iter()
        .partition(|&i| master.values()[i] != 0.);
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
