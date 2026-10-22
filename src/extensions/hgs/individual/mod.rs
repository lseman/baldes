//! Individual representation for HGS population.
//!
//! An individual encodes a complete VRP solution with:
//! - A **giant tour** (`chrom_t`) — permutation of all customers
//! - A set of **routes** (`chrom_r`) — feasible vehicle tours derived from the giant tour
//! - Predecessor/successor maps for fast arc queries
//! - Cost, feasibility, and fitness information

use crate::extensions::hgs::params::MY_EPSILON;

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use crate::extensions::hgs::{Params, ProblemType, TimeWindow};

static NEXT_INDIVIDUAL_ID: AtomicUsize = AtomicUsize::new(1);

fn next_individual_id() -> usize {
    NEXT_INDIVIDUAL_ID.fetch_add(1, AtomicOrdering::Relaxed)
}

/// Wrapper for f64 that implements Ord by comparing bit patterns.
/// This allows using f64 as a BTreeMap key. Note: NaN and negative zero
/// have special behavior; we assume valid positive distances only.
#[derive(Clone, Copy, Debug)]
pub struct SortableF64(pub f64);

impl PartialEq for SortableF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for SortableF64 {}

impl PartialOrd for SortableF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortableF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Raw bit patterns order negative values above positive ones.
        self.0.total_cmp(&other.0)
    }
}

/// Cost and penalty breakdown for a solution.
#[derive(Clone, Debug, Default)]
pub struct CostSol {
    /// Penalized cost = distance + capacity_penalty + time_warp_penalty + wait_penalty
    pub penalized_cost: f64,
    /// Number of active routes
    pub nb_routes: usize,
    /// Total travel distance across all routes
    pub distance: f64,
    /// Total capacity excess across all routes
    pub capacity_excess: i32,
    /// Total wait time across all routes
    pub wait_time: f64,
    /// Total time warp across all routes
    pub time_warp: f64,
    /// Whether the solution is feasible (no capacity or time-warp violations)
    pub is_feasible: bool,
}

/// A solution individual in the HGS population.
pub struct Individual<'p> {
    /// Unique identifier for this individual.
    pub id: usize,
    /// Reference to problem parameters.
    pub params: &'p Params,
    /// Number of vehicles available.
    pub nb_vehicles: usize,
    /// Solution cost breakdown.
    pub my_cost_sol: CostSol,
    /// Giant tour — permutation of all customers (1-based indices into cli).
    pub chrom_t: Vec<usize>,
    /// Routes per vehicle. `chrom_r[r]` is the ordered list of customers on route r.
    pub chrom_r: Vec<Vec<usize>>,
    /// `successors[i]` = customer after i in the giant tour (0 = depot).
    pub successors: Vec<usize>,
    /// `predecessors[i]` = customer before i in the giant tour (0 = depot).
    pub predecessors: Vec<usize>,
    /// Proximity index: `(SortableF64, individual_id)` sorted by distance.
    /// Used for diversity management.
    pub indivs_per_proximity: BTreeSet<(SortableF64, usize)>,
    /// Feasibility flag.
    pub is_feasible: bool,
    /// Biased fitness value (penalized cost adjusted by frequency).
    pub biased_fitness: f64,
}

impl<'p> Clone for Individual<'p> {
    fn clone(&self) -> Self {
        Self {
            id: next_individual_id(),
            params: self.params,
            nb_vehicles: self.nb_vehicles,
            my_cost_sol: self.my_cost_sol.clone(),
            chrom_t: self.chrom_t.clone(),
            chrom_r: self.chrom_r.clone(),
            successors: self.successors.clone(),
            predecessors: self.predecessors.clone(),
            indivs_per_proximity: BTreeSet::new(),
            is_feasible: self.is_feasible,
            biased_fitness: self.biased_fitness,
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/extensions/hgs/individual/tests.rs"]
mod tests;

mod construction;
mod diversity;
mod evaluation;
mod representation;
