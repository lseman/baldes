use std::collections::{BTreeMap, BTreeSet};

use crate::core::options::Options;
use crate::{invalid, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackParameters {
    pub max_cuts_per_round: usize,
    pub use_rounding: bool,
    pub max_one_k: i32,
    pub violation_tolerance: f64,
}

impl Default for KnapsackParameters {
    fn default() -> Self {
        Self {
            max_cuts_per_round: 100,
            use_rounding: false,
            max_one_k: 1,
            violation_tolerance: 0.02,
        }
    }
}

pub fn knapsack_parameters(
    options: &Options,
    initial: KnapsackParameters,
) -> Result<KnapsackParameters> {
    let mut parameters = initial;
    let mode = options.get("BALDESresConsKnapsackCutsMode")?;
    parameters.use_rounding = mode > 0.0;
    parameters.max_one_k = mode.abs() as i32;
    let per_round = options.get("BALDESresConsKnapsackCutsMaxNumPerRound")?;
    if per_round < 1.0 {
        return Err(invalid(
            "BALDESresConsKnapsackCutsMaxNumPerRound must be positive",
        ));
    }
    parameters.max_cuts_per_round = per_round as usize;
    validate_parameters(&parameters)?;
    Ok(parameters)
}

fn validate_parameters(parameters: &KnapsackParameters) -> Result<()> {
    if parameters.max_cuts_per_round < 1 || parameters.max_cuts_per_round > i32::MAX as usize {
        return Err(invalid("knapsack max_cuts_per_round must be at least 1"));
    }
    if parameters.max_one_k < 0 {
        return Err(invalid("knapsack max_one_k must be non-negative"));
    }
    if !parameters.violation_tolerance.is_finite() || parameters.violation_tolerance < 0.0 {
        return Err(invalid("knapsack violation_tolerance must be non-negative"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnapsackKind {
    Rounding,
    Steps,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteLoadKnapsackCut {
    pub kind: KnapsackKind,
    pub numerator: i32,
    pub denominator: i32,
    pub rhs: i32,
    pub lhs: f64,
    pub violation: f64,
    pub thresholds: Vec<i32>,
    pub coefficients: BTreeMap<i32, i32>,
}

impl RouteLoadKnapsackCut {
    pub fn coefficient(&self, load: i32) -> i32 {
        if load < 1 {
            return 0;
        }
        match self.coefficients.range(..=load).next_back() {
            Some((_, &val)) => val,
            None => 0,
        }
    }
}

mod pricing;
pub use pricing::KnapsackPricing;

mod resource;
pub use resource::{
    KnapsackColumn, KnapsackGraph, KnapsackPreparationRow, KnapsackResource, KnapsackRow,
    ResourceKnapsackCut, ResourceKnapsackSeparator,
};

mod separation;
pub use separation::KnapsackSeparator;

mod candidates;
use candidates::*;
