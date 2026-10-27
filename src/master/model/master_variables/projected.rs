//! Proof-only static-domain projection when an unbounded dual box is adverse.
use super::*;
use crate::master::preprocessing::{propagate_conservative, Constraint};

pub(in crate::master::model) fn infer<const D: usize>(state: &State<D>) -> Option<Vec<Bounds>> {
    if Instant::now() >= state.deadline {
        return None;
    }
    let rows = state.shared_rows.len();
    // Drop only a network contribution whose sign proves the resulting
    // static inequality necessary. Mixed signs retain no finite assertion.
    let mut lower_zero = vec![true; rows];
    let mut upper_zero = vec![true; rows];
    for oracle in &state.oracles {
        for &(row, a) in oracle
            .network
            .arc_entries
            .iter()
            .flatten()
            .chain(oracle.network.route_entries.iter())
        {
            if a < 0. {
                lower_zero[row] = false;
            }
            if a > 0. {
                upper_zero[row] = false;
            }
        }
        for entry in &oracle.activity_rows {
            lower_zero[entry.row] = false;
            upper_zero[entry.row] = false;
        }
        for &(_, row) in oracle
            .nonlinear_rows
            .iter()
            .chain(&oracle.knapsack_rows)
            .chain(&oracle.presence_rows)
        {
            upper_zero[row] = false;
        }
    }
    let mut entries = vec![vec![]; rows];
    for (v, variable) in state.master_variables.iter().enumerate() {
        for &(row, a) in &variable.entries {
            if a != 0. {
                entries[row].push((v, a));
            }
        }
    }
    let mut constraints = vec![];
    for (r, row) in state.shared_rows.iter().enumerate() {
        let rhs = state.rhs[state.shared_offset + r];
        if row.sense != RowSense::GreaterEqual && lower_zero[r] {
            constraints.push(Constraint {
                entries: entries[r].clone(),
                sense: RowSense::LessEqual,
                rhs,
            });
        }
        if row.sense != RowSense::LessEqual && upper_zero[r] {
            constraints.push(Constraint {
                entries: entries[r].clone(),
                sense: RowSense::GreaterEqual,
                rhs,
            });
        }
    }
    let mut bounds: Vec<_> = state.master_variables.iter().map(|v| v.bounds).collect();
    let report = propagate_conservative(&mut bounds, &constraints, 1_000_000).ok()?;
    if report.infeasible || Instant::now() >= state.deadline {
        None
    } else {
        Some(bounds)
    }
}
