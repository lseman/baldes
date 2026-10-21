//! Nonlinear shared rows over sets of original-variable events.
use super::*;
pub(super) mod propagation;
use crate::cuts::soft_conflicts::{PresencePricing, PresenceRow, PricingInput};

/// Presence coefficient for one pricing subproblem. Event lists are aligned
/// with physical graph arcs; repeated visits emit each event only once.
#[derive(Clone, Debug)]
pub struct PresenceCoefficients {
    pub network: i32,
    pub row: PresenceRow,
    pub arc_events: Vec<Vec<usize>>,
}

pub(super) fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    if oracle.presence_rows.is_empty() {
        return Ok(vec![]);
    }
    let arcs = route
        .arcs
        .iter()
        .map(|id| {
            oracle
                .index
                .get(id)
                .copied()
                .ok_or_else(|| invalid("unknown presence route arc"))
        })
        .collect::<Result<Vec<_>>>()?;
    let values = oracle.solver.presence_route_coefficients(&arcs)?;
    Ok(oracle
        .presence_rows
        .iter()
        .map(|&(input, row)| (row, values[input] as f64))
        .collect())
}

impl<const D: usize> NodeSnapshot<D> {
    /// Append a caller-supplied valid presence row. PairCount counts satisfied
    /// stored pairs (including duplicates); AnyPair charges at most once.
    /// Both fixing and pricing replay the same original physical arc events.
    pub fn with_presence_row(
        &self,
        row: Row,
        coefficients: &[PresenceCoefficients],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(row, &[])?;
        install(&mut node.state, shared, coefficients)?;
        super::nonlinear::refresh_row(node, shared)
    }

    /// Presence row coupled to independent master variables.
    pub fn with_presence_master_row(
        &self,
        row: Row,
        coefficients: &[PresenceCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_presence_row(row, coefficients)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }
}

/// Install already indexed original-domain presence coefficients. Fresh import
/// rebuilds use this before fixed-route replay and whole-domain preprocessing.
pub(super) fn install<const D: usize>(
    state: &mut State<D>,
    shared: usize,
    coefficients: &[PresenceCoefficients],
) -> Result<()> {
    let mut seen = HashSet::new();
    for c in coefficients {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == c.network)
            .ok_or_else(|| invalid("unknown presence network"))?;
        if !seen.insert(c.network) {
            return Err(invalid("duplicate presence network"));
        }
        let prepared = PresencePricing::new(
            oracle.network.graph.arcs.len(),
            vec![PricingInput {
                row: c.row.clone(),
                arc_events: c.arc_events.clone(),
                dual: 0.,
            }],
        )?;
        prepared.validate_engine(oracle.network.graph.arcs.len())?;
    }
    for c in coefficients {
        let oracle = state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == c.network)
            .unwrap();
        let mut inputs = oracle.solver.presence_rows().to_vec();
        let index = inputs.len();
        inputs.push(PricingInput {
            row: c.row.clone(),
            arc_events: c.arc_events.clone(),
            dual: 0.,
        });
        oracle.solver.set_presence_rows(inputs)?;
        oracle.presence_rows.push((index, shared));
    }
    Ok(())
}
