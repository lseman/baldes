//! Shared resource step rows with exact binary-grid additive loads.
use super::*;
mod lattice;
pub(super) mod propagation;
mod replay;
use crate::cuts::knapsack::{ResourceKnapsackCut, RouteLoadKnapsackCut};
pub(super) use replay::requires_forward;

/// Nonnegative monotone terminal-load coefficient on one network.
/// Resource is a solver coordinate; scale converts it to the cut's load units.
#[derive(Clone, Debug)]
pub struct ResourceKnapsackCoefficients {
    pub network: i32,
    pub resource: usize,
    pub scale: f64,
    pub cut: RouteLoadKnapsackCut,
}

pub(super) fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    let mut result = vec![];
    for &(index, row) in &oracle.knapsack_rows {
        let cut = &oracle.solver.cuts.knapsack_cuts[index];
        let &(resource, scale) = &cut.graph_resources[&oracle.network.id];
        let q = replay::load(oracle, route, resource)?;
        let load = (q * scale + 1e-6).floor() as i32;
        result.push((row, cut.cut.coefficient(load) as f64));
    }
    Ok(result)
}

pub(super) fn validate<const D: usize>(
    g: &Graph<D>,
    c: &ResourceKnapsackCoefficients,
) -> Result<()> {
    let r = c.resource;
    if r >= g.resources.len() || !c.scale.is_finite() || c.scale <= 0. {
        return Err(invalid("resource knapsack resource coordinate or scale"));
    }
    lattice::validate(g, r)?;
    let mut previous = 0;
    for &a in c.cut.coefficients.values() {
        if a < previous {
            return Err(invalid("resource knapsack monotone coefficients"));
        }
        previous = a;
    }
    Ok(())
}

impl<const D: usize> NodeSnapshot<D> {
    /// Install a caller-supplied valid <= row with monotone resource-load
    /// coefficients. Exact binary-grid initial usage, waiting and vertex/arc windows use
    /// earliest forward replay; such schedules select forward-only pricing.
    pub fn with_resource_knapsack_row(
        &self,
        row: Row,
        coefficients: &[ResourceKnapsackCoefficients],
    ) -> Result<Self> {
        validate_row(&self.state, row.sense, coefficients)?;
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(row, &[])?;
        install(&mut node.state, shared, coefficients)?;
        super::nonlinear::refresh_row(node, shared)
    }

    pub fn with_resource_knapsack_master_row(
        &self,
        row: Row,
        coefficients: &[ResourceKnapsackCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_resource_knapsack_row(row, coefficients)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }
}

pub(super) fn validate_row<const D: usize>(
    state: &State<D>,
    sense: RowSense,
    coefficients: &[ResourceKnapsackCoefficients],
) -> Result<()> {
    if sense != RowSense::LessEqual {
        return Err(invalid("resource knapsack row requires <= sense"));
    }
    let mut seen = HashSet::new();
    for c in coefficients {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == c.network)
            .ok_or_else(|| invalid("unknown resource knapsack network"))?;
        if !seen.insert(c.network) {
            return Err(invalid("duplicate resource knapsack network"));
        }
        validate(&oracle.network.graph, c)?;
    }
    Ok(())
}

pub(super) fn install<const D: usize>(
    state: &mut State<D>,
    shared: usize,
    coefficients: &[ResourceKnapsackCoefficients],
) -> Result<()> {
    validate_row(state, state.shared_rows[shared].sense, coefficients)?;
    for c in coefficients {
        let oracle = state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == c.network)
            .unwrap();
        let index = oracle.solver.cuts.knapsack_cuts.len();
        oracle.solver.cuts.knapsack_cuts.push(ResourceKnapsackCut {
            id: i32::try_from(index).map_err(|_| invalid("resource knapsack index"))?,
            row: shared,
            cut_class: 0,
            graph_resources: BTreeMap::from([(c.network, (c.resource, c.scale))]),
            cut: c.cut.clone(),
            dual: 0.,
        });
        oracle.solver.knapsack_exact_coefficients = true;
        oracle.knapsack_rows.push((index, shared));
    }
    for oracle in &mut state.oracles {
        if requires_forward(oracle) {
            oracle
                .solver
                .set_option("BALDESuseBidirectionalSearch", 0.)?;
        }
    }
    Ok(())
}
