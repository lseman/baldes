//! Shared rows over deterministic activities and selected paired source values.
use super::*;
use crate::graph::coupled::CoupledResource;

/// One coefficient times an intrinsic custom/cumulative activity or selected paired value. Resource is
/// the original coupled descriptor index, independent of scalar coordinates.
/// Repeated descriptors contribute repeatedly; signed finite coefficients are
/// allowed. Model-cost suppression does not change the row activity.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccumulatedActivityCoefficients {
    pub network: i32,
    pub resource: usize,
    pub coefficient: f64,
}

/// One intrinsic activity contribution to a shared row of its owning network.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccumulatedActivityEntry {
    pub resource: usize,
    pub row: usize,
    pub coefficient: f64,
}

pub(super) fn paired<const D: usize>(graph: &Graph<D>, resource: usize) -> bool {
    matches!(
        graph
            .coupled_kernel()
            .and_then(|kernel| kernel.definitions().get(resource)),
        Some(CoupledResource::PairedFront {
            mode: crate::graph::coupled::PairedResourceMode::CostBounded
                | crate::graph::coupled::PairedResourceMode::CostUnbounded,
        })
    )
}

pub(super) fn validate<const D: usize>(
    g: &Graph<D>,
    c: &AccumulatedActivityCoefficients,
) -> Result<()> {
    if !c.coefficient.is_finite()
        || !(paired(g, c.resource)
            || matches!(
                g.coupled_kernel()
                    .and_then(|k| k.definitions().get(c.resource)),
                Some(
                    CoupledResource::CustomResource { .. } | CoupledResource::CumulativeCost { .. }
                )
            ))
    {
        return Err(invalid(
            "activity row requires a finite coefficient and deterministic activity descriptor",
        ));
    }
    Ok(())
}

pub(super) fn install<const D: usize>(
    state: &mut State<D>,
    row: usize,
    coefficients: &[AccumulatedActivityCoefficients],
) -> Result<()> {
    for c in coefficients {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == c.network)
            .ok_or_else(|| invalid("unknown activity row network"))?;
        validate(&oracle.network.graph, c)?;
    }
    for c in coefficients {
        let oracle = state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == c.network)
            .unwrap();
        oracle.activity_rows.push(AccumulatedActivityEntry {
            resource: c.resource,
            row,
            coefficient: c.coefficient,
        });
    }
    Ok(())
}

pub(super) fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    let mut values = BTreeMap::new();
    for (row, value) in entries_for(oracle, route, false)?
        .into_iter()
        .chain(entries_for(oracle, route, true)?)
    {
        *values.entry(row).or_insert(0.) += value;
    }
    Ok(values.into_iter().collect())
}
pub(super) fn paired_entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    entries_for(oracle, route, true)
}
pub(super) fn deterministic_entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    entries_for(oracle, route, false)
}
fn entries_for<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    paired_only: bool,
) -> Result<Vec<(usize, f64)>> {
    if oracle.activity_rows.is_empty() {
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
                .ok_or_else(|| invalid("unknown activity row arc"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rows = BTreeMap::<usize, f64>::new();
    for c in oracle
        .activity_rows
        .iter()
        .filter(|c| c.coefficient != 0. && paired(&oracle.network.graph, c.resource) == paired_only)
    {
        let activity = column_value(oracle, route, &arcs, c.resource)?;
        let value = activity * c.coefficient;
        let sum = rows.entry(c.row).or_default();
        *sum += value;
        if !value.is_finite() || !sum.is_finite() {
            return Err(invalid("activity row coefficient overflow"));
        }
    }
    Ok(rows.into_iter().collect())
}

/// Original column activity; selected paired metadata must match recipient replay.
pub(super) fn column_value<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    arcs: &[usize],
    resource: usize,
) -> Result<f64> {
    value_on_graph(&oracle.network.graph, route, arcs, resource)
}

pub(crate) fn value_on_graph<const D: usize>(
    graph: &Graph<D>,
    route: &Route,
    arcs: &[usize],
    resource: usize,
) -> Result<f64> {
    if paired(graph, resource) {
        if !route.selected_values_match(graph, arcs)? {
            return Err(invalid("paired row values lack matching retained split"));
        }
        route
            .coupled_values
            .iter()
            .find(|(r, _)| *r == resource)
            .map(|(_, v)| *v)
            .ok_or_else(|| invalid("paired row value missing"))
    } else {
        graph
            .coupled_kernel()
            .ok_or_else(|| invalid("missing activity kernel"))?
            .accumulated_activity(resource, arcs)
    }
}

pub(super) fn price<const D: usize>(
    oracle: &mut Oracle<D>,
    duals: &[f64],
    offset: usize,
) -> Result<()> {
    let mut adjustments = BTreeMap::<usize, f64>::new();
    for c in oracle
        .activity_rows
        .iter()
        .filter(|c| !paired(&oracle.network.graph, c.resource))
    {
        let value = adjustments.entry(c.resource).or_default();
        *value -= duals[offset + c.row] * c.coefficient;
        if !value.is_finite() {
            return Err(invalid("activity row dual overflow"));
        }
    }
    oracle
        .solver
        .graph
        .set_coupled_pricing_adjustments(&adjustments.into_iter().collect::<Vec<_>>())
}

/// Source singleton proof for a stored row value. A proposal is sufficient to
/// prove attainment; optimization completeness is required only for a bound.
pub(super) fn certificate_value<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
    resource: usize,
    work: &mut usize,
    deadline: Instant,
) -> Option<[f64; 2]> {
    let graph = &oracle.network.graph;
    let kernel = graph.coupled_kernel()?;
    let arcs = route
        .arcs
        .iter()
        .map(|id| oracle.index.get(id).copied())
        .collect::<Option<Vec<_>>>()?;
    if !paired(graph, resource) {
        let (valid, interval) =
            kernel.certify_accumulated_activity(resource, &arcs, work, Some(deadline))?;
        return valid.then_some(interval);
    }
    if Instant::now() >= deadline {
        return None;
    }
    let value = route.coupled_values.iter().find(|(r, _)| *r == resource)?.1;
    if kernel.retained_paired_machine_resource() != Some(resource) {
        return None;
    }
    let required = arcs.len().checked_add(1)?.checked_pow(2)?.checked_mul(16)?;
    *work = work.checked_sub(required)?;
    let selected = kernel
        .joined_machine_path_matching(&arcs, Some(value))
        .ok()??;
    (Instant::now() < deadline && selected.1 == Some(value)).then_some([value, value])
}

impl<const D: usize> NodeSnapshot<D> {
    /// Append a shared row with deterministic activities and optional
    /// additive arc/route terms. Original-path replay, pricing and fixed-route
    /// RHS shifts all retain the same descriptor identities.
    pub fn with_accumulated_activity_row(
        &self,
        row: Row,
        coefficients: &[AccumulatedActivityCoefficients],
        linear: &[RowCoefficients],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(row, linear)?;
        install(&mut node.state, shared, coefficients)?;
        super::nonlinear::refresh_row(node, shared)
    }

    pub fn with_accumulated_activity_master_row(
        &self,
        row: Row,
        coefficients: &[AccumulatedActivityCoefficients],
        linear: &[RowCoefficients],
        masters: &[(i32, f64)],
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_accumulated_activity_row(row, coefficients, linear)?;
        node.attach_master_coefficients(shared, masters)?;
        Ok(node)
    }
}
