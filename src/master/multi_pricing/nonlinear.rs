//! Shared nonlinear rows using registered pricing-kernel coefficients.
use super::*;
use crate::cuts::{CutClass, MemoryKind, RankOneCut};
mod kernels;
pub(super) mod propagation;
mod registered;
pub(super) use kernels::RowKernels;

/// Existing nonnegative registry coefficient with physical graph support.
/// Both master replay and labeling use the normalized descriptor.
#[derive(Clone, Debug)]
pub struct RegisteredCutCoefficients {
    pub network: i32,
    pub cut: RankOneCut,
}

pub(super) fn arc_upper<const D: usize>(oracle: &Oracle<D>, index: usize, arc: usize) -> f64 {
    let cut = &oracle.solver.cuts.cuts[index];
    let physical = &oracle.network.graph.arcs[arc];
    f64::from(cut.weights[physical.head] > 0)
        + f64::from(!cut.visit_once && cut.arc_weights.get(&physical.id).is_some_and(|&w| w > 0))
}

/// Coefficient floor(sum of visited vertex weights / denominator).
/// Weights include all physical vertices; source and sink must have weight zero.
/// Each visit counts, including repeated visits. Omitted networks contribute zero.
#[derive(Clone, Debug)]
pub struct RankOneCoefficients {
    pub network: i32,
    pub weights: Vec<i32>,
    pub denominator: i32,
}

pub(super) fn entries<const D: usize>(
    oracle: &Oracle<D>,
    route: &Route,
) -> Result<Vec<(usize, f64)>> {
    if oracle.nonlinear_rows.is_empty()
        && oracle.knapsack_rows.is_empty()
        && oracle.presence_rows.is_empty()
        && oracle.resource_groups.is_empty()
    {
        return Ok(vec![]);
    }
    let physical = route
        .arcs
        .iter()
        .map(|id| {
            let arc = &oracle.network.graph.arcs[*oracle
                .index
                .get(id)
                .ok_or_else(|| invalid("unknown nonlinear route arc"))?];
            Ok((arc.tail, arc.head, arc.id))
        })
        .collect::<Result<Vec<_>>>()?;
    // A shared row may contain several kernels from the same oracle. Replay
    // must publish one summed coefficient, as pricing applies the row dual to
    // every kernel. Keep integer accumulation exact before converting to LP.
    let mut sums = std::collections::BTreeMap::<usize, i64>::new();
    for &(cut, row) in &oracle.nonlinear_rows {
        let coefficient = i64::from(oracle.solver.cuts.route_coefficient(
            &oracle.solver.graph,
            cut,
            &physical,
            false,
        )?);
        let sum = sums.entry(row).or_default();
        *sum = sum
            .checked_add(coefficient)
            .filter(|&v| v >= 0 && v <= (1_i64 << 53))
            .ok_or_else(|| invalid("composite nonlinear coefficient overflow"))?;
    }
    let mut result: Vec<_> = sums.into_iter().map(|(r, a)| (r, a as f64)).collect();
    result.extend(super::resource_rows::entries(oracle, route)?);
    result.extend(super::presence_rows::entries(oracle, route)?);
    result.extend(super::resource_groups::entries(oracle, route));
    Ok(result)
}

impl<const D: usize> NodeSnapshot<D> {
    /// Append a caller-supplied valid full-memory floor row to every relevant
    /// oracle and the retained master. Fixing shifts its RHS using the actual
    /// nonlinear route coefficient, never a sum of independent arc weights.
    pub fn with_rank_one_row(
        &self,
        row: Row,
        coefficients: &[RankOneCoefficients],
    ) -> Result<Self> {
        let mut seen = HashSet::new();
        for c in coefficients {
            let oracle = self
                .state
                .oracles
                .iter()
                .find(|o| o.network.id == c.network)
                .ok_or_else(|| invalid("unknown nonlinear row network"))?;
            let g = &oracle.network.graph;
            if !seen.insert(c.network)
                || c.weights.len() != g.vertices.len()
                || !(1..8).contains(&c.denominator)
                || c.weights.iter().any(|&w| w < 0 || w >= c.denominator)
                || c.weights[g.source] != 0
                || c.weights[g.sink] != 0
            {
                return Err(invalid("nonlinear row weights or denominator"));
            }
        }
        let registered: Vec<_> = coefficients
            .iter()
            .map(|c| RegisteredCutCoefficients {
                network: c.network,
                cut: RankOneCut {
                    weights: c.weights.clone(),
                    graph_support: BTreeMap::new(),
                    graph_pricing_support: BTreeMap::new(),
                    native_kpath_support: None,
                    memory: vec![true; c.weights.len()],
                    graph_dependent_memory: false,
                    graph_memories: BTreeMap::new(),
                    memory_kind: MemoryKind::Vertex,
                    arc_memory: Default::default(),
                    graph_arc_memories: BTreeMap::new(),
                    arc_weights: BTreeMap::new(),
                    visit_once: false,
                    native_type2: false,
                    denominator: c.denominator,
                    dual: 0.,
                    class: CutClass::Packing,
                    rhs: 0,
                    kind: 0,
                    id: 0,
                    ordered_support: None,
                },
            })
            .collect();
        self.with_registered_cut_row(row, &registered)
    }

    /// Install registered packing/covering, limited-memory, arc-weighted or
    /// strong k-path coefficients across networks. The caller supplies a valid
    /// row; this validates the pricing descriptor, not mathematical cut validity.
    pub fn with_registered_cut_row(
        &self,
        row: Row,
        coefficients: &[RegisteredCutCoefficients],
    ) -> Result<Self> {
        self.registered_row(row, coefficients, false)
    }

    /// Append one shared row whose coefficient is the sum of all supplied
    /// kernels. Multiple descriptors may name the same network; repeated
    /// descriptors intentionally contribute repeatedly. This is a sum, not
    /// an indicator of whether any kernel contributes.
    pub fn with_composite_cut_row(
        &self,
        row: Row,
        coefficients: &[RegisteredCutCoefficients],
    ) -> Result<Self> {
        self.registered_row(row, coefficients, true)
    }

    fn registered_row(
        &self,
        row: Row,
        coefficients: &[RegisteredCutCoefficients],
        composite: bool,
    ) -> Result<Self> {
        let shared = self.state.shared_rows.len();
        let mut node = self.with_row(row, &[])?;
        install_registered(&mut node.state, shared, coefficients, composite)?;
        refresh_row(node, shared)
    }
}

pub(super) fn install_registered<const D: usize>(
    state: &mut State<D>,
    shared: usize,
    coefficients: &[RegisteredCutCoefficients],
    composite: bool,
) -> Result<()> {
    let mut seen = HashSet::new();
    let mut normalized = vec![];
    for c in coefficients {
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == c.network)
            .ok_or_else(|| invalid("unknown nonlinear row network"))?;
        if !seen.insert(c.network) && !composite {
            return Err(invalid("duplicate nonlinear row network"));
        }
        normalized.push((
            c.network,
            registered::normalize(&c.cut, &oracle.solver.graph)?,
        ));
    }
    for (network, mut cut) in normalized {
        let oracle = state
            .oracles
            .iter_mut()
            .find(|o| o.network.id == network)
            .unwrap();
        let index = oracle.solver.cuts.cuts.len();
        cut.id = i32::try_from(index).map_err(|_| invalid("nonlinear cut index"))?;
        oracle.solver.cuts.cuts.push(cut);
        oracle.nonlinear_rows.push((index, shared));
    }
    Ok(())
}

pub(super) fn refresh_row<const D: usize>(
    mut node: NodeSnapshot<D>,
    shared: usize,
) -> Result<NodeSnapshot<D>> {
    let physical = node.state.shared_offset + shared;
    let value = |route: &NetworkRoute| -> Result<f64> {
        let oracle = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        Ok(entries(oracle, &route.route)?
            .into_iter()
            .find(|&(r, _)| r == shared)
            .map_or(0., |(_, a)| a))
    };
    let mut rhs = node.state.shared_rows[shared].rhs;
    for fixed in &node.state.fixed_routes {
        rhs -= fixed.quantity * value(&fixed.route)?;
    }
    if !rhs.is_finite() {
        return Err(invalid("fixed nonlinear row overflow"));
    }
    for (column, metadata) in node.warm.columns.iter_mut().enumerate() {
        if let Some(metadata) = metadata {
            if metadata.tag < 0 {
                continue;
            }
            let coefficient = value(&node.state.routes[metadata.tag as usize])?;
            if coefficient != 0. {
                node.master.set_coefficient(physical, column, coefficient)?;
                metadata.entries.push((physical, coefficient));
            }
        }
    }
    for metadata in &mut node.warm.generated_pool {
        if metadata.tag < 0 {
            continue;
        }
        let coefficient = value(&node.state.routes[metadata.tag as usize])?;
        if coefficient != 0. {
            metadata.entries.push((physical, coefficient));
        }
    }
    node.state.rhs[physical] = rhs;
    node.master.set_rhs(&node.state.rhs)?;
    Ok(node)
}
