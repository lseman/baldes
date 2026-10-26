//! Integer aggregate flows over weighted physical arcs: model-registered
//! generic variables and undirected edges. A generic variable here is a fixed
//! integer combination of arc traversals, so every integral routing solution
//! gives it an integral value and floor/ceil children partition the space.
use crate::{invalid, master::routing::RoutingContext, Result};
use std::collections::BTreeMap;

/// A model-registered aggregate branching variable: a name and integer
/// coefficients on physical arc IDs. Repeated arcs are merged.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateVariable {
    name: String,
    arcs: Vec<(i32, f64)>,
}
impl AggregateVariable {
    pub fn new(name: impl Into<String>, arcs: &[(i32, f64)]) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(invalid("aggregate branching variable needs a model name"));
        }
        let mut merged = BTreeMap::<i32, crate::master::numerics::ExactSum>::new();
        for &(arc, coefficient) in arcs {
            merged
                .entry(arc)
                .or_default()
                .add(coefficient)
                .ok_or_else(|| invalid("aggregate coefficient sum overflow"))?;
        }
        let mut coefficients = Vec::with_capacity(merged.len());
        for (arc, mut sum) in merged {
            let coefficient = sum
                .upper()
                .ok_or_else(|| invalid("aggregate coefficient sum overflow"))?;
            sum.add(-coefficient)
                .ok_or_else(|| invalid("aggregate coefficient sum overflow"))?;
            if sum.sign() != 0. {
                return Err(invalid(
                    "aggregate coefficient sum must be exactly representable",
                ));
            }
            if coefficient != 0. {
                coefficients.push((arc, coefficient));
            }
        }
        let arcs = coefficients;
        validate_coefficients(&arcs)?;
        Ok(Self { name, arcs })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Arc IDs and coefficients, sorted by arc ID.
    pub fn arcs(&self) -> &[(i32, f64)] {
        &self.arcs
    }
    /// The same variable over physical arc positions, or None when an arc ID
    /// is absent from this problem.
    pub(crate) fn positions<const D: usize>(
        &self,
        problem: &RoutingContext<D>,
    ) -> Option<Vec<(usize, f64)>> {
        let mut arcs = self
            .arcs
            .iter()
            .map(|&(id, coefficient)| Some((problem.arc_position(id)?, coefficient)))
            .collect::<Option<Vec<_>>>()?;
        arcs.sort_by_key(|&(arc, _)| arc);
        Some(arcs)
    }
}

fn validate_coefficients<T>(arcs: &[(T, f64)]) -> Result<()> {
    if arcs.is_empty() {
        return Err(invalid("aggregate branching variable needs a nonzero arc"));
    }
    // Integral coefficients keep the aggregate integral at integer routes;
    // the magnitude bound keeps every route activity exact in binary64.
    if arcs
        .iter()
        .any(|&(_, c)| !c.is_finite() || c.fract() != 0. || c.abs() > 1_048_576.)
    {
        return Err(invalid(
            "aggregate branching coefficients must be integers of magnitude at most 2^20",
        ));
    }
    Ok(())
}

/// `arcs` are physical arc positions sorted by position.
pub(super) fn validate<const D: usize>(
    problem: &RoutingContext<D>,
    arcs: &[(usize, f64)],
) -> Result<()> {
    validate_coefficients(arcs)?;
    if arcs.windows(2).any(|w| w[0].0 >= w[1].0) {
        return Err(invalid("aggregate branch arcs must be sorted and distinct"));
    }
    if arcs
        .last()
        .is_some_and(|&(arc, _)| arc >= problem.model.networks[0].graph.arcs.len())
    {
        return Err(invalid("aggregate branch arc index"));
    }
    Ok(())
}

pub(super) fn coefficient(arcs: &[(usize, f64)], arc: usize) -> f64 {
    arcs.binary_search_by_key(&arc, |&(a, _)| a)
        .map_or(0., |i| arcs[i].1)
}

/// Every undirected edge {u, v}, u < v, with its arcs in both directions.
/// Loops have no edge. Parallel arcs share their edge.
pub(crate) fn edges<const D: usize>(
    problem: &RoutingContext<D>,
) -> Vec<(String, Vec<(usize, f64)>)> {
    let mut edges = BTreeMap::<(usize, usize), Vec<(usize, f64)>>::new();
    for (position, arc) in problem.model.networks[0].graph.arcs.iter().enumerate() {
        if arc.tail != arc.head {
            let key = (arc.tail.min(arc.head), arc.tail.max(arc.head));
            edges.entry(key).or_default().push((position, 1.));
        }
    }
    edges
        .into_iter()
        .map(|((u, v), arcs)| (format!("edge:{u}-{v}"), arcs))
        .collect()
}
