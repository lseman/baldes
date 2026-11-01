//! Explicit conjunction-defined whole-column cardinality restrictions.
use super::*;
use crate::cuts::soft_conflicts::PresenceRow;
pub use crate::cuts::soft_conflicts::{ComponentBound, ComponentSense, ComponentSequence};
mod generator;
pub use generator::{
    ComponentClassGenerator, ComponentGeneratorCoordinate, ComponentPriorityRule,
    ComponentVariableKind,
};

pub(super) fn generated_proposals<const D: usize>(
    output: &Output<D>,
    generators: &[ComponentClassGenerator],
    quota: usize,
    tolerance: f64,
    deadline: Option<Instant>,
) -> Result<Vec<model_strong::Proposal>> {
    generator::collect(output, generators, quota, tolerance, deadline)
}

/// Sparse integer coordinate on original physical arcs. `None` omits the
/// coordinate; `Some(0)` records a present coordinate with value zero.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComponentCoordinate {
    pub id: usize,
    pub arc_values: Vec<Option<i64>>,
}

/// Explicitly registered class. Priority and name enter the ordinary branch
/// dispatcher; generation of native class-tree candidates is a separate policy.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComponentClassBranching {
    pub name: String,
    pub priority: f64,
    /// Native class-weight admission threshold (BapCodCutViolationTolerance).
    #[serde(default = "default_minimum_fractionality")]
    pub minimum_fractionality: f64,
    pub network: i32,
    pub coordinates: Vec<ComponentCoordinate>,
    pub sequence: ComponentSequence,
}

fn default_minimum_fractionality() -> f64 {
    0.02
}

impl ComponentClassBranching {
    pub(super) fn coefficients<const D: usize>(
        &self,
        node: &NodeSnapshot<D>,
    ) -> Result<PresenceCoefficients> {
        let oracle = node
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == self.network)
            .ok_or_else(|| invalid("component class network"))?;
        self.for_graph(&oracle.network.graph)
    }

    pub(crate) fn for_graph<const D: usize>(
        &self,
        graph: &Graph<D>,
    ) -> Result<PresenceCoefficients> {
        if self.name.is_empty()
            || !self.priority.is_finite()
            || !self.minimum_fractionality.is_finite()
            || self.minimum_fractionality < 0.
            || self.minimum_fractionality > 0.5
        {
            return Err(invalid("component class name or priority"));
        }
        if graph.source == graph.sink {
            return Err(invalid(
                "component classes require distinct pricing terminals",
            ));
        }
        let mut indices = BTreeMap::new();
        for (index, c) in self.coordinates.iter().enumerate() {
            if c.arc_values.len() != graph.arcs.len() || indices.insert(c.id, index).is_some() {
                return Err(invalid("component coordinate mapping"));
            }
        }
        let mut sequence = self.sequence.clone();
        for bound in &mut sequence.bounds {
            bound.coordinate = *indices
                .get(&bound.coordinate)
                .ok_or_else(|| invalid("unregistered component coordinate"))?;
        }
        let mut emissions = vec![];
        let mut arc_events = vec![vec![]; graph.arcs.len()];
        for (coordinate, c) in self.coordinates.iter().enumerate() {
            for (arc, value) in c.arc_values.iter().enumerate() {
                if let Some(value) = value {
                    arc_events[arc].push(emissions.len());
                    emissions.push((coordinate, *value));
                }
            }
        }
        let terminal = emissions.len();
        for (events, arc) in arc_events.iter_mut().zip(&graph.arcs) {
            if arc.head == graph.sink {
                events.push(terminal);
            }
        }
        let row = PresenceRow::component_class(self.coordinates.len(), sequence, emissions)?;
        Ok(PresenceCoefficients {
            network: self.network,
            row,
            arc_events,
        })
    }

    pub(super) fn proposal<const D: usize>(
        &self,
        output: &Output<D>,
        tolerance: f64,
    ) -> Result<Option<model_strong::Proposal>> {
        let c = self.coefficients(&output.snapshot)?;
        let oracle = output
            .snapshot
            .state
            .oracles
            .iter()
            .find(|o| o.network.id == self.network)
            .unwrap();
        let pricing = crate::cuts::soft_conflicts::PresencePricing::new(
            c.arc_events.len(),
            vec![crate::cuts::soft_conflicts::PricingInput {
                row: c.row,
                arc_events: c.arc_events,
                dual: 0.,
            }],
        )?;
        let mut value = 0.;
        for (route, quantity) in output
            .routes
            .iter()
            .zip(&output.values)
            .map(|(r, &q)| (r, q))
            .chain(output.fixed_routes.iter().map(|r| (&r.route, r.quantity)))
        {
            if route.network != self.network {
                continue;
            }
            let arcs = route
                .route
                .arcs
                .iter()
                .map(|id| {
                    oracle
                        .index
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("component route arc"))
                })
                .collect::<Result<Vec<_>>>()?;
            value += quantity * pricing.route_coefficients(&arcs)?[0] as f64;
        }
        let fraction = (value - value.round()).abs();
        let threshold = self.minimum_fractionality;
        if fraction <= tolerance
            || !(threshold < fraction - (fraction.abs().max(threshold.abs()) * 1e-10 + 1e-6))
        {
            return Ok(None);
        }
        let key = format!(
            "component:{}:{}",
            self.network,
            serde_json::to_string(self).map_err(|_| invalid("component identity"))?
        );
        Ok(Some(model_strong::Proposal {
            name: self.name.clone(),
            history_key: key,
            priority: self.priority,
            value,
            restriction: model_strong::Restriction::Component(self.clone()),
            history_eligible: true,
            fresh: true,
        }))
    }
}

impl<const D: usize> NodeSnapshot<D> {
    /// Count complete columns satisfying every component predicate. Coefficients
    /// apply identically to future pricing columns, fixed routes and MIP replay.
    pub fn with_component_bounds(
        &self,
        class: &ComponentClassBranching,
        bounds: crate::master::preprocessing::Bounds,
    ) -> Result<Self> {
        if bounds.lower.is_nan()
            || bounds.upper.is_nan()
            || bounds.lower == f64::INFINITY
            || bounds.upper == f64::NEG_INFINITY
            || bounds.lower > bounds.upper
        {
            return Err(invalid("component class bounds"));
        }
        let coefficients = class.coefficients(self)?;
        let mut node = self.fork()?;
        if bounds.lower.is_finite() {
            node = node.with_presence_row(
                Row {
                    sense: RowSense::GreaterEqual,
                    rhs: bounds.lower,
                },
                std::slice::from_ref(&coefficients),
            )?;
        }
        if bounds.upper.is_finite() {
            node = node.with_presence_row(
                Row {
                    sense: RowSense::LessEqual,
                    rhs: bounds.upper,
                },
                std::slice::from_ref(&coefficients),
            )?;
        }
        for oracle in &mut node.state.oracles {
            oracle.solver.enumeration_preserve_arc_paths = true;
        }
        if bounds.lower.is_finite() || bounds.upper.is_finite() {
            node.state.component_branches.push(class.clone());
        }
        Ok(node)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/component_tests.rs"]
mod tests;
