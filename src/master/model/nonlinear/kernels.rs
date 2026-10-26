//! Original descriptors retained while imported column domains grow.
use super::*;

#[derive(Clone, Debug)]
pub(in crate::master::model) enum RowKernels {
    Presence(Vec<PresenceCoefficients>),
    Registered {
        coefficients: Vec<RegisteredCutCoefficients>,
        composite: bool,
    },
    Resource(Vec<ResourceKnapsackCoefficients>),
}

impl RowKernels {
    /// Validate against original graphs without solving a restricted master.
    pub(in crate::master::model) fn validate<const D: usize>(
        &self,
        problem: &Problem<D>,
        row: &Row,
    ) -> Result<()> {
        let network = |id| {
            problem
                .networks
                .iter()
                .find(|n| n.id == id)
                .ok_or_else(|| invalid("unknown nonlinear routing subproblem"))
        };
        let mut seen = HashSet::new();
        match self {
            Self::Presence(coefficients) => {
                use crate::cuts::soft_conflicts::{PresencePricing, PricingInput};
                for c in coefficients {
                    let n = network(c.network)?;
                    if !seen.insert(c.network) {
                        return Err(invalid("duplicate presence routing subproblem"));
                    }
                    PresencePricing::new(
                        n.graph.arcs.len(),
                        vec![PricingInput {
                            row: c.row.clone(),
                            arc_events: c.arc_events.clone(),
                            dual: 0.,
                        }],
                    )?
                    .validate_engine(n.graph.arcs.len())?;
                }
            }
            Self::Registered {
                coefficients,
                composite,
            } => {
                for c in coefficients {
                    let n = network(c.network)?;
                    if !seen.insert(c.network) && !composite {
                        return Err(invalid("duplicate nonlinear routing subproblem"));
                    }
                    // Pricing assigns the subproblem identity to its graph.
                    let mut graph = n.graph.clone();
                    graph.graph_id = n.id;
                    registered::normalize(&c.cut, &graph)?;
                }
            }
            Self::Resource(coefficients) => {
                if row.sense != RowSense::LessEqual {
                    return Err(invalid("resource knapsack row requires <= sense"));
                }
                for c in coefficients {
                    let n = network(c.network)?;
                    if !seen.insert(c.network) {
                        return Err(invalid("duplicate resource knapsack routing subproblem"));
                    }
                    resource_rows::validate(&n.graph, c)?;
                }
            }
        }
        Ok(())
    }

    /// Install before fixed-route RHS shifts, seeds and preprocessing.
    pub(in crate::master::model) fn install<const D: usize>(
        &self,
        state: &mut State<D>,
        shared: usize,
    ) -> Result<()> {
        match self {
            Self::Presence(c) => presence_rows::install(state, shared, c),
            Self::Registered {
                coefficients,
                composite,
            } => install_registered(state, shared, coefficients, *composite),
            Self::Resource(c) => resource_rows::install(state, shared, c),
        }
    }
}
