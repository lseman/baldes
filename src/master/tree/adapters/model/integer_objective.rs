//! Validation for rounding node bounds up (native `integerValuedBound`).
use super::*;

fn integral(value: f64) -> bool {
    value.is_finite() && value.fract() == 0.
}

fn integral_graph<const D: usize>(graph: &Graph<D>) -> bool {
    graph.arcs.iter().all(|a| integral(a.cost))
        && !graph.coupled_kernel().is_some_and(|k| k.has_costs())
}

/// Every integer solution has an integer objective only if every arc cost,
/// static master-variable cost and fixed objective contribution is an
/// integer, with no resource-dependent or original-variable cost terms. A
/// continuous master variable can take fractional values, so its cost must
/// be zero.
pub(super) fn validate<const D: usize>(start: &Start<D>, options: &ModelTreeOptions) -> Result<()> {
    let variable_ok = |v: &MasterVariable| {
        if v.integer {
            integral(v.cost)
        } else {
            v.cost == 0.
        }
    };
    let valid = options.node.original_variable_costs.is_empty()
        && match start {
            Start::Fresh(problem, variables) => {
                problem.networks.iter().all(|n| integral_graph(&n.graph))
                    && variables.iter().all(variable_ok)
            }
            Start::Snapshot(node) => {
                node.state
                    .oracles
                    .iter()
                    .all(|o| integral_graph(o.network.graph.as_ref()))
                    && node.state.master_variables.iter().all(variable_ok)
                    && integral(node.master.objective_offset)
            }
        };
    if valid {
        Ok(())
    } else {
        Err(invalid(
            "integer objective bounds require integer arc, master and fixed costs",
        ))
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/master/tree/tree_controls_tests.rs"]
mod tests;
