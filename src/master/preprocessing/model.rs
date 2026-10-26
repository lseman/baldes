//! Whole-domain flow propagation and complete transferred-master deductions.
use super::*;
use crate::master::preprocessing::{propagate_mixed_conservative, Bounds, Constraint, Report};

#[derive(Clone, Debug)]
pub struct PreprocessingOptions {
    pub enabled: bool,
    /// Arithmetic work across the flow and, if available, complete-master pass.
    pub work_limit: usize,
    /// Strengthen the original integer domain, rather than preserve the LP.
    pub integer_domains: bool,
}
impl Default for PreprocessingOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            work_limit: 1_000_000,
            integer_domains: false,
        }
    }
}

/// Bounds on remaining aggregate arc usage and route count in one network.
/// Arc order follows the original graph. These are not per-route windows.
#[derive(Clone, Debug)]
pub struct NetworkActivityBounds {
    pub network: i32,
    pub arcs: Vec<Bounds>,
    pub multiplicity: Bounds,
}

#[derive(Clone, Debug, Default)]
pub struct PreprocessingReport {
    /// False means missing future columns prevented whole-master deductions.
    pub complete_domain: bool,
    pub propagation: Report,
    pub tightened_columns: usize,
    pub fixed_columns: usize,
    pub forbidden_arcs: usize,
    pub skipped: Option<String>,
    pub networks: Vec<NetworkActivityBounds>,
    pub master_bounds: Vec<(i32, Bounds)>,
    pub route_upper_bounds: Vec<(usize, f64)>,
}

pub(super) fn apply<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    loaded: &[(usize, usize)],
    options: &PreprocessingOptions,
) -> Result<()> {
    if !options.enabled {
        state.preprocessing = PreprocessingReport {
            skipped: Some("Disabled".into()),
            ..Default::default()
        };
        return Ok(());
    }
    let flow_work = if state.universe.is_some() {
        options.work_limit / 2
    } else {
        options.work_limit
    };
    let mut report =
        super::flow_adapter::apply(master, state, loaded, flow_work, options.integer_domains)?;
    if report.propagation.infeasible || report.skipped.is_some() {
        state.preprocessing = report;
        return Ok(());
    }
    if state.universe.is_none() {
        state.preprocessing = report;
        return Ok(());
    }
    report.complete_domain = true;
    let present: HashSet<_> = loaded.iter().map(|&(_, tag)| tag).collect();
    if !state
        .universe
        .as_ref()
        .unwrap()
        .iter()
        .all(|t| present.contains(t))
    {
        report.skipped = Some("UnloadedDomainColumns".into());
        state.preprocessing = report;
        return Ok(());
    }
    if !bounds::exact_fixed_reduction(state, master)? {
        report.skipped = Some("RoundedFixedReduction".into());
        state.preprocessing = report;
        return Ok(());
    }
    let mut rows: Vec<_> = master
        .senses
        .iter()
        .zip(&master.rhs)
        .map(|(&sense, &rhs)| Constraint {
            entries: vec![],
            sense,
            rhs,
        })
        .collect();
    let mut bounds = vec![
        Bounds {
            lower: 0.,
            upper: 0.
        };
        master.costs.len()
    ];
    // Static columns are loaded first and never selected by route cleanup.
    for (index, variable) in state.master_variables.iter().enumerate() {
        for (side, sign) in [(0usize, 1.), (1, -1.)] {
            let column = 2 * index + side;
            let semantic = super::master_variables::upper(variable, sign);
            // A finite backend infinity sentinel is not an original bound.
            bounds[column].upper = if semantic.is_finite() {
                semantic.min(master.upper[column])
            } else {
                f64::INFINITY
            };
            for &(r, a) in &variable.entries {
                rows[state.shared_offset + r]
                    .entries
                    .push((column, sign * a));
            }
        }
    }
    // Artificial variables are absent from the original formulation. Retained
    // non-domain columns have upper zero after atomic enumeration transfer.
    for &(column, tag) in loaded {
        if column >= bounds.len() || tag >= state.routes.len() {
            return Err(invalid("preprocessing column registry"));
        }
        bounds[column].upper = super::route_bounds::upper(state, tag)
            .ok_or_else(|| invalid("preprocessing route domain"))?
            .min(if master.upper[column] == f64::MAX {
                f64::INFINITY
            } else {
                master.upper[column]
            });
        let route = &state.routes[tag];
        let oracle = state
            .oracles
            .iter()
            .find(|o| o.network.id == route.network)
            .unwrap();
        let coefficients = entries(oracle, &route.route, state.shared_offset)?;
        // Propagation removes columns, so rounded coefficient aggregation must
        // not stand in for the exact original activity. Decline this pass if a
        // matrix entry lost a residual, including a sum rounded to zero.
        let mut exact = vec![crate::master::numerics::ExactSum::default(); rows.len()];
        let mut valid = true;
        for &(row, value) in &oracle.route_entries {
            valid &= exact[row].add(value).is_some();
        }
        for &(row, value) in oracle.network.route_entries.iter() {
            valid &= exact[state.shared_offset + row].add(value).is_some();
        }
        for id in &route.route.arcs {
            for &(row, value) in &oracle.network.arc_entries[oracle.index[id]] {
                valid &= exact[state.shared_offset + row].add(value).is_some();
            }
        }
        for (row, value) in super::nonlinear::entries(oracle, &route.route)? {
            valid &= exact[state.shared_offset + row].add(value).is_some();
        }
        for &(row, value) in &coefficients {
            valid &= exact[row].add(-value).is_some();
        }
        if !valid || exact.iter().any(|sum| sum.sign() != 0.) {
            report.skipped = Some("RoundedColumnCoefficients".into());
            state.preprocessing = report;
            return Ok(());
        }
        for (row, value) in coefficients {
            rows[row].entries.push((column, value));
        }
    }
    let budget = options.work_limit - flow_work;
    let mut integers = vec![false; bounds.len()];
    if options.integer_domains {
        for (index, v) in state.master_variables.iter().enumerate() {
            integers[2 * index] = v.integer;
            integers[2 * index + 1] = v.integer;
        }
        for &(column, _) in loaded {
            integers[column] = true;
        }
    }
    let pass = propagate_mixed_conservative(&mut bounds, &rows, &integers, budget)?;
    report.propagation.bound_updates += pass.bound_updates;
    report.propagation.rows_processed += pass.rows_processed;
    report.propagation.work_used += pass.work_used;
    report.propagation.complete &= pass.complete;
    report.propagation.infeasible |= pass.infeasible;
    if !report.propagation.infeasible {
        for &(column, tag) in loaded {
            let upper = bounds[column]
                .upper
                .min(super::enumeration::route_upper(state, tag).unwrap());
            state.route_upper_bounds.insert(tag, upper);
            report.route_upper_bounds.push((tag, upper));
            if upper < master.upper[column] {
                master.set_upper(column, upper)?;
                report.tightened_columns += 1;
                report.fixed_columns += usize::from(upper == 0.);
            }
        }
    }
    if !report.propagation.infeasible {
        let disabled: HashSet<_> = loaded
            .iter()
            .filter(|&&(column, _)| bounds[column].upper == 0.)
            .map(|&(_, tag)| tag)
            .collect();
        state
            .universe
            .as_mut()
            .unwrap()
            .retain(|tag| !disabled.contains(tag));
    }
    state.preprocessing = report;
    Ok(())
}

impl<const D: usize> NodeSnapshot<D> {
    pub fn preprocessing_report(&self) -> &PreprocessingReport {
        &self.state.preprocessing
    }

    /// Fork and propagate whole-domain flows and, when complete enumeration
    /// is available, nonlinear master activities. Restricted columns alone
    /// never justify a deduction about missing future routes.
    pub fn preprocess(&self) -> Result<Self> {
        self.preprocess_with_options(&PreprocessingOptions::default())
    }

    pub fn preprocess_with_options(&self, options: &PreprocessingOptions) -> Result<Self> {
        let mut node = self.fork()?;
        let loaded: Vec<_> = node
            .warm
            .columns
            .iter()
            .enumerate()
            .filter_map(|(c, m)| {
                m.as_ref()
                    .filter(|m| m.tag >= 0)
                    .map(|m| (c, m.tag as usize))
            })
            .collect();
        let previous_bounds: Vec<_> = node
            .state
            .master_variables
            .iter()
            .map(|v| v.bounds)
            .collect();
        apply(&mut node.master, &mut node.state, &loaded, options)?;
        super::master_variables::publish::bounds(
            &mut node.master,
            &mut node.state,
            &previous_bounds,
            &mut vec![],
            Some(&mut node.warm),
        )?;
        Ok(node)
    }
}
