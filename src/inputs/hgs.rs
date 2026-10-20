//! HGS route collection and validated restricted-master seeding.
use super::{Instance, Rounding};
use crate::{
    hgs::driver::{self, HgsOptions},
    invalid,
    master::root::{RootOptions, RootProblem},
    Result,
};
use std::{collections::HashSet, time::Duration};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HgsColumns {
    /// Routes from the best feasible solution found by HGS.
    #[default]
    Best,
    /// All distinct encountered routes that pass target-graph feasibility checks.
    All,
    /// Up to N cheapest distinct feasible routes, ranked by original arc cost.
    Top(usize),
}

impl HgsColumns {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "best" => Ok(Self::Best),
            "all" => Ok(Self::All),
            _ => match value.parse::<usize>() {
                Ok(n) if n > 0 => Ok(Self::Top(n)),
                _ => Err(invalid(
                    "HGS columns must be 'best', 'all', or a positive integer",
                )),
            },
        }
    }
}

fn select_new_columns(
    candidates: impl IntoIterator<Item = crate::Route>,
    existing: &[crate::Route],
    selection: HgsColumns,
) -> Vec<crate::Route> {
    let mut seen: HashSet<_> = existing.iter().map(|r| r.arcs.clone()).collect();
    let mut routes: Vec<_> = candidates
        .into_iter()
        .filter(|route| seen.insert(route.arcs.clone()))
        .collect();
    if let HgsColumns::Top(n) = selection {
        routes.sort_by(|a, b| {
            a.original_cost
                .total_cmp(&b.original_cost)
                .then(a.arcs.cmp(&b.arcs))
        });
        routes.truncate(n);
    }
    routes
}

#[derive(Clone, Debug)]
pub struct HgsWarmStart {
    /// Positive HGS search budget, separate from the subsequent solver budget.
    pub time_limit: Duration,
    pub columns: HgsColumns,
    pub seed: i32,
}

impl Default for HgsWarmStart {
    fn default() -> Self {
        Self {
            time_limit: Duration::from_secs(10),
            columns: HgsColumns::Best,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HgsSeedSummary {
    pub inserted_columns: usize,
    pub rejected_columns: usize,
    pub incumbent: Option<f64>,
    pub elapsed_secs: f64,
}

impl Instance {
    /// Run HGS before root/tree solving and append unique, resource-feasible
    /// initial columns. A complete validated best solution also supplies an
    /// incumbent. `problem` must use this instance's split-depot vertex IDs.
    /// HGS uses the selected distance metric; the target graph validates every route.
    pub fn seed_hgs<const D: usize>(
        &self,
        problem: &RootProblem<D>,
        rounding: Rounding,
        options: &mut RootOptions,
        warm: &HgsWarmStart,
    ) -> Result<HgsSeedSummary> {
        let graph = &problem.graph;
        let n = self.customers.len().saturating_sub(1);
        if warm.time_limit.is_zero()
            || n == 0
            || graph.vertices.len() != n + 2
            || graph.source != 0
            || graph.sink != n + 1
        {
            return Err(invalid(
                "HGS seeding requires a positive budget and the instance's split-depot graph",
            ));
        }
        if warm.columns == HgsColumns::Top(0) {
            return Err(invalid("HGS column count must be positive"));
        }
        graph.validate()?;
        let mut instance = self.clone();
        instance.rounding = rounding;
        // Capacity graphs have zero time consumption, irrespective of the
        // original file's time windows. Give HGS the same problem semantics.
        if graph.arcs.iter().all(|a| a.consumption[0] == 0.) {
            for customer in &mut instance.customers {
                customer.ready = 0.;
                customer.due = f64::INFINITY;
                customer.service = 0.;
            }
        }
        let controls = HgsOptions {
            time_limit: warm.time_limit.as_secs_f64(),
            seed: warm.seed,
            verbose: options.cg.log_interval.is_some(),
            ..Default::default()
        };
        let found = if warm.columns != HgsColumns::Best {
            driver::run_hgs_collecting_routes(&instance, &controls)
        } else {
            driver::run_hgs(&instance, &controls)
        }
        .map_err(invalid)?;
        let arcs: std::collections::HashMap<_, _> = graph
            .arcs
            .iter()
            .map(|a| ((a.tail, a.head), a.id))
            .collect();
        let convert = |customers: &[usize]| {
            let nodes: Vec<_> = std::iter::once(graph.source)
                .chain(customers.iter().copied())
                .chain(std::iter::once(graph.sink))
                .collect();
            let ids = nodes
                .windows(2)
                .map(|w| arcs.get(&(w[0], w[1])).copied())
                .collect::<Option<Vec<_>>>()?;
            let route = problem.elementary_route(&ids)?;
            crate::master::root::exact_route_feasible(problem, &route)?;
            Some(route)
        };
        let best = found
            .routes
            .iter()
            .map(|r| convert(r))
            .collect::<Option<Vec<_>>>();
        let incumbent = best
            .as_ref()
            .filter(|r| !r.is_empty())
            .and_then(|r| problem.unit_route_incumbent(r));
        if let Some(upper) = incumbent {
            options.incumbent = options.incumbent.min(upper);
        }
        let candidates = if warm.columns != HgsColumns::Best {
            &found.columns
        } else {
            &found.routes
        };
        let mut rejected_columns = 0;
        let feasible = candidates.iter().filter_map(|route| {
            let converted = convert(route);
            if converted.is_none() {
                rejected_columns += 1;
            }
            converted
        });
        let selected = select_new_columns(feasible, &options.initial_routes, warm.columns);
        let inserted_columns = selected.len();
        options.initial_routes.extend(selected);
        Ok(HgsSeedSummary {
            inserted_columns,
            rejected_columns,
            incumbent,
            elapsed_secs: found.elapsed_secs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn route(id: i32, cost: f64) -> crate::Route {
        crate::Route {
            arcs: vec![id],
            nodes: vec![],
            cost,
            original_cost: cost,
            resources: vec![],
            resource_profile: vec![],
        }
    }
    #[test]
    fn numeric_selection_ranks_unique_new_columns_and_handles_short_pools() {
        let pool = vec![
            route(5, 9.),
            route(3, 2.),
            route(1, 1.),
            route(2, 2.),
            route(2, 2.),
        ];
        let existing = vec![route(1, 1.)];
        let top = select_new_columns(pool.clone(), &existing, HgsColumns::Top(2));
        assert_eq!(
            top.iter().map(|r| r.arcs[0]).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(
            select_new_columns(pool.clone(), &existing, HgsColumns::Top(100)).len(),
            3
        );
        assert_eq!(
            select_new_columns(pool, &existing, HgsColumns::All)[0].arcs,
            vec![5]
        );
        assert!(select_new_columns(vec![], &existing, HgsColumns::Top(1)).is_empty());
        assert_eq!(HgsColumns::parse("7").unwrap(), HgsColumns::Top(7));
        for invalid in ["0", "-1", "1.5", "wrong", "184467440737095516160"] {
            assert!(HgsColumns::parse(invalid).is_err());
        }
    }
}
