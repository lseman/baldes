//! Optional Clp root master adapter for routing instances.
use super::*;
impl Instance {
    #[cfg(feature = "clp")]
    pub fn capacity_cut_config(
        &self,
        rounding: Rounding,
        two_path: bool,
    ) -> crate::master::root::CapacityCutConfig {
        use crate::cuts::capacity::TimeWindows;
        let config = crate::master::root::CapacityCutConfig::new(
            self.customers.iter().map(|c| c.demand).collect(),
            self.capacity,
        );
        if !two_path {
            return config;
        }
        config.with_time_windows(TimeWindows {
            ready: self.customers.iter().map(|c| c.ready).collect(),
            due: self.customers.iter().map(|c| c.due).collect(),
            service: self.customers.iter().map(|c| c.service).collect(),
            travel: (0..self.customers.len())
                .map(|a| {
                    (0..self.customers.len())
                        .map(|b| self.distance(a, b, rounding))
                        .collect()
                })
                .collect(),
        })
    }
    #[cfg(feature = "clp")]
    pub fn solve_root(
        &self,
        ng_size: usize,
        rounding: Rounding,
        mut options: crate::master::root::RootOptions,
    ) -> Result<crate::master::root::RootOutput> {
        let graph = self.graph(ng_size, rounding)?;
        options.cg.multiplicity = (self.customers.len() - 1) as f64;
        let problem = crate::master::root::RootProblem::set_partitioning(
            graph,
            (self.customers.len() - 1) as f64,
        )?;
        crate::master::root::solve(problem, options)
    }
}
