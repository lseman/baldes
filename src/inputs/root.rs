//! Optional Clp root master adapter for routing instances.
use super::*;
impl Instance {
    /// Global integer-load packing row obtained by summing customer-cover
    /// equalities. Demands are gcd-scaled; the pricing load is coordinate 1.
    /// Fenchel-only separation is opt in; no rounding/1K cuts are enabled here.
    pub fn fenchel_cut_config(&self) -> Result<crate::master::root::KnapsackCutConfig> {
        use crate::cuts::knapsack::{FenchelParameters, KnapsackRow};
        let mut divisor = 0_i64;
        let mut demands = Vec::new();
        for customer in self.customers.iter().skip(1) {
            let q = customer.demand;
            if !q.is_finite() || q < 0.0 || q.fract() != 0.0 || q > f64::from(i32::MAX / 8) {
                return Err(invalid(
                    "Fenchel routing cuts require nonnegative integer demands",
                ));
            }
            let q = q as i64;
            let (mut a, mut b) = (divisor, q);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            divisor = a;
            demands.push(q);
        }
        let divisor = divisor.max(1);
        let total = demands
            .iter()
            .try_fold(0_i64, |sum, &q| sum.checked_add(q / divisor))
            .ok_or_else(|| invalid("Fenchel normalized demand sum overflow"))?;
        let limits = FenchelParameters::default();
        if total > limits.max_capacity as i64 {
            return Err(invalid("Fenchel normalized total demand exceeds the default capacity limit; use explicit Rust resource rows and limits"));
        }
        let mut config = crate::master::root::KnapsackCutConfig::new(vec![KnapsackRow::new(
            0,
            1,
            1.0 / divisor as f64,
            total as i32,
        )])
        .with_fenchel(limits);
        config.parameters.max_one_k = 0;
        config.parameters.use_rounding = false;
        Ok(config)
    }

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
