//! Solomon instance input and the split-depot no-cut VRPTW model.
use crate::{invalid, Graph, Result};
#[derive(Clone, Debug)]
pub struct Customer {
    pub x: f64,
    pub y: f64,
    pub demand: f64,
    pub ready: f64,
    pub due: f64,
    pub service: f64,
}
#[derive(Clone, Debug)]
pub struct Instance {
    pub name: String,
    pub vehicles: usize,
    pub capacity: f64,
    pub customers: Vec<Customer>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    None,
    Nearest,
    Tenth,
}
impl Instance {
    pub fn parse(text: &str) -> Result<Self> {
        let lines: Vec<_> = text
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let name = lines
            .first()
            .ok_or_else(|| invalid("empty Solomon instance"))?
            .to_string();
        let vehicle = lines
            .iter()
            .position(|s| s.to_ascii_uppercase().starts_with("NUMBER"))
            .and_then(|i| lines.get(i + 1))
            .ok_or_else(|| invalid("Solomon vehicle table is missing"))?;
        let numbers = vehicle
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| invalid("invalid Solomon vehicle row"))?;
        if numbers.len() < 2
            || numbers[0] < 1.
            || numbers[0].fract() != 0.
            || numbers[1] <= 0.
            || !numbers[1].is_finite()
        {
            return Err(invalid("invalid Solomon vehicle count/capacity"));
        }
        let start = lines
            .iter()
            .position(|s| s.to_ascii_uppercase().starts_with("CUST NO"))
            .ok_or_else(|| invalid("Solomon customer table is missing"))?
            + 1;
        let mut customers = vec![];
        for row in &lines[start..] {
            let values = row
                .split_whitespace()
                .map(str::parse::<f64>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| invalid("invalid Solomon customer row"))?;
            if values.len() < 7 {
                continue;
            }
            if values.iter().any(|v| !v.is_finite())
                || values[0] != customers.len() as f64
                || values[3] < 0.
                || values[4] > values[5]
                || values[6] < 0.
            {
                return Err(invalid(
                    "invalid Solomon customer identity or resource data",
                ));
            }
            customers.push(Customer {
                x: values[1],
                y: values[2],
                demand: values[3],
                ready: values[4],
                due: values[5],
                service: values[6],
            });
        }
        if customers.len() < 2 {
            return Err(invalid("Solomon instance needs a depot and customers"));
        }
        Ok(Self {
            name,
            vehicles: numbers[0] as usize,
            capacity: numbers[1],
            customers,
        })
    }
    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).map_err(|e| invalid(format!("Solomon input: {e}")))?;
        Self::parse(&text)
    }
    pub fn truncate(&mut self, n: usize) -> Result<()> {
        if n == 0 || n >= self.customers.len() {
            return Err(invalid("customer limit out of range"));
        }
        self.customers.truncate(n + 1);
        Ok(())
    }
    pub fn distance(&self, i: usize, j: usize, rounding: Rounding) -> f64 {
        let a = &self.customers[i];
        let b = &self.customers[j];
        let d = ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).sqrt();
        match rounding {
            Rounding::None => d,
            Rounding::Nearest => (d + 0.5).floor(),
            Rounding::Tenth => (d * 10.).floor() / 10.,
        }
    }
    pub fn graph(&self, ng_size: usize, rounding: Rounding) -> Result<Graph> {
        self.build_graph(ng_size, rounding, true)
    }
    /// Capacity-only routing: time windows and service times impose no constraints.
    pub fn capacity_graph(&self, ng_size: usize, rounding: Rounding) -> Result<Graph> {
        self.build_graph(ng_size, rounding, false)
    }
    fn build_graph(&self, ng_size: usize, rounding: Rounding, time_windows: bool) -> Result<Graph> {
        let n = self.customers.len() - 1;
        let sink = n + 1;
        let mut graph = Graph::new(n + 2, 2)?;
        let mut add = |u: usize, v: usize| -> Result<()> {
            let i = if u == sink { 0 } else { u };
            let j = if v == sink { 0 } else { v };
            let from = &self.customers[i];
            let to = &self.customers[j];
            let d = self.distance(i, j, rounding);
            let travel = if time_windows { from.service + d } else { 0. };
            if (time_windows && from.ready + travel > to.due + 1e-6)
                || (u != 0 && v != sink && from.demand + to.demand > self.capacity + 1e-6)
            {
                return Ok(());
            }
            let id = graph.add_arc(u, v, d)?;
            graph.arc_mut(id)?.consumption = [travel, if v == sink { 0. } else { to.demand }];
            Ok(())
        };
        for j in 1..=n {
            add(0, j)?;
            add(j, sink)?;
            for k in 1..=n {
                if j != k {
                    add(j, k)?;
                }
            }
        }
        for v in 0..=sink {
            let i = if v == sink { 0 } else { v };
            let (ready, due) = if time_windows {
                (self.customers[i].ready, self.customers[i].due)
            } else {
                (0., 1.)
            };
            graph.set_bounds(v, 0, ready, due)?;
            graph.set_bounds(v, 1, 0., self.capacity)?;
            let mut order: Vec<_> = (0..=sink).collect();
            order.sort_by(|&a, &b| {
                let ai = if a == sink { 0 } else { a };
                let bi = if b == sink { 0 } else { b };
                self.distance(i, ai, rounding)
                    .total_cmp(&self.distance(i, bi, rounding))
                    .then(a.cmp(&b))
            });
            order.truncate(ng_size.min(n + 2));
            graph.set_ng(v, &order)?;
        }
        graph.initial[0] = Some(if time_windows {
            self.customers[0].ready
        } else {
            0.
        });
        Ok(graph)
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
