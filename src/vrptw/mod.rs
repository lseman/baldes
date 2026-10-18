//! Solomon and CVRPLIB (TSPLIB format) instance input and the split-depot
//! VRPTW model.
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
    /// Depot first. CVRPLIB customers have no time window: due is infinite.
    pub customers: Vec<Customer>,
    /// The file's distance convention: none for Solomon, TSPLIB nint for
    /// EUC_2D, ceil for CEIL_2D. Callers may choose another.
    pub rounding: Rounding,
    /// Explicit distances (CVRPLIB EDGE_WEIGHT_SECTION), in customer order;
    /// they replace the coordinates and ignore the rounding.
    pub distances: Option<Vec<Vec<f64>>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    None,
    /// floor(d + 0.5): TSPLIB nint, the CVRPLIB EUC_2D convention.
    Nearest,
    /// Truncated to tenths. Graph times use HGS's integer metric (see
    /// [`Rounding::scale`]); costs stay in instance units.
    Tenth,
    /// TSPLIB CEIL_2D.
    Ceil,
}
impl Rounding {
    /// Factor applied to graph times: 10 for `Tenth`, so travel times, time
    /// windows and service times are integers, as in HGS. Only the time
    /// resource is scaled: pricing and stabilization tolerances are absolute,
    /// and scaled costs slow column generation down several times.
    pub fn scale(self) -> f64 {
        if self == Self::Tenth {
            10.
        } else {
            1.
        }
    }

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Self::None),
            "nearest" => Ok(Self::Nearest),
            "tenth" => Ok(Self::Tenth),
            "ceil" => Ok(Self::Ceil),
            _ => Err(invalid("rounding must be none, nearest, tenth or ceil")),
        }
    }
}
impl Instance {
    /// Whether every customer has a finite time window (CVRPLIB has none).
    pub fn has_time_windows(&self) -> bool {
        self.customers.iter().all(|c| c.due.is_finite())
    }
    pub fn truncate(&mut self, n: usize) -> Result<()> {
        if n == 0 || n >= self.customers.len() {
            return Err(invalid("customer limit out of range"));
        }
        self.customers.truncate(n + 1);
        if let Some(matrix) = &mut self.distances {
            matrix.truncate(n + 1);
            for row in matrix {
                row.truncate(n + 1);
            }
        }
        Ok(())
    }
    pub fn distance(&self, i: usize, j: usize, rounding: Rounding) -> f64 {
        if let Some(matrix) = &self.distances {
            return matrix[i][j];
        }
        let a = &self.customers[i];
        let b = &self.customers[j];
        let d = ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).sqrt();
        match rounding {
            Rounding::None => d,
            Rounding::Nearest => (d + 0.5).floor(),
            Rounding::Tenth => (d * 10.).floor() / 10.,
            Rounding::Ceil => d.ceil(),
        }
    }
}

mod graph;
mod parsing;
#[cfg(feature = "clp")]
mod root;
