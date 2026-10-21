//! Electric routing foundations shared by future direct and subpath pricing.
//!
//! Continuous charging, elementary pricing and route-based column generation.
//! Quantities use caller-selected, consistent time/energy/load/cost units.

use crate::{invalid, Result};

mod direct;
mod fixed_order;
mod functions;
mod master;
mod subpath;
pub use direct::{
    DirectOptions, EvPricingResult, EvPricingStatistics, EvPricingStatus, PricedRoute,
};
pub use fixed_order::{ChargingOptions, ChargingSolution, ChargingStatus, FixedOrderResult};
pub use master::{
    EvColumn, EvColumnGenerationOptions, EvColumnGenerationResult, EvColumnGenerationStatus,
    EvIteration, EvMaster, EvMasterPhase, EvMasterSnapshot, EvPricingBackend,
};
pub use subpath::SubpathOptions;

/// Cumulative time to charge from empty, represented by (energy, time) knots.
/// Strict increase in both coordinates makes partial charging invertible.
#[derive(Clone, Debug)]
pub struct ChargingCurve {
    knots: Vec<(f64, f64)>,
}

fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

impl ChargingCurve {
    pub fn new(knots: Vec<(f64, f64)>) -> Result<Self> {
        if knots.len() < 2
            || knots[0] != (0.0, 0.0)
            || knots
                .iter()
                .any(|&(e, t)| !nonnegative(e) || !nonnegative(t))
            || knots
                .windows(2)
                .any(|w| w[0].0 >= w[1].0 || w[0].1 >= w[1].1)
        {
            return Err(invalid(
                "charging curve needs finite, strictly increasing knots from (0, 0)",
            ));
        }
        Ok(Self { knots })
    }

    pub fn capacity(&self) -> f64 {
        self.knots[self.knots.len() - 1].0
    }

    fn interpolate(&self, x: f64, inverse: bool) -> Result<f64> {
        let coordinate = |p: (f64, f64)| if inverse { (p.1, p.0) } else { p };
        let last = coordinate(self.knots[self.knots.len() - 1]);
        if !nonnegative(x) || x > last.0 {
            return Err(invalid("charging curve argument outside domain"));
        }
        for w in self.knots.windows(2) {
            let (a, b) = (coordinate(w[0]), coordinate(w[1]));
            if x <= b.0 {
                return Ok(a.1 + (x - a.0) / (b.0 - a.0) * (b.1 - a.1));
            }
        }
        Ok(last.1)
    }

    pub fn cumulative_time(&self, energy: f64) -> Result<f64> {
        self.interpolate(energy, false)
    }

    pub fn duration(&self, arrival: f64, departure: f64) -> Result<f64> {
        if departure < arrival {
            return Err(invalid("charging cannot remove energy"));
        }
        Ok(self.cumulative_time(departure)? - self.cumulative_time(arrival)?)
    }

    /// Charge for at most `duration`, saturating at the curve's capacity.
    pub fn charge_for(&self, arrival: f64, duration: f64) -> Result<f64> {
        if !nonnegative(duration) {
            return Err(invalid("invalid charging duration"));
        }
        let time = self.cumulative_time(arrival)?;
        let full = self.knots[self.knots.len() - 1].1;
        self.interpolate((time + duration).min(full), true)
    }
}

#[derive(Clone, Debug)]
pub enum LocationKind {
    Depot,
    Customer {
        demand: f64,
    },
    /// Each station carries its vehicle-specific charging technology curve.
    Station {
        curve: ChargingCurve,
    },
}

#[derive(Clone, Debug)]
pub struct Location {
    pub kind: LocationKind,
    /// Customer service must start within this inclusive window. At stations,
    /// service and charging must also finish by closing time.
    pub window: (f64, f64),
    pub service: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Travel {
    pub time: f64,
    pub energy: f64,
    pub cost: f64,
}

/// Single vehicle type; station capacity, tariffs and regeneration unsupported.
#[derive(Clone, Debug)]
pub struct Problem {
    pub locations: Vec<Location>,
    /// Directed row-major matrix; None denotes an unavailable arc.
    pub travel: Vec<Vec<Option<Travel>>>,
    pub depot: usize,
    pub battery: f64,
    pub load_capacity: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Visit {
    pub location: usize,
    /// Explicit amount added, rather than target SoC. Zero at non-stations.
    pub recharge: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct StopState {
    pub arrival: f64,
    pub start: f64,
    pub departure: f64,
    pub arrival_energy: f64,
    pub departure_energy: f64,
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub stops: Vec<StopState>,
    pub travel_cost: f64,
    pub delivered: f64,
}

impl Problem {
    pub fn validate(&self) -> Result<()> {
        let n = self.locations.len();
        if self.depot >= n
            || !matches!(self.locations[self.depot].kind, LocationKind::Depot)
            || !nonnegative(self.battery)
            || self.battery == 0.0
            || !nonnegative(self.load_capacity)
            || self.travel.len() != n
            || self.travel.iter().any(|r| r.len() != n)
        {
            return Err(invalid(
                "invalid EV dimensions, depot or vehicle capacities",
            ));
        }
        for location in &self.locations {
            let (open, close) = location.window;
            if !nonnegative(open)
                || !nonnegative(close)
                || open > close
                || !nonnegative(location.service)
            {
                return Err(invalid("invalid EV time window or service duration"));
            }
            match &location.kind {
                LocationKind::Customer { demand } if !nonnegative(*demand) => {
                    return Err(invalid("invalid EV customer demand"))
                }
                LocationKind::Station { curve } if curve.capacity() != self.battery => {
                    return Err(invalid(
                        "station curve must cover exactly the vehicle battery",
                    ))
                }
                _ => {}
            }
        }
        if self
            .travel
            .iter()
            .flatten()
            .flatten()
            .any(|a| !nonnegative(a.time) || !nonnegative(a.energy) || !nonnegative(a.cost))
        {
            return Err(invalid("invalid EV travel data"));
        }
        Ok(())
    }

    /// Independently replay a closed route, initially fully charged. Customers
    /// may occur at most once; stations may repeat. Partial customer coverage is
    /// allowed so individual master columns can be checked. Comparisons are
    /// strict floating-point comparisons: no hidden feasibility tolerance.
    pub fn replay(&self, visits: &[Visit], departure: f64) -> Result<Schedule> {
        self.validate()?;
        if visits.len() < 2
            || visits[0].location != self.depot
            || visits[visits.len() - 1].location != self.depot
            || !nonnegative(departure)
        {
            return Err(invalid("EV route must start and end at its depot"));
        }
        let mut schedule = Schedule {
            stops: Vec::new(),
            travel_cost: 0.0,
            delivered: 0.0,
        };
        let mut seen = vec![false; self.locations.len()];
        let (mut time, mut energy) = (departure, self.battery);
        for (i, visit) in visits.iter().enumerate() {
            let location = self
                .locations
                .get(visit.location)
                .ok_or_else(|| invalid("unknown EV location"))?;
            if !nonnegative(visit.recharge) {
                return Err(invalid("invalid recharge amount"));
            }
            if i > 0 {
                let arc = self.travel[visits[i - 1].location][visit.location]
                    .ok_or_else(|| invalid("unavailable EV travel arc"))?;
                if energy < arc.energy {
                    return Err(invalid("EV battery exhausted"));
                }
                energy -= arc.energy;
                time += arc.time;
                schedule.travel_cost += arc.cost;
            }
            let arrival = time;
            let arrival_energy = energy;
            let start = time.max(location.window.0);
            if !start.is_finite() || start > location.window.1 {
                return Err(invalid("EV time window missed"));
            }
            time = start + location.service;
            match &location.kind {
                LocationKind::Customer { demand } => {
                    if seen[visit.location] {
                        return Err(invalid("EV customer repeated"));
                    }
                    seen[visit.location] = true;
                    schedule.delivered += demand;
                    if schedule.delivered > self.load_capacity {
                        return Err(invalid("EV load capacity exceeded"));
                    }
                }
                LocationKind::Station { curve } => {
                    let target = energy + visit.recharge;
                    time += curve.duration(energy, target)?;
                    energy = target;
                    if time > location.window.1 {
                        return Err(invalid("charging station closed"));
                    }
                }
                LocationKind::Depot if i != 0 && i != visits.len() - 1 => {
                    return Err(invalid("intermediate EV depot unsupported"))
                }
                LocationKind::Depot => {}
            }
            if !matches!(location.kind, LocationKind::Station { .. }) && visit.recharge != 0.0 {
                return Err(invalid("recharge at non-station"));
            }
            if !time.is_finite()
                || !schedule.travel_cost.is_finite()
                || !schedule.delivered.is_finite()
            {
                return Err(invalid("EV schedule arithmetic overflow"));
            }
            schedule.stops.push(StopState {
                arrival,
                start,
                departure: time,
                arrival_energy,
                departure_energy: energy,
            });
        }
        Ok(schedule)
    }
}

#[cfg(all(test, feature = "clp"))]
mod master_tests;
#[cfg(test)]
mod tests;
