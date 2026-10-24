//! Budgeted-uncertainty load and service-start vectors.
//!
//! Entry γ is the worst case when at most γ quantities deviate. Forward
//! labels follow Munari et al. (2019), equations (4.7)-(4.8). Backward load
//! uses the same recursion over the suffix. Backward time keeps the latest
//! start at the label's vertex that keeps the suffix feasible for at most γ
//! suffix deviations, excluding that vertex's own window:
//!
//!   B_γ(tail) = min(min(l, B_γ) - d, min(l, B_{γ-1}) - d - d̂),
//!
//! feasible while e <= min(l, B_Γ). Forward entries are nondecreasing and
//! backward time entries nonincreasing in γ, so the Γ entry is binding.
//! A realization splits its deviations into a prefix share a and a suffix
//! share b <= Γ - a, so joins test every split exactly.
use super::*;

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Load { capacity: f64 },
    Time { initial: f64 },
}
#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedBudgeted {
    kind: Kind,
    /// First state entry of this resource; it owns `budget + 1` entries.
    offset: usize,
    budget: usize,
    /// Per arc: nominal quantity, deviation, and head window (time only).
    steps: Vec<[f64; 2]>,
    windows: Vec<Window>,
    /// Arcs leaving the source. Backward labels can complete there without
    /// a join, so their latest departures are checked against the start.
    from_source: std::sync::Arc<[bool]>,
}

impl PreparedBudgeted {
    pub(super) fn prepare<const D: usize>(
        g: &Graph<D>,
        definitions: &[CoupledResource],
    ) -> Vec<Self> {
        let from_source: std::sync::Arc<[bool]> =
            g.arcs.iter().map(|a| a.tail == g.source).collect();
        let mut offset = 0;
        definitions
            .iter()
            .filter_map(|r| {
                let (kind, budget, steps, windows) = match r {
                    CoupledResource::BudgetedLoad {
                        capacity,
                        budget,
                        demands,
                        deviations,
                    } => (
                        Kind::Load {
                            capacity: *capacity,
                        },
                        *budget,
                        demands
                            .iter()
                            .zip(deviations)
                            .map(|(&q, &d)| [q, d])
                            .collect(),
                        vec![],
                    ),
                    CoupledResource::BudgetedTime {
                        initial,
                        budget,
                        durations,
                        deviations,
                        windows,
                    } => (
                        Kind::Time { initial: *initial },
                        *budget,
                        durations
                            .iter()
                            .zip(deviations)
                            .map(|(&t, &d)| [t, d])
                            .collect(),
                        windows.clone(),
                    ),
                    _ => return None,
                };
                let prepared = Self {
                    kind,
                    offset,
                    budget,
                    steps,
                    windows,
                    from_source: from_source.clone(),
                };
                offset += budget + 1;
                Some(prepared)
            })
            .collect()
    }
    pub(super) fn width(prepared: &[Self]) -> usize {
        prepared.last().map_or(0, |p| p.offset + p.budget + 1)
    }
    fn slice<'a>(&self, values: &'a [f64]) -> &'a [f64] {
        &values[self.offset..=self.offset + self.budget]
    }
    pub(super) fn root(prepared: &[Self], forward: bool) -> Vec<f64> {
        let mut values = Vec::with_capacity(Self::width(prepared));
        for p in prepared {
            let value = match p.kind {
                Kind::Load { .. } => 0.,
                Kind::Time { initial } if forward => initial,
                Kind::Time { .. } => f64::INFINITY,
            };
            values.extend(std::iter::repeat(value).take(p.budget + 1));
        }
        values
    }
    /// Extend in place; false when some admissible realization is infeasible.
    pub(super) fn extend(&self, values: &mut [f64], arc: usize, forward: bool) -> Result<bool> {
        let [step, deviation] = self.steps[arc];
        let v = &mut values[self.offset..=self.offset + self.budget];
        match self.kind {
            Kind::Load { capacity } => {
                for g in (1..v.len()).rev() {
                    v[g] = finite(v[g] + step)?.max(finite(v[g - 1] + step + deviation)?);
                }
                v[0] = finite(v[0] + step)?;
                Ok(v[self.budget] <= capacity)
            }
            Kind::Time { .. } if forward => {
                let window = self.windows[arc];
                for g in (1..v.len()).rev() {
                    v[g] = finite(v[g] + step)?
                        .max(finite(v[g - 1] + step + deviation)?)
                        .max(window.lower);
                }
                v[0] = finite(v[0] + step)?.max(window.lower);
                Ok(v[self.budget] <= window.upper)
            }
            Kind::Time { initial } => {
                let window = self.windows[arc];
                for x in v.iter_mut() {
                    *x = x.min(window.upper);
                }
                if v[self.budget] < window.lower {
                    return Ok(false);
                }
                for g in (1..v.len()).rev() {
                    v[g] = finite(v[g] - step)?.min(finite(v[g - 1] - step - deviation)?);
                }
                v[0] = finite(v[0] - step)?;
                Ok(!self.from_source[arc] || initial <= v[self.budget])
            }
        }
    }
    /// Componentwise order preserved by every later extension and join.
    pub(super) fn dominates(&self, a: &[f64], b: &[f64], forward: bool) -> bool {
        let (a, b) = (self.slice(a), self.slice(b));
        match self.kind {
            Kind::Time { .. } if !forward => a.iter().zip(b).all(|(a, b)| a >= b),
            _ => a.iter().zip(b).all(|(a, b)| a <= b),
        }
    }
    /// Join a forward prefix and a backward suffix at one vertex.
    pub(super) fn join(&self, f: &[f64], b: &[f64]) -> Result<bool> {
        let (f, b) = (self.slice(f), self.slice(b));
        let gamma = self.budget;
        Ok(match self.kind {
            Kind::Load { capacity } => {
                for a in 0..=gamma {
                    if finite(f[a] + b[gamma - a])? > capacity {
                        return Ok(false);
                    }
                }
                true
            }
            Kind::Time { .. } => (0..=gamma).all(|a| f[a] <= b[gamma - a]),
        })
    }
}

impl CoupledKernel {
    pub(super) fn budgets_extend(
        &self,
        values: &mut [f64],
        arc: usize,
        forward: bool,
    ) -> Result<bool> {
        for p in &self.budgeted {
            if !p.extend(values, arc, forward)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(super) fn budgets_dominate(&self, a: &[f64], b: &[f64], forward: bool) -> bool {
        a.len() == b.len() && self.budgeted.iter().all(|p| p.dominates(a, b, forward))
    }
    pub(super) fn budgets_join(
        &self,
        f: &CoupledLabelState,
        b: &CoupledLabelState,
    ) -> Result<bool> {
        for p in &self.budgeted {
            if !p.join(&f.budgets, &b.budgets)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
