//! Curvature radius updates with explicit state and measured feedback.
use super::{ProximalBand, ProximalParameters};
use crate::{invalid, Result};

/// Radius state is distinct from a half-width. Both radii and curvature must
/// come from the caller's initialized/inherited stabilization state.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct CurvatureState {
    pub outer_radius: f64,
    /// Present for a five-piece penalty, absent for a three-piece penalty.
    pub inner_radius: Option<f64>,
    pub curvature: f64,
}

/// Measured mean absolute distance from the accepted center, over active
/// ordinary rows. Dynamic stabilization-only rows do not belong in this mean.
#[derive(Clone, Copy, Debug)]
pub struct CurvatureFeedback {
    pub mean_dual_distance: Option<f64>,
    pub advance_rate: f64,
}

/// Automatic radius schedule. Initial values may come from model setup or an
/// inherited stabilization snapshot; no auxiliary LP objective initializes it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct AutomaticCurvatureParameters {
    #[cfg_attr(
        feature = "clp",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub initial: Option<CurvatureState>,
    #[cfg_attr(feature = "clp", serde(default))]
    pub five_piece: bool,
    #[cfg_attr(feature = "clp", serde(default = "default_one"))]
    pub kappa: f64,
    /// Used instead of automatic bootstrap when kappa is nonpositive.
    #[cfg_attr(feature = "clp", serde(default = "default_one"))]
    pub curvature: f64,
    #[cfg_attr(feature = "clp", serde(default = "default_one"))]
    pub advance_rate: f64,
    #[cfg_attr(feature = "clp", serde(default = "default_ten"))]
    pub artificial_activity_factor: f64,
}

fn default_one() -> f64 {
    1.
}
fn default_ten() -> f64 {
    10.
}

impl Default for AutomaticCurvatureParameters {
    fn default() -> Self {
        Self {
            initial: None,
            five_piece: false,
            kappa: default_one(),
            curvature: default_one(),
            advance_rate: default_one(),
            artificial_activity_factor: default_ten(),
        }
    }
}

impl AutomaticCurvatureParameters {
    pub(crate) fn validate(&self) -> Result<()> {
        if let Some(initial) = self.initial {
            initial.validate()?;
            if initial.inner_radius.is_some() != self.five_piece {
                return Err(invalid(
                    "curvature initial state and function type disagree",
                ));
            }
        }
        if !self.kappa.is_finite()
            || !(self.kappa as f32).is_finite()
            || (self.kappa > 0. && self.kappa as f32 <= 0.)
            || !self.curvature.is_finite()
            || self.curvature <= 0.
            || !(self.curvature as f32).is_finite()
            || self.curvature as f32 <= 0.
            || !self.artificial_activity_factor.is_finite()
            || self.artificial_activity_factor <= 0.
            || !(self.artificial_activity_factor as f32).is_finite()
            || self.artificial_activity_factor as f32 <= 0.
        {
            return Err(invalid("invalid curvature kappa"));
        }
        if !self.advance_rate.is_finite() || !(0. ..=1.).contains(&self.advance_rate) {
            return Err(invalid("invalid curvature advance rate"));
        }
        Ok(())
    }

    pub(crate) fn initial_state(&self) -> CurvatureState {
        self.initial.unwrap_or(CurvatureState {
            outer_radius: 1e12,
            inner_radius: self.five_piece.then_some(1e12),
            curvature: if self.kappa > 0. {
                1e12
            } else {
                self.curvature as f32 as f64
            },
        })
    }

    pub(crate) fn update(&self, state: &mut CurvatureState, mean: Option<f64>) -> Result<bool> {
        if let Some(mean) = mean {
            let sentinel_margin = state.curvature.abs().max(1e12) * 1e-6 + 1e-6;
            if (state.curvature - 1e12).abs() <= sentinel_margin && mean > mean.abs() * 1e-10 + 1e-6
            {
                let curvature = mean / (self.kappa as f32 as f64);
                if !curvature.is_finite() || curvature <= 0. {
                    return Err(invalid("automatic curvature initialization overflow"));
                }
                state.outer_radius = mean;
                state.inner_radius = self.five_piece.then_some(mean);
                state.curvature = curvature;
            }
        }
        state.update(CurvatureFeedback {
            mean_dual_distance: mean,
            advance_rate: self.advance_rate,
        })
    }
}

impl CurvatureState {
    pub(crate) fn validate(&self) -> Result<()> {
        if !self.outer_radius.is_finite()
            || self.outer_radius < 0.
            || !self.curvature.is_finite()
            || self.curvature <= 0.
            || self.inner_radius.is_some_and(|r| !r.is_finite() || r < 0.)
        {
            return Err(invalid("invalid curvature stabilization state"));
        }
        Ok(())
    }
    /// Returns false without changing state when the required observation is
    /// unavailable. The advance rate is rounded to its stored float precision.
    /// This primitive does not infer active rows, initialize curvature, or
    /// decide when the surrounding stabilization schedule should call it.
    pub fn update(&mut self, feedback: CurvatureFeedback) -> Result<bool> {
        let Some(mean) = feedback.mean_dual_distance else {
            return Ok(false);
        };
        if !mean.is_finite()
            || mean < 0.
            || !feedback.advance_rate.is_finite()
            || !(0. ..=1.).contains(&feedback.advance_rate)
            || !self.outer_radius.is_finite()
            || self.outer_radius < 0.
            || !self.curvature.is_finite()
            || self.curvature <= 0.
            || self.inner_radius.is_some_and(|r| !r.is_finite() || r < 0.)
        {
            return Err(invalid("invalid curvature stabilization state or feedback"));
        }
        let rate = feedback.advance_rate as f32 as f64;
        let mut outer = self.outer_radius;
        let mut inner = self.inner_radius;
        if let Some(ref mut inner) = inner {
            if mean < outer - margin(mean, outer) {
                *inner -= rate * (*inner - outer);
                outer -= rate * (outer - mean);
            } else if *inner - margin(*inner, mean) <= mean {
                outer += rate * (*inner - outer);
                *inner += (mean - *inner) * rate;
            } else {
                outer += rate * (mean - outer);
            }
        } else {
            outer = (mean - outer) * rate + outer;
        }
        if !outer.is_finite() || inner.is_some_and(|r| !r.is_finite()) {
            return Err(invalid("curvature radius update overflow"));
        }
        self.outer_radius = outer;
        self.inner_radius = inner;
        Ok(true)
    }

    /// Convert active radii into their explicit penalty bands. The surrounding
    /// solver can use these on a private LP; this produces no objective bound.
    /// A radius at the native finite infinity sentinel disables that band.
    pub fn bands(&self) -> Result<Option<ProximalParameters>> {
        if !self.curvature.is_finite() || self.curvature <= 0. {
            return Err(invalid("invalid proximal curvature"));
        }
        let mut bands = Vec::new();
        for radius in [self.inner_radius, Some(self.outer_radius)]
            .into_iter()
            .flatten()
        {
            if !radius.is_finite() || radius < 0. {
                return Err(invalid("invalid proximal radius"));
            }
            if radius >= 1e12 - margin(radius, 1e12) || radius == 0. {
                continue;
            }
            let penalty = radius / self.curvature;
            if !penalty.is_finite() || penalty <= 0. {
                return Err(invalid("invalid curvature-derived penalty"));
            }
            bands.push(ProximalBand {
                half_width: radius * 0.5,
                penalty,
            });
        }
        // The stored inner radius need not be smaller than the stored outer
        // radius. Additive symmetric penalties are independent of order.
        bands.sort_by(|a, b| a.half_width.total_cmp(&b.half_width));
        // Equal-width bands retain separate artificial variables: merging
        // their caps would change the per-variable activity threshold.
        if bands.is_empty() {
            return Ok(None);
        }
        let parameters = ProximalParameters {
            bands,
            multi_point: false,
            max_points: 1,
            curvature: None,
            child_half_interval_factor: 10.,
            explicit: None,
            rhs_row_scaling: false,
            artificial_activity_factor: 10.,
        };
        parameters.validate()?;
        Ok(Some(parameters))
    }
}

fn margin(a: f64, b: f64) -> f64 {
    a.abs().max(b.abs()) * 1e-10 + 1e-6
}

/// Compute required feedback only when every active ordinary row has valid
/// center and query coordinates. Missing coordinates do not become zeroes.
pub fn mean_dual_distance(center: &[f64], current: &[f64], rows: &[usize]) -> Option<f64> {
    if rows.is_empty() {
        return Some(0.);
    }
    let mut total = 0.;
    for &row in rows {
        let a = *center.get(row)?;
        let b = *current.get(row)?;
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        total += (a - b).abs();
    }
    let mean = total / rows.len() as f64;
    mean.is_finite().then_some(mean)
}

#[cfg(test)]
#[path = "../../../tests/unit/master/stabilization/curvature/tests.rs"]
mod tests;
