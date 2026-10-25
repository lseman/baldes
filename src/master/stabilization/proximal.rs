//! Explicit piecewise-linear proximal pricing on a private restricted master.
use crate::master::{LpMaster, LpSolveOutcome};
use crate::{invalid, Result};
use std::time::Duration;

/// A symmetric band around a dual center. A finite activity cap gives a
/// linear penalty outside the band; an infinite cap makes the band hard.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct ProximalBand {
    pub half_width: f64,
    #[cfg_attr(feature = "clp", serde(with = "penalty_json"))]
    pub penalty: f64,
}

#[cfg(feature = "clp")]
mod penalty_json {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
        if *value == f64::INFINITY {
            serializer.serialize_none()
        } else {
            serializer.serialize_f64(*value)
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
        Ok(Option::<f64>::deserialize(deserializer)?.unwrap_or(f64::INFINITY))
    }
}

/// Explicit box-step (one hard band), three-piece (one soft band), or
/// five-piece (two soft bands) dual regularization, in unscaled row units.
/// This replaces smoothing when selected. Optional curvature derives bands
/// automatically; optional multipoint retains outer-band support planes.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct ProximalParameters {
    #[cfg_attr(feature = "clp", serde(default))]
    pub bands: Vec<ProximalBand>,
    /// Derive bands from the automatic radius schedule instead of explicit
    /// bands. Completed pricing observations drive this schedule.
    #[cfg_attr(
        feature = "clp",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub curvature: Option<super::curvature::AutomaticCurvatureParameters>,
    /// Use retained pricing points as support planes for the outer band.
    #[cfg_attr(
        feature = "clp",
        serde(default, skip_serializing_if = "std::ops::Not::not")
    )]
    pub multi_point: bool,
    /// Maximum retained support planes; the native public Kappa default is 1.
    #[cfg_attr(
        feature = "clp",
        serde(
            default = "default_max_points",
            skip_serializing_if = "is_default_max_points"
        )
    )]
    pub max_points: usize,
}

fn default_max_points() -> usize {
    1
}
#[cfg(feature = "clp")]
fn is_default_max_points(value: &usize) -> bool {
    *value == default_max_points()
}

impl ProximalParameters {
    pub fn automatic_three_piece() -> Self {
        Self::automatic(super::curvature::AutomaticCurvatureParameters::default())
    }

    pub fn automatic_five_piece() -> Self {
        Self::automatic(super::curvature::AutomaticCurvatureParameters {
            five_piece: true,
            ..Default::default()
        })
    }

    pub fn automatic(curvature: super::curvature::AutomaticCurvatureParameters) -> Self {
        Self {
            bands: vec![],
            curvature: Some(curvature),
            multi_point: false,
            max_points: default_max_points(),
        }
    }

    pub fn box_step(half_width: f64) -> Self {
        Self {
            curvature: None,
            multi_point: false,
            max_points: default_max_points(),
            bands: vec![ProximalBand {
                half_width,
                penalty: f64::INFINITY,
            }],
        }
    }

    pub fn three_piece(half_width: f64, penalty: f64) -> Self {
        Self {
            curvature: None,
            multi_point: false,
            max_points: default_max_points(),
            bands: vec![ProximalBand {
                half_width,
                penalty,
            }],
        }
    }

    pub fn five_piece(inner: ProximalBand, outer: ProximalBand) -> Self {
        Self {
            curvature: None,
            multi_point: false,
            max_points: default_max_points(),
            bands: vec![inner, outer],
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if let Some(curvature) = &self.curvature {
            curvature.validate()?;
            if !self.bands.is_empty() || self.max_points == 0 || self.multi_point {
                return Err(invalid("automatic curvature requires no explicit bands"));
            }
            return Ok(());
        }
        if self.max_points == 0
            || self.bands.is_empty()
            || self.bands.len() > 2
            || self.bands.iter().any(|b| {
                !b.half_width.is_finite()
                    || b.half_width < 0.
                    || b.penalty.is_nan()
                    || b.penalty <= 0.
            })
            || self
                .bands
                .windows(2)
                .any(|b| b[0].half_width > b[1].half_width || !b[0].penalty.is_finite())
            || (self.multi_point && !self.bands.last().is_some_and(|b| b.penalty.is_finite()))
        {
            return Err(invalid("invalid explicit proximal bands"));
        }
        Ok(())
    }
}

/// Accepted dual center and automatic curvature inherited by a compatible
/// child formulation. Auxiliary columns and support planes are not retained.
#[derive(Clone, Debug)]
pub struct ProximalSnapshot {
    pub center: Vec<f64>,
    pub bound: f64,
    pub phase: usize,
    pub curvature: Option<super::curvature::CurvatureState>,
    pub configuration: Option<super::curvature::AutomaticCurvatureParameters>,
}

#[derive(Default)]
pub(crate) struct ProximalState {
    center: Vec<f64>,
    bound: f64,
    phase: Option<usize>,
    samples: Vec<super::multipoint::SupportPlane>,
    max_points: usize,
    next_point_id: u64,
    curvature: Option<super::curvature::CurvatureState>,
    configuration: Option<super::curvature::AutomaticCurvatureParameters>,
    last_mean: Option<f64>,
    pending_iteration: bool,
    artificial_values: Option<Vec<f64>>,
}

impl ProximalState {
    /// A CG schedule/ordinary artificial-penalty restart does not erase the
    /// automatic rule's accepted geometry. Unfinished observations are dropped.
    pub fn restart_schedule(&mut self) {
        if self.configuration.is_some() {
            self.pending_iteration = false;
            self.artificial_values = None;
        } else {
            self.reset();
        }
    }
    /// A reset invalidates the center after row/domain/objective changes.
    pub fn reset(&mut self) {
        self.center.clear();
        self.phase = None;
        self.bound = f64::NEG_INFINITY;
        self.samples.clear();
        self.next_point_id = 0;
        self.curvature = None;
        self.configuration = None;
        self.last_mean = None;
        self.pending_iteration = false;
        self.artificial_values = None;
    }

    pub fn snapshot(&self) -> Option<ProximalSnapshot> {
        Some(ProximalSnapshot {
            center: self.center.clone(),
            bound: self.bound,
            phase: self.phase?,
            curvature: self.curvature,
            configuration: self.configuration.clone(),
        })
    }

    /// The caller must first establish compatible row order and formulation.
    /// Inherited centers are retained, but inherited bounds are not promoted
    /// to bounds for a changed child domain.
    pub fn inherit(&mut self, snapshot: &ProximalSnapshot) -> Result<()> {
        if snapshot.center.iter().any(|x| !x.is_finite())
            || snapshot.bound.is_nan()
            || snapshot.bound == f64::INFINITY
        {
            return Err(invalid("invalid inherited proximal center"));
        }
        if let Some(curvature) = snapshot.curvature {
            curvature.validate()?;
        }
        if let Some(configuration) = &snapshot.configuration {
            configuration.validate()?;
            if snapshot
                .curvature
                .is_some_and(|state| state.inner_radius.is_some() != configuration.five_piece)
            {
                return Err(invalid("inherited curvature function type disagrees"));
            }
        }
        self.reset();
        self.center.clone_from(&snapshot.center);
        self.phase = Some(snapshot.phase);
        self.curvature = snapshot.curvature;
        self.configuration.clone_from(&snapshot.configuration);
        Ok(())
    }

    /// A resumed solve may number its pricing stages differently. Only this
    /// explicit boundary operation remaps the inherited phase identity.
    pub fn inherit_for_solve(&mut self, snapshot: &ProximalSnapshot, phase: usize) -> Result<()> {
        self.inherit(snapshot)?;
        self.phase = Some(phase);
        Ok(())
    }

    /// Latch the original LP observation once, before private pricing or a
    /// serious-step center update. Retries within an iteration share it.
    pub fn begin_iteration(
        &mut self,
        raw: &[f64],
        ordinary_rows: &[usize],
        phase: usize,
        parameters: &ProximalParameters,
    ) -> Result<()> {
        parameters.validate()?;
        if self.center.len() != raw.len()
            || self.phase.is_none()
            || (self.phase != Some(phase) && parameters.curvature.is_none())
        {
            self.reset();
            self.center = if parameters.curvature.is_some() {
                vec![0.; raw.len()]
            } else {
                raw.to_vec()
            };
            self.phase = Some(phase);
        }
        self.phase = Some(phase);
        if let Some(configuration) = &parameters.curvature {
            if self.configuration.as_ref() != Some(configuration) {
                self.curvature = Some(configuration.initial_state());
                self.configuration = Some(configuration.clone());
                self.pending_iteration = false;
                self.last_mean = None;
            }
            if !self.pending_iteration {
                self.last_mean =
                    super::curvature::mean_dual_distance(&self.center, raw, ordinary_rows);
                self.pending_iteration = true;
            }
        }
        Ok(())
    }

    /// Complete one latched iteration. Aborted pricing must not invoke this.
    pub fn after_iteration(&mut self, parameters: &ProximalParameters) -> Result<bool> {
        if !self.pending_iteration {
            return Ok(false);
        }
        self.pending_iteration = false;
        let Some(configuration) = &parameters.curvature else {
            return Ok(false);
        };
        let Some(state) = &mut self.curvature else {
            return Ok(false);
        };
        configuration.update(state, self.last_mean)
    }

    /// Maximum absolute activity of each band in the last successful private
    /// solve, ordered from inner to outer. Failed/skipped solves return None.
    pub fn take_artificial_values(&mut self) -> Option<Vec<f64>> {
        self.artificial_values.take()
    }

    pub fn observe(&mut self, dual: &[f64], bound: f64, exact: bool) {
        // Serious steps require a completed pricing pass. An auxiliary LP
        // objective is never a bound for the original column universe.
        if exact && bound.is_finite() && bound > self.bound {
            self.center = dual.to_vec();
            self.bound = bound;
            if self.max_points > 0 {
                for sample in &mut self.samples {
                    sample.protected_center = false;
                }
                self.register_point(dual, true);
                // All activity metadata is already validated at observation.
                let _ = super::multipoint::retain_planes(&mut self.samples, self.max_points);
            }
        }
    }

    /// Active stabilization artificials in a completed private solve request
    /// another round after increasing curvature. This does not certify a bound.
    pub fn artificial_activity(
        &mut self,
        active: bool,
        parameters: &ProximalParameters,
    ) -> Result<bool> {
        let Some(configuration) = parameters.curvature.as_ref().filter(|_| active) else {
            return Ok(false);
        };
        let Some(state) = &mut self.curvature else {
            return Ok(false);
        };
        let curvature = state.curvature * (configuration.artificial_activity_factor as f32 as f64);
        if !curvature.is_finite() || curvature <= 0. {
            return Err(invalid("curvature artificial-activity update overflow"));
        }
        let mut next = *state;
        next.curvature = curvature;
        next.update(super::curvature::CurvatureFeedback {
            mean_dual_distance: self.last_mean,
            advance_rate: configuration.advance_rate,
        })?;
        *state = next;
        Ok(true)
    }

    fn register_point(&mut self, point: &[f64], protected: bool) {
        if let Some(sample) = self.samples.iter_mut().find(|s| s.point == point) {
            sample.protected_center |= protected;
        } else {
            self.samples.push(super::multipoint::SupportPlane {
                point: point.to_vec(),
                dual_activity: 0.,
                zero_age: 0,
                protected_center: protected,
                creation_id: self.next_point_id,
            });
            self.next_point_id = self.next_point_id.saturating_add(1);
        }
    }

    pub fn pricing_point<M: LpMaster>(
        &mut self,
        master: &M,
        raw: &[f64],
        phase: usize,
        parameters: &ProximalParameters,
        excluded: &[usize],
        remaining: Duration,
    ) -> Result<Vec<f64>> {
        self.artificial_values = None;
        if self.center.len() != raw.len()
            || self.phase.is_none()
            || (self.phase != Some(phase) && parameters.curvature.is_none())
        {
            self.reset();
            self.center = if parameters.curvature.is_some() {
                vec![0.; raw.len()]
            } else {
                raw.to_vec()
            };
            self.phase = Some(phase);
        }
        self.phase = Some(phase);
        parameters.validate()?;
        let mut automatic;
        let parameters = if let Some(config) = &parameters.curvature {
            if self.configuration.as_ref() != Some(config) {
                self.curvature = None;
                self.configuration = Some(config.clone());
                self.pending_iteration = false;
                self.last_mean = None;
            }
            let state = self.curvature.get_or_insert_with(|| config.initial_state());
            let Some(bands) = state.bands()? else {
                return Ok(raw.to_vec());
            };
            automatic = bands;
            automatic.multi_point = parameters.multi_point;
            automatic.max_points = parameters.max_points;
            &automatic
        } else {
            self.curvature = None;
            self.configuration = None;
            parameters
        };
        self.max_points = if parameters.multi_point {
            parameters.max_points
        } else {
            0
        };
        if parameters.multi_point {
            self.register_point(&self.center.clone(), false);
            super::multipoint::retain_planes(&mut self.samples, self.max_points)?;
        }
        // Backends may decline independent copies. Raw pricing preserves the
        // original solve contract in that case, and on an unfinished copy solve.
        let Some(mut private) = master.try_clone_for_transaction()? else {
            return Ok(raw.to_vec());
        };
        let mut outer_columns = vec![];
        let mut band_columns = vec![vec![]; parameters.bands.len()];
        for (row, &center) in self.center.iter().enumerate() {
            if excluded.contains(&row) {
                continue;
            }
            for (band_index, band) in parameters.bands.iter().enumerate() {
                let outer = parameters.multi_point && band_index + 1 == parameters.bands.len();
                let mut pair = [0; 2];
                // +e constrains pi <= center+width; -e constrains the
                // opposite side. Finite upper bounds soften these constraints.
                for (i, sign) in [-1., 1.].into_iter().enumerate() {
                    let cost = if outer {
                        0.
                    } else {
                        sign * center + band.half_width
                    };
                    if !cost.is_finite() {
                        return Ok(raw.to_vec());
                    }
                    let column = private.add_column(cost, &[(row, sign)])?;
                    band_columns[band_index].push(column);
                    pair[i] = column;
                    if band.penalty.is_finite() {
                        private.set_upper(column, band.penalty)?;
                    }
                }
                if outer {
                    outer_columns.push((row, pair[0], pair[1]));
                }
            }
        }
        if parameters.multi_point {
            let samples: Vec<_> = self.samples.iter().map(|s| s.point.clone()).collect();
            super::multipoint::add_support_planes(&mut private, &outer_columns, &samples)?;
        }
        if private.solve_with_time_limit(remaining)? != LpSolveOutcome::Optimal {
            return Ok(raw.to_vec());
        }
        let mut point = private.duals().get(..raw.len()).unwrap_or(&[]).to_vec();
        if point.len() != raw.len() || point.iter().any(|x| !x.is_finite()) {
            return Ok(raw.to_vec());
        }
        self.artificial_values = band_columns
            .iter()
            .map(|columns| {
                columns.iter().try_fold(0_f64, |maximum, &column| {
                    let value = *private.values().get(column)?;
                    value.is_finite().then_some(maximum.max(value.abs()))
                })
            })
            .collect();
        for &row in excluded {
            if let Some(value) = point.get_mut(row) {
                *value = raw[row];
            }
        }
        if parameters.multi_point {
            let activity = private
                .duals()
                .get(raw.len()..raw.len() + self.samples.len())
                .map(|values| values.iter().map(|v| -*v).collect::<Vec<_>>());
            if super::multipoint::observe_activity(&mut self.samples, activity.as_deref()) {
                super::multipoint::retain_planes(&mut self.samples, self.max_points)?;
            }
            self.register_point(&point, false);
        }
        Ok(point)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/stabilization/proximal/tests.rs"]
mod tests;
