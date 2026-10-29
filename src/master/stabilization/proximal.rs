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

/// Native rule-1 defaults. Positive Kappa initializes intervals from the first
/// positive mean; nonpositive Kappa uses configured stored-float intervals.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(default, deny_unknown_fields))]
pub struct ExplicitParameters {
    /// Native functiontype1 normalization: InnerAngle float1.01, OuterAngle0.
    pub box_step: bool,
    pub five_piece: bool,
    pub kappa: f64,
    pub outer_half_width: f64,
    pub inner_half_width: f64,
    pub outer_penalty: f64,
    pub inner_penalty: f64,
}
impl Default for ExplicitParameters {
    fn default() -> Self {
        Self {
            box_step: false,
            five_piece: false,
            kappa: 1.,
            outer_half_width: 100.,
            inner_half_width: 10.,
            outer_penalty: 0.9,
            inner_penalty: 0.9,
        }
    }
}
impl ExplicitParameters {
    fn validate(&self) -> Result<()> {
        if self.box_step && self.five_piece {
            return Err(invalid("explicit box-step and five-piece modes conflict"));
        }
        if !((self.kappa as f32).is_finite())
            || [
                self.outer_half_width,
                self.inner_half_width,
                self.outer_penalty,
                self.inner_penalty,
            ]
            .iter()
            .any(|&value| !(value as f32).is_finite() || value < 0.)
        {
            return Err(invalid("invalid explicit explicit stabilization controls"));
        }
        Ok(())
    }
    fn initial_bands(&self) -> Vec<ProximalBand> {
        let initial = |width: f64| {
            if (self.kappa as f32) > 0. {
                1e12
            } else {
                width as f32 as f64
            }
        };
        let mut bands = vec![];
        if self.five_piece {
            bands.push(ProximalBand {
                half_width: initial(self.outer_half_width),
                penalty: self.outer_penalty as f32 as f64,
            });
        }
        bands.push(ProximalBand {
            half_width: initial(
                (self.inner_half_width as f32).min(self.outer_half_width as f32) as f64,
            ),
            penalty: if self.box_step {
                1.01_f32 as f64
            } else {
                self.inner_penalty as f32 as f64
            },
        });
        bands
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
    /// Maximum retained support planes; the explicit public Kappa default is 1.
    #[cfg_attr(
        feature = "clp",
        serde(
            default = "default_max_points",
            skip_serializing_if = "is_default_max_points"
        )
    )]
    pub max_points: usize,
    /// Native explicit-rule child interval factor, widened from stored float.
    #[cfg_attr(
        feature = "clp",
        serde(
            default = "default_child_factor",
            skip_serializing_if = "is_default_child_factor"
        )
    )]
    pub child_half_interval_factor: f64,
    /// Optional recovered explicit rule-1 initialization and interval schedule.
    #[cfg_attr(
        feature = "clp",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub explicit: Option<ExplicitParameters>,
    /// Use original row RHS for configured-band artificial geometry. Native
    /// explicit and automatic workflows always use the recovered RHS scaling.
    #[cfg_attr(
        feature = "clp",
        serde(default, skip_serializing_if = "std::ops::Not::not")
    )]
    pub rhs_row_scaling: bool,
    #[cfg_attr(
        feature = "clp",
        serde(
            default = "default_child_factor",
            skip_serializing_if = "is_default_child_factor"
        )
    )]
    pub artificial_activity_factor: f64,
}

fn default_child_factor() -> f64 {
    10.
}
#[cfg(feature = "clp")]
fn is_default_child_factor(value: &f64) -> bool {
    *value == default_child_factor()
}

fn default_max_points() -> usize {
    1
}
#[cfg(feature = "clp")]
fn is_default_max_points(value: &usize) -> bool {
    *value == default_max_points()
}

impl ProximalParameters {
    /// Recovered explicit functiontype1; unlike `box_step`, its normalized cap
    /// is finite float1.01 and its default intervals bootstrap from Kappa.
    pub fn finite_box_step() -> Self {
        Self::explicit(ExplicitParameters {
            box_step: true,
            ..Default::default()
        })
    }
    pub fn explicit_three_piece() -> Self {
        Self::explicit(ExplicitParameters::default())
    }
    pub fn explicit_five_piece() -> Self {
        Self::explicit(ExplicitParameters {
            five_piece: true,
            ..Default::default()
        })
    }
    pub fn explicit(explicit: ExplicitParameters) -> Self {
        let mut parameters = Self::three_piece(0., 1.);
        parameters.bands.clear();
        parameters.explicit = Some(explicit);
        parameters.rhs_row_scaling = true;
        parameters
    }
    pub fn explicit_multipoint() -> Self {
        let band = ProximalBand {
            half_width: 0.,
            penalty: 0.9_f32 as f64,
        };
        let mut parameters = Self::five_piece(band, band);
        parameters.multi_point = true;
        parameters.rhs_row_scaling = true;
        parameters
    }
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
            child_half_interval_factor: default_child_factor(),
            explicit: None,
            rhs_row_scaling: true,
            artificial_activity_factor: 10.,
        }
    }

    pub fn box_step(half_width: f64) -> Self {
        Self {
            curvature: None,
            multi_point: false,
            max_points: default_max_points(),
            child_half_interval_factor: default_child_factor(),
            explicit: None,
            rhs_row_scaling: false,
            artificial_activity_factor: 10.,
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
            child_half_interval_factor: default_child_factor(),
            explicit: None,
            rhs_row_scaling: false,
            artificial_activity_factor: 10.,
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
            child_half_interval_factor: default_child_factor(),
            explicit: None,
            rhs_row_scaling: false,
            artificial_activity_factor: 10.,
            bands: vec![inner, outer],
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let activity_factor = self.artificial_activity_factor as f32;
        if !activity_factor.is_finite() || activity_factor <= 0. {
            return Err(invalid("invalid proximal artificial-activity factor"));
        }
        if let Some(explicit) = &self.explicit {
            explicit.validate()?;
            if !self.bands.is_empty()
                || self.curvature.is_some()
                || self.multi_point
                || self.max_points == 0
            {
                return Err(invalid(
                    "explicit explicit configuration conflicts with other proximal modes",
                ));
            }
        }
        let child_factor = self.child_half_interval_factor as f32;
        if !child_factor.is_finite() || child_factor < 0. {
            return Err(invalid("invalid proximal child half-interval factor"));
        }
        if let Some(curvature) = &self.curvature {
            curvature.validate()?;
            if !self.bands.is_empty() || self.max_points == 0 || self.multi_point {
                return Err(invalid("automatic curvature requires no explicit bands"));
            }
            return Ok(());
        }
        if self.explicit.is_some() {
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

/// Accepted dual center, explicit intervals and automatic curvature inherited
/// by a compatible child. Multipoint samples retain pure dual coordinates;
/// auxiliary LP columns and rows are never retained.
#[derive(Clone, Debug)]
pub struct ProximalSnapshot {
    pub center: Vec<f64>,
    pub bound: f64,
    pub phase: usize,
    pub curvature: Option<super::curvature::CurvatureState>,
    pub configuration: Option<super::curvature::AutomaticCurvatureParameters>,
    /// Explicit bands retain their inner/outer identities, independent of widths.
    pub explicit_bands: Vec<ProximalBand>,
    pub explicit_baseline: Option<f64>,
    pub support_planes: Vec<super::multipoint::SupportPlane>,
    pub max_points: usize,
    pub next_point_id: u64,
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
    explicit_bands: Vec<ProximalBand>,
    explicit_baseline: Option<f64>,
}

impl ProximalState {
    /// A CG schedule/ordinary artificial-penalty restart does not erase the
    /// accepted proximal geometry. Unfinished observations are dropped.
    pub fn restart_schedule(&mut self) {
        self.pending_iteration = false;
        self.artificial_values = None;
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
        self.explicit_bands.clear();
        self.explicit_baseline = None;
    }

    pub fn snapshot(&self) -> Option<ProximalSnapshot> {
        Some(ProximalSnapshot {
            center: self.center.clone(),
            bound: self.bound,
            phase: self.phase?,
            curvature: self.curvature,
            configuration: self.configuration.clone(),
            explicit_bands: self.explicit_bands.clone(),
            explicit_baseline: self.explicit_baseline,
            support_planes: self.samples.clone(),
            max_points: self.max_points,
            next_point_id: self.next_point_id,
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
        if snapshot.explicit_bands.iter().any(|band| {
            !band.half_width.is_finite()
                || band.half_width < 0.
                || band.penalty.is_nan()
                || band.penalty < 0.
        }) || snapshot
            .explicit_baseline
            .is_some_and(|x| !x.is_finite() || x < 0.)
        {
            return Err(invalid("invalid inherited explicit proximal intervals"));
        }
        self.configuration.clone_from(&snapshot.configuration);
        self.explicit_bands.clone_from(&snapshot.explicit_bands);
        self.explicit_baseline = snapshot.explicit_baseline;
        if (!snapshot.support_planes.is_empty() && snapshot.max_points == 0)
            || snapshot.support_planes.iter().any(|plane| {
                plane.point.len() != snapshot.center.len()
                    || plane.point.iter().any(|x| !x.is_finite())
                    || !plane.dual_activity.is_finite()
                    || plane.creation_id >= snapshot.next_point_id
            })
        {
            return Err(invalid("invalid inherited multipoint support planes"));
        }
        self.samples.clone_from(&snapshot.support_planes);
        self.max_points = snapshot.max_points;
        self.next_point_id = snapshot.next_point_id;
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
        if self.center.len() != raw.len() || self.phase.is_none() {
            self.reset();
            self.center = vec![0.; raw.len()];
            self.phase = Some(phase);
        }
        if self.phase != Some(phase) {
            self.bound = f64::NEG_INFINITY;
        }
        self.phase = Some(phase);
        if parameters.explicit.is_some() && !self.pending_iteration {
            self.last_mean = super::curvature::mean_dual_distance(&self.center, raw, ordinary_rows);
            self.pending_iteration = true;
        }
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
        if parameters.curvature.is_none() && !parameters.multi_point {
            if let Some(explicit) = &parameters.explicit {
                if self.pending_iteration {
                    self.pending_iteration = false;
                    if let (Some(mean), Some(outer)) = (self.last_mean, self.explicit_bands.last())
                    {
                        let sentinel_margin = outer.half_width.abs().max(1e12) * 1e-6 + 1e-6;
                        if (outer.half_width - 1e12).abs() <= sentinel_margin
                            && mean > mean.abs() * 1e-10 + 1e-6
                        {
                            let scale = (explicit.kappa as f32 as f64) * mean;
                            let outer_width = if explicit.five_piece || explicit.box_step {
                                scale * 0.1
                            } else {
                                scale
                            };
                            if !scale.is_finite() || scale < 0. {
                                return Err(invalid(
                                    "explicit explicit interval bootstrap overflow",
                                ));
                            }
                            self.explicit_bands.last_mut().unwrap().half_width = outer_width;
                            if explicit.five_piece {
                                self.explicit_bands[0].half_width = scale;
                            }
                            self.explicit_baseline = Some(outer_width);
                            return Ok(true);
                        }
                    }
                }
            }
            // Native explicit mode halves the outer interval when no inner
            // stabilization artificial has significant activity.
            let inner_activity = if self.explicit_bands.len() > 1 {
                self.artificial_values
                    .as_ref()
                    .and_then(|v| v.first())
                    .copied()
            } else {
                self.artificial_values.as_ref().map(|_| 0.)
            };
            if inner_activity.is_some_and(|value| value.abs() <= value.abs() * 1e-6 + 1e-6) {
                if let Some(outer) = self.explicit_bands.last_mut() {
                    let margin = outer.half_width.abs().max(1e12) * 1e-10 + 1e-6;
                    if outer.half_width < 1e12 - margin {
                        outer.half_width *= 0.5;
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
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
        if active && parameters.curvature.is_none() {
            let factor = parameters.artificial_activity_factor as f32 as f64;
            let count = self.explicit_bands.len();
            let mut next = self.explicit_bands.clone();
            let mut changed = false;
            for (index, band) in next.iter_mut().enumerate() {
                if (!parameters.multi_point || index + 1 == count) && band.penalty.is_finite() {
                    let penalty = band.penalty / factor;
                    if !penalty.is_finite() {
                        return Err(invalid("proximal artificial-cap update overflow"));
                    }
                    changed |= penalty != band.penalty;
                    band.penalty = penalty;
                }
            }
            self.explicit_bands = next;
            return Ok(changed);
        }
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
        if self.center.len() != raw.len() || self.phase.is_none() {
            self.reset();
            self.center = vec![0.; raw.len()];
            self.phase = Some(phase);
        }
        if self.phase != Some(phase) {
            self.bound = f64::NEG_INFINITY;
        }
        self.phase = Some(phase);
        parameters.validate()?;
        let mut automatic;
        let mut explicit;
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
            automatic.rhs_row_scaling = true;
            &automatic
        } else {
            self.curvature = None;
            self.configuration = None;
            if self.explicit_bands.is_empty() {
                self.explicit_bands = parameters.explicit.as_ref().map_or_else(
                    || parameters.bands.clone(),
                    ExplicitParameters::initial_bands,
                );
                self.explicit_baseline = self.explicit_bands.last().map(|band| band.half_width);
            }
            explicit = parameters.clone();
            explicit.bands.clone_from(&self.explicit_bands);
            // Preserve logical inner/outer identities even when explicit child
            // setup makes the inner half-width larger than the outer one.
            &explicit
        };
        self.max_points = if parameters.multi_point {
            parameters.max_points
        } else {
            0
        };
        if parameters.multi_point {
            if !self
                .samples
                .iter()
                .any(|sample| sample.point == self.center)
            {
                let center = std::mem::take(&mut self.center);
                self.register_point(&center, false);
                self.center = center;
            }
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
            let rhs = *master
                .rhs()
                .get(row)
                .ok_or_else(|| invalid("proximal row RHS missing"))?;
            let rhs_scaling = parameters.rhs_row_scaling || parameters.explicit.is_some();
            let coefficient_scale = if rhs_scaling && rhs.abs() > 1e-6 {
                rhs
            } else {
                1.
            };
            let cost_scale = if rhs_scaling && rhs.abs() > rhs.abs() * 1e-6 + 1e-6 {
                rhs
            } else {
                1.
            };
            for (band_index, band) in parameters.bands.iter().enumerate() {
                let outer = parameters.multi_point && band_index + 1 == parameters.bands.len();
                let mut pair = [0; 2];
                // +e constrains pi <= center+width; -e constrains the
                // opposite side. Finite upper bounds soften these constraints.
                for (i, sign) in [-1., 1.].into_iter().enumerate() {
                    let cost = if outer {
                        0.
                    } else {
                        cost_scale * (sign * center + band.half_width)
                    };
                    if !cost.is_finite() {
                        return Ok(raw.to_vec());
                    }
                    let column = private.add_column(cost, &[(row, sign * coefficient_scale)])?;
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
            super::multipoint::add_support_plane_refs(
                &mut private,
                &outer_columns,
                self.samples.iter().map(|sample| sample.point.as_slice()),
            )?;
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
