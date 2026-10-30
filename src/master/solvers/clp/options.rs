//! Native Clp reoptimization policy, independent of LP feasibility tolerances.
use super::*;

/// Matrix scaling policy. Scaling is disabled by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClpScaling {
    #[default]
    Off,
    Equilibrium,
    Geometric,
    Auto,
    /// Five simultaneous infinity-norm Ruiz equilibration passes.
    Ruiz,
}
impl ClpScaling {
    pub(super) fn code(self) -> i32 {
        match self {
            Self::Off => 0,
            Self::Equilibrium => 1,
            Self::Geometric => 2,
            Self::Auto => 3,
            Self::Ruiz => 6,
        }
    }
}
impl std::str::FromStr for ClpScaling {
    type Err = crate::core::error::OrError;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "equilibrium" => Ok(Self::Equilibrium),
            "geometric" => Ok(Self::Geometric),
            "auto" => Ok(Self::Auto),
            "ruiz" => Ok(Self::Ruiz),
            _ => Err(invalid(
                "Clp scaling must be auto, geometric, equilibrium, ruiz, or off",
            )),
        }
    }
}

/// Reoptimization settings for the Clp backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClpOptions {
    pub scaling: ClpScaling,
    /// Retain native work areas between solves; initialization and matrix
    /// invalidation remain enabled. Does not force reuse of a factorization.
    pub keep_work_areas: bool,
}
impl Default for ClpOptions {
    fn default() -> Self {
        Self {
            scaling: ClpScaling::Off,
            keep_work_areas: true,
        }
    }
}
impl ClpMaster {
    pub fn clp_options(&self) -> ClpOptions {
        self.clp_options
    }

    /// Changing policy invalidates the cached solution. Scaling mode changes
    /// also clear native scale factors before the next solve.
    pub fn set_clp_options(&mut self, options: ClpOptions) {
        if self.clp_options != options {
            if let Some(model) = self
                .model
                .filter(|_| self.clp_options.scaling != options.scaling)
            {
                // SAFETY: live, exclusively borrowed native model. Clear old
                // row/column factors as well as the scaled matrix before a
                // positive-to-positive mode change on a retained model.
                unsafe {
                    Clp_scaling(model.as_ptr(), 0);
                    if options.scaling != ClpScaling::Off {
                        Clp_scaling(model.as_ptr(), options.scaling.code());
                    }
                }
            }
            self.clp_options = options;
            self.solved = false;
        }
    }
}
