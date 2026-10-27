//! Explicit dependent-resource objective conventions.
use super::RawFunctionError;

/// The dependent front and its objective baseline are separate objects.
///
/// This convention does not change the initial function or any resource
/// transition. It determines the charge of a nonempty prefix or complete route.
/// A fresh search root has accumulated charge zero under either convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "clp", serde(rename_all = "snake_case"))]
pub enum PairedCostNormalization {
    /// Charge the terminal dependent value, including the initial value.
    AbsoluteFinalValue,
    /// Charge the terminal value minus the initial front's minimum.
    #[default]
    FinalMinusInitialMinimum,
}

impl PairedCostNormalization {
    /// Accumulated objective at a fresh root, before any physical transition.
    pub fn root_charge(self) -> f64 {
        0.
    }

    /// Convert a nonempty prefix's or complete route's front minimum into its
    /// objective component. The front remains in its original coordinates.
    ///
    /// `scale` is nonnegative: zero suppresses the objective in phase I without
    /// changing function-domain or resource-feasibility checks at the caller.
    /// Suppression precedes subtraction so an unused difference cannot overflow.
    pub fn charge(
        self,
        initial_minimum: f64,
        terminal_minimum: f64,
        scale: f64,
    ) -> Result<f64, RawFunctionError> {
        self.charge_using(initial_minimum, terminal_minimum, scale, false)
    }

    /// Charge a particular selected value, not an optimum of a minimum front.
    /// Signed weights are valid here; optimizing their controls is the caller's
    /// separate obligation. Minimum-front labeling rejects negative models.
    pub(crate) fn charge_selected(
        self,
        initial: f64,
        selected: f64,
        scale: f64,
    ) -> Result<f64, RawFunctionError> {
        self.charge_using(initial, selected, scale, true)
    }

    fn charge_using(
        self,
        initial_minimum: f64,
        terminal_minimum: f64,
        scale: f64,
        selected: bool,
    ) -> Result<f64, RawFunctionError> {
        if !initial_minimum.is_finite() || !terminal_minimum.is_finite() || !scale.is_finite() {
            return Err(RawFunctionError::NonFinite("dependent objective component"));
        }
        if scale < 0. && !selected {
            return Err(RawFunctionError::Unsupported(
                "negative scales require maximizing the dependent front",
            ));
        }
        if scale == 0. {
            return Ok(0.);
        }
        let amount = match self {
            Self::AbsoluteFinalValue => terminal_minimum,
            Self::FinalMinusInitialMinimum => terminal_minimum - initial_minimum,
        };
        if !amount.is_finite() {
            return Err(RawFunctionError::ArithmeticOverflow(
                "dependent objective baseline",
            ));
        }
        let charge = amount * scale;
        if !charge.is_finite() {
            return Err(RawFunctionError::ArithmeticOverflow(
                "dependent objective scaling",
            ));
        }
        Ok(charge)
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/normalization/tests.rs"]
mod tests;
