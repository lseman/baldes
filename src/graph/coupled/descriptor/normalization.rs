//! Explicit dependent-resource objective conventions.
use super::RawFunctionError;

/// The dependent front and its objective baseline are separate objects.
///
/// This convention does not change the initial function or any resource
/// transition. It determines the charge of a nonempty prefix or complete route.
/// A fresh search root has accumulated charge zero under either convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "clp", serde(rename_all = "snake_case"))]
pub enum PairedCostNormalization {
    /// Charge the terminal dependent value, including the initial value.
    AbsoluteFinalValue,
    /// Charge the terminal value minus the initial front's minimum.
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
        if !initial_minimum.is_finite() || !terminal_minimum.is_finite() || !scale.is_finite() {
            return Err(RawFunctionError::NonFinite("dependent objective component"));
        }
        if scale < 0. {
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
mod tests {
    use super::*;

    #[test]
    fn absolute_native_prefix_charges_include_the_initial_value() {
        let absolute = PairedCostNormalization::AbsoluteFinalValue;
        let relative = PairedCostNormalization::FinalMinusInitialMinimum;
        assert_eq!(absolute.root_charge(), 0.);
        assert_eq!(relative.root_charge(), 0.);
        for (minimum, native_charge, relative_charge) in [(6., 6., 1.), (7., 7., 2.), (8., 8., 3.)]
        {
            assert_eq!(absolute.charge(5., minimum, 1.).unwrap(), native_charge);
            assert_eq!(relative.charge(5., minimum, 1.).unwrap(), relative_charge);
        }
    }

    #[test]
    fn suppression_does_not_evaluate_an_overflowing_unused_difference() {
        let relative = PairedCostNormalization::FinalMinusInitialMinimum;
        assert_eq!(relative.charge(-f64::MAX, f64::MAX, 0.).unwrap(), 0.);
        assert!(relative.charge(-f64::MAX, f64::MAX, 1.).is_err());
        assert!(relative.charge(0., f64::MAX, 2.).is_err());
        assert!(relative.charge(0., 1., -1.).is_err());
        assert!(relative.charge(f64::NAN, 1., 0.).is_err());
    }

    #[cfg(feature = "clp")]
    #[test]
    fn serialized_conventions_remain_explicit_and_distinct() {
        for (convention, wire) in [
            (
                PairedCostNormalization::AbsoluteFinalValue,
                "absolute_final_value",
            ),
            (
                PairedCostNormalization::FinalMinusInitialMinimum,
                "final_minus_initial_minimum",
            ),
        ] {
            let encoded = serde_json::to_string(&convention).unwrap();
            assert_eq!(encoded, format!("\"{wire}\""));
            assert_eq!(
                serde_json::from_str::<PairedCostNormalization>(&encoded).unwrap(),
                convention
            );
        }
    }
}
