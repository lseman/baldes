use std::{error::Error, fmt};

/// Boundary behavior for the slope encoding, independent of knot functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "clp", serde(rename_all = "snake_case"))]
pub enum EvaluationPolicy {
    /// Exact closed domain; no coordinate snapping or extrapolation.
    StrictDomain,
    /// Snap at the first coordinate, and just beyond the final coordinate,
    /// using strict absolute/relative comparisons recovered for this encoding.
    /// This policy is not a tolerance for feasibility or dominance proofs.
    NativeTolerance,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawFunctionError {
    NonFinite(&'static str),
    Unsupported(&'static str),
    EmptyPreparedFunction,
    InvalidDomain,
    InvalidOrdering { pair: usize },
    ArithmeticOverflow(&'static str),
}

impl fmt::Display for RawFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(field) => write!(f, "non-finite slope-function {field}"),
            Self::Unsupported(reason) => {
                write!(f, "unsupported raw descriptor operation: {reason}")
            }
            Self::EmptyPreparedFunction => {
                write!(f, "a prepared slope function needs an initial pair")
            }
            Self::InvalidDomain => write!(f, "slope-function lower domain exceeds upper domain"),
            Self::InvalidOrdering { pair } => write!(
                f,
                "slope-function pair {pair} must end after the preceding coordinate"
            ),
            Self::ArithmeticOverflow(operation) => {
                write!(f, "non-finite slope-function arithmetic in {operation}")
            }
        }
    }
}
impl Error for RawFunctionError {}

type Result<T> = std::result::Result<T, RawFunctionError>;

/// Unprepared affine descriptor: its pair means `(intercept, slope)`.
/// This differs from a one-pair *prepared* point function.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize))]
pub struct RawAffine {
    intercept: f64,
    slope: f64,
}

impl RawAffine {
    pub fn new(intercept: f64, slope: f64) -> Result<Self> {
        finite(intercept, "intercept")?;
        finite(slope, "slope")?;
        Ok(Self { intercept, slope })
    }

    pub fn from_pair(pair: (f64, f64)) -> Result<Self> {
        Self::new(pair.0, pair.1)
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Evaluate `intercept + slope * argument`, with separate operations.
    /// IEEE underflow and signed zeros are retained; non-finite results fail.
    pub fn evaluate(&self, argument: f64) -> Result<f64> {
        finite(argument, "argument")?;
        let product = arithmetic(self.slope * argument, "affine multiplication")?;
        arithmetic(self.intercept + product, "affine addition")
    }

    /// Anchor at `lower` using affine arithmetic, then retain the literal slope.
    /// Later prepared evaluations need not round like direct affine evaluation.
    pub fn prepare(&self, lower: f64, upper: f64) -> Result<PreparedSlopeFunction> {
        finite(lower, "domain lower")?;
        finite(upper, "domain upper")?;
        if lower > upper {
            return Err(RawFunctionError::InvalidDomain);
        }
        let initial = self.evaluate(lower)?;
        if lower == upper {
            return PreparedSlopeFunction::constant_point(initial, lower);
        }
        PreparedSlopeFunction::from_prepared_pairs(vec![(initial, lower), (self.slope, upper)])
    }
}

/// Immutable validated prepared slope representation, with a closed domain.
/// No conversion to endpoint knots is claimed to preserve literal arithmetic.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize))]
pub struct PreparedSlopeFunction {
    pairs: Vec<(f64, f64)>,
}

impl PreparedSlopeFunction {
    /// A single pair is a prepared point `(value, coordinate)`, not an affine
    /// `(intercept, slope)` descriptor. All later coordinates strictly increase.
    pub fn from_prepared_pairs(pairs: Vec<(f64, f64)>) -> Result<Self> {
        let &(mut value, mut previous) = pairs
            .first()
            .ok_or(RawFunctionError::EmptyPreparedFunction)?;
        finite(value, "initial value")?;
        finite(previous, "initial coordinate")?;
        for (index, &(slope, end)) in pairs.iter().enumerate().skip(1) {
            finite(slope, "slope")?;
            finite(end, "segment end")?;
            if end <= previous {
                return Err(RawFunctionError::InvalidOrdering { pair: index });
            }
            let span = arithmetic(end - previous, "segment span")?;
            let product = arithmetic(slope * span, "segment multiplication")?;
            value = arithmetic(value + product, "segment accumulation")?;
            previous = end;
        }
        Ok(Self { pairs })
    }

    pub fn constant_point(value: f64, coordinate: f64) -> Result<Self> {
        Self::from_prepared_pairs(vec![(value, coordinate)])
    }

    pub fn pairs(&self) -> &[(f64, f64)] {
        &self.pairs
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.pairs[0].1, self.pairs[self.pairs.len() - 1].1)
    }

    /// `None` means outside the selected boundary policy. A genuine value1e12
    /// is returned as `Some(1e12)` rather than confused with an outside sentinel.
    /// Interior knots use the segment ending at that knot. Fully traversed
    /// segments accumulate in input order; no fused multiply/add is requested.
    pub fn evaluate(&self, argument: f64, policy: EvaluationPolicy) -> Result<Option<f64>> {
        finite(argument, "argument")?;
        let (lower, upper) = self.domain();
        let mut value = self.pairs[0].0;
        if argument == lower
            || (policy == EvaluationPolicy::NativeTolerance && near(argument, lower)?)
        {
            return Ok(Some(value));
        }
        if argument < lower {
            return Ok(None);
        }
        if policy == EvaluationPolicy::StrictDomain && argument > upper {
            return Ok(None);
        }
        if self.pairs.len() == 1 {
            return if argument == lower
                || (policy == EvaluationPolicy::NativeTolerance && near(argument, upper)?)
            {
                Ok(Some(value))
            } else {
                Ok(None)
            };
        }
        let mut previous = lower;
        for &(slope, end) in &self.pairs[1..] {
            let end_of_evaluation = if argument <= end { argument } else { end };
            let span = arithmetic(end_of_evaluation - previous, "evaluation span")?;
            let product = arithmetic(slope * span, "evaluation multiplication")?;
            value = arithmetic(value + product, "evaluation accumulation")?;
            if argument <= end {
                return Ok(Some(value));
            }
            previous = end;
        }
        if policy == EvaluationPolicy::NativeTolerance && near(argument, upper)? {
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }
}

fn finite(value: f64, field: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(RawFunctionError::NonFinite(field))
    }
}

fn arithmetic(value: f64, operation: &'static str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RawFunctionError::ArithmeticOverflow(operation))
    }
}

fn near(argument: f64, boundary: f64) -> Result<bool> {
    let difference = arithmetic(argument - boundary, "boundary difference")?.abs();
    if difference < 0.0001 {
        return Ok(true);
    }
    let sum = arithmetic(argument + boundary, "boundary sum")?.abs();
    let relative = arithmetic(2.220446049250313e-10 * sum, "relative boundary band")?;
    Ok(difference < relative)
}

#[cfg(feature = "clp")]
impl<'de> serde::Deserialize<'de> for RawAffine {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            intercept: f64,
            slope: f64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.intercept, wire.slope).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "clp")]
impl<'de> serde::Deserialize<'de> for PreparedSlopeFunction {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            pairs: Vec<(f64, f64)>,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::from_prepared_pairs(wire.pairs).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_prepared_slope_values_and_left_knot_are_preserved() {
        let function =
            PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 5.), (-1., 10.)])
                .unwrap();
        let actual: Vec<_> = [0., 2., 5., 7., 10.]
            .into_iter()
            .map(|x| {
                function
                    .evaluate(x, EvaluationPolicy::StrictDomain)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        assert_eq!(actual, [2., 8., 17., 15., 12.]);
        let before = f64::from_bits(5_f64.to_bits() - 1);
        assert_eq!(
            function
                .evaluate(before, EvaluationPolicy::StrictDomain)
                .unwrap(),
            Some(16.999999999999996)
        );
        assert_eq!(
            function
                .evaluate(
                    f64::from_bits(5_f64.to_bits() + 1),
                    EvaluationPolicy::StrictDomain
                )
                .unwrap(),
            Some(17.)
        );
    }

    #[test]
    fn fractional_native_witness_is_not_endpoint_knot_interpolation() {
        let x0 = 61.42565465487604;
        let y0 = 7.245618290940143;
        let slope = 9.462315279587411;
        let x1 = 68.72567520394728;
        let argument = 64.1889633870269;
        let function =
            PreparedSlopeFunction::from_prepared_pairs(vec![(y0, x0), (slope, x1)]).unwrap();
        let native = function
            .evaluate(argument, EvaluationPolicy::StrictDomain)
            .unwrap()
            .unwrap();
        let endpoint = y0 + slope * (x1 - x0);
        let knots = y0 + ((argument - x0) / (x1 - x0)) * (endpoint - y0);
        assert_eq!(native, 33.39291672938855);
        assert_eq!(knots, 33.39291672938856);
        assert_ne!(native.to_bits(), knots.to_bits());
    }

    #[test]
    fn explicit_boundary_policy_matches_native_start_and_end_observations() {
        let function =
            PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 5.), (-1., 10.)])
                .unwrap();
        assert_eq!(
            function
                .evaluate(-0.00001, EvaluationPolicy::StrictDomain)
                .unwrap(),
            None
        );
        assert_eq!(
            function
                .evaluate(-0.00001, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(2.)
        );
        assert_eq!(
            function
                .evaluate(0.00001, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(2.)
        );
        assert_eq!(
            function
                .evaluate(-0.0001, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            None
        );
        assert_eq!(
            function
                .evaluate(0.0001, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(2.0003)
        );
        assert_eq!(
            function
                .evaluate(10.00001, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(12.)
        );
        assert_eq!(
            function
                .evaluate(10.001, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            None
        );
        let large =
            PreparedSlopeFunction::from_prepared_pairs(vec![(2., 1e8), (3., 1e8 + 10.)]).unwrap();
        assert_eq!(
            large
                .evaluate(1e8 - 0.01, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(2.)
        );
        assert_eq!(
            large
                .evaluate(1e8 + 0.01, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(2.)
        );
        assert_eq!(
            large
                .evaluate(1e8 + 10.01, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(32.)
        );
    }

    #[test]
    fn affine_pair_and_prepared_point_have_distinct_meanings() {
        let affine = RawAffine::from_pair((2., 3.)).unwrap();
        assert_eq!(affine.evaluate(10.).unwrap(), 32.);
        assert_eq!(
            affine.prepare(0., 10.).unwrap().pairs(),
            &[(2., 0.), (3., 10.)]
        );
        let point = PreparedSlopeFunction::constant_point(2., 3.).unwrap();
        assert_eq!(point.domain(), (3., 3.));
        assert_eq!(
            point.evaluate(3., EvaluationPolicy::StrictDomain).unwrap(),
            Some(2.)
        );
        assert_eq!(
            point.evaluate(10., EvaluationPolicy::StrictDomain).unwrap(),
            None
        );
        assert_eq!(affine.prepare(3., 3.).unwrap().pairs(), &[(11., 3.)]);
    }

    #[test]
    fn genuine_sentinel_value_is_not_misclassified_as_outside() {
        let function =
            PreparedSlopeFunction::from_prepared_pairs(vec![(1e12, 0.), (0., 1.)]).unwrap();
        assert_eq!(
            function
                .evaluate(0.5, EvaluationPolicy::StrictDomain)
                .unwrap(),
            Some(1e12)
        );
        assert_eq!(
            function
                .evaluate(2., EvaluationPolicy::StrictDomain)
                .unwrap(),
            None
        );
    }

    #[test]
    fn validation_and_arithmetic_failures_are_explicit() {
        assert!(PreparedSlopeFunction::from_prepared_pairs(vec![]).is_err());
        assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (1., 0.)]).is_err());
        assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(0., 1.), (1., 0.)]).is_err());
        assert!(PreparedSlopeFunction::from_prepared_pairs(vec![(f64::NAN, 0.)]).is_err());
        assert!(
            PreparedSlopeFunction::from_prepared_pairs(vec![(0., 0.), (f64::MAX, 2.)]).is_err()
        );
        assert!(RawAffine::new(f64::MAX, 1.)
            .unwrap()
            .evaluate(f64::MAX)
            .is_err());
        let point = PreparedSlopeFunction::constant_point(0., 1e308).unwrap();
        assert_eq!(
            point
                .evaluate(1e308, EvaluationPolicy::NativeTolerance)
                .unwrap(),
            Some(0.)
        );
        assert!(point
            .evaluate(1e308 - 1e300, EvaluationPolicy::NativeTolerance)
            .is_err());
        assert!(point
            .evaluate(f64::INFINITY, EvaluationPolicy::StrictDomain)
            .is_err());
    }

    #[test]
    fn signed_zero_at_anchor_is_preserved_without_a_knot_conversion() {
        let point = PreparedSlopeFunction::constant_point(-0., 0.).unwrap();
        assert_eq!(
            point
                .evaluate(0., EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap()
                .to_bits(),
            (-0_f64).to_bits()
        );
        let function =
            PreparedSlopeFunction::from_prepared_pairs(vec![(-0., 0.), (0., 1.)]).unwrap();
        assert_eq!(
            function
                .evaluate(0., EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap()
                .to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            function
                .evaluate(0., EvaluationPolicy::NativeTolerance)
                .unwrap()
                .unwrap()
                .to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            function
                .evaluate(0.5, EvaluationPolicy::StrictDomain)
                .unwrap()
                .unwrap()
                .to_bits(),
            0_f64.to_bits()
        );
    }

    #[cfg(feature = "clp")]
    #[test]
    fn serde_round_trip_cannot_bypass_prepared_validation() {
        let function =
            PreparedSlopeFunction::from_prepared_pairs(vec![(2., 0.), (3., 5.)]).unwrap();
        let wire = serde_json::to_string(&function).unwrap();
        assert_eq!(
            serde_json::from_str::<PreparedSlopeFunction>(&wire).unwrap(),
            function
        );
        assert!(
            serde_json::from_str::<PreparedSlopeFunction>(r#"{"pairs":[[0,1],[2,0]]}"#).is_err()
        );
        assert!(serde_json::from_str::<PreparedSlopeFunction>(r#"{"pairs":[]}"#).is_err());
        assert!(
            serde_json::from_str::<RawAffine>(r#"{"intercept":0,"slope":1,"extra":3}"#).is_err()
        );
    }
}
