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
    EndpointTolerance,
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
            || (policy == EvaluationPolicy::EndpointTolerance && near(argument, lower)?)
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
                || (policy == EvaluationPolicy::EndpointTolerance && near(argument, upper)?)
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
        if policy == EvaluationPolicy::EndpointTolerance && near(argument, upper)? {
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
#[path = "../../../../tests/unit/graph/coupled/descriptor/slope/tests.rs"]
mod tests;
