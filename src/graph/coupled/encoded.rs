//! Literal function encodings used by numeric resource programs.
use super::kernel::bounds::Range;
use super::*;

/// A function and its evaluation contract. Slope descriptors retain their
/// original prepared pairs; they are never converted into interpolated knots.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize))]
#[cfg_attr(
    feature = "clp",
    serde(tag = "encoding", rename_all = "snake_case", deny_unknown_fields)
)]
pub enum EncodedFunction {
    Knots {
        function: PiecewiseLinearFunction,
    },
    SlopeSegments {
        function: PreparedSlopeFunction,
        policy: EvaluationPolicy,
    },
}
#[cfg(feature = "clp")]
impl<'de> serde::Deserialize<'de> for EncodedFunction {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(tag = "encoding", rename_all = "snake_case", deny_unknown_fields)]
        enum Input {
            Knots {
                function: PiecewiseLinearFunction,
            },
            SlopeSegments {
                function: PreparedSlopeFunction,
                policy: EvaluationPolicy,
            },
        }
        let value = match <Input as serde::Deserialize>::deserialize(deserializer)? {
            Input::Knots { function } => Self::Knots { function },
            Input::SlopeSegments { function, policy } => Self::SlopeSegments { function, policy },
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}
impl From<PiecewiseLinearFunction> for EncodedFunction {
    fn from(function: PiecewiseLinearFunction) -> Self {
        Self::Knots { function }
    }
}
/// Empty is a proved empty image/preimage. Unknown carries no pruning proof.
#[derive(Clone, Copy, Debug)]
pub(super) enum FunctionEnclosure {
    Empty,
    Known(Range),
    Unknown,
}
impl FunctionEnclosure {
    fn conservative(self) -> Option<Range> {
        match self {
            Self::Empty => None,
            Self::Known(range) => Some(range),
            Self::Unknown => Some(Range::unknown()),
        }
    }
}
impl EncodedFunction {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Knots { function } => function.validate(),
            // PreparedSlopeFunction construction/deserialization validates its
            // private representation. Revalidate to keep this installation boundary explicit.
            Self::SlopeSegments { function, .. } => {
                PreparedSlopeFunction::from_prepared_pairs(function.pairs().to_vec())
                    .map(|_| ())
                    .map_err(|error| invalid(error.to_string()))
            }
        }
    }
    /// Conservative hull of accepted arguments. Native tolerance uses an
    /// unknown hull because it can admit values beyond literal endpoints.
    pub fn domain(&self) -> Result<Window> {
        self.validate()?;
        Ok(self.prepared_domain())
    }
    pub fn value(&self, argument: f64) -> Result<Option<f64>> {
        self.validate()?;
        self.evaluate(argument)
    }
    pub(super) fn evaluate(&self, argument: f64) -> Result<Option<f64>> {
        match self {
            Self::Knots { function } => function.evaluate(argument),
            Self::SlopeSegments { function, policy } => function
                .evaluate(argument, *policy)
                .map_err(|error| invalid(error.to_string())),
        }
    }
    pub(super) fn prepared_domain(&self) -> Window {
        match self {
            Self::Knots { function } => function.prepared_domain(),
            Self::SlopeSegments {
                function,
                policy: EvaluationPolicy::StrictDomain,
            } => {
                let (lower, upper) = function.domain();
                Window { lower, upper }
            }
            // Native endpoint tolerance accepts arguments outside the descriptor's
            // exact endpoints. An exact-endpoint hull would incorrectly prune them.
            Self::SlopeSegments { .. } => Window {
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY,
            },
        }
    }
    pub(super) fn image(&self, argument: Range) -> FunctionEnclosure {
        match self {
            Self::Knots { function } => function
                .range(argument)
                .map_or(FunctionEnclosure::Empty, FunctionEnclosure::Known),
            Self::SlopeSegments { .. } => {
                let domain = self.prepared_domain();
                if argument.hi < domain.lower || argument.lo > domain.upper {
                    FunctionEnclosure::Empty
                } else {
                    FunctionEnclosure::Unknown
                }
            }
        }
    }
    pub(super) fn inverse_image(&self, input: Range, output: Range) -> FunctionEnclosure {
        match self {
            Self::Knots { function } => function
                .preimage(input, output)
                .map_or(FunctionEnclosure::Empty, FunctionEnclosure::Known),
            Self::SlopeSegments { .. } => {
                let domain = self.prepared_domain();
                if input.hi < domain.lower || input.lo > domain.upper || output.lo > output.hi {
                    FunctionEnclosure::Empty
                } else {
                    FunctionEnclosure::Unknown
                }
            }
        }
    }
    pub(super) fn range(&self, argument: Range) -> Option<Range> {
        self.image(argument).conservative()
    }
    pub(super) fn preimage(&self, input: Range, output: Range) -> Option<Range> {
        match self.inverse_image(input, output) {
            FunctionEnclosure::Unknown => Some(input),
            other => other.conservative(),
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/graph/coupled/encoded/tests.rs"]
mod tests;
