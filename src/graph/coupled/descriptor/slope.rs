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

    /// Prepare a singleton affine input with the machine's two-pair output.
    /// Equal or reversed bounds retain the literal machine-only geometry.
    pub fn prepare_machine(&self, lower: f64, upper: f64) -> Result<PreparedSlopeFunction> {
        finite(lower, "domain lower")?;
        finite(upper, "domain upper")?;
        let initial = self.evaluate(lower)?;
        PreparedSlopeFunction::from_machine_pairs(vec![(initial, lower), (self.slope, upper)])
    }
}

/// Immutable validated prepared slope representation, with a closed domain.
/// No conversion to endpoint knots is claimed to preserve literal arithmetic.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clp", derive(serde::Serialize))]
pub struct PreparedSlopeFunction {
    pairs: Vec<(f64, f64)>,
    // Generated only by the machine-front algebra. It preserves the native
    // fixed array's zero-length segment; source constructors never admit it.
    #[cfg_attr(feature = "clp", serde(skip))]
    machine_zero_length: bool,
    // Other machine arrays can have duplicate endpoints or a reversed first
    // segment. Their consumers are not those of a point front.
    #[cfg_attr(feature = "clp", serde(skip))]
    machine_internal_zero_length: bool,
}

impl PartialEq for PreparedSlopeFunction {
    fn eq(&self, other: &Self) -> bool {
        let same =
            |a: f64, b: f64| a == b || a.is_nan() && b.is_nan() && a.to_bits() == b.to_bits();
        self.machine_zero_length == other.machine_zero_length
            && self.machine_internal_zero_length == other.machine_internal_zero_length
            && self.pairs.len() == other.pairs.len()
            && self
                .pairs
                .iter()
                .zip(&other.pairs)
                .all(|(a, b)| same(a.0, b.0) && same(a.1, b.1))
    }
}

impl PreparedSlopeFunction {
    /// Truncate using the ordered machine scan, including its near-knot
    /// advancement, one-sided clipping and exhausted-stream output. This is
    /// a machine operation; an error does not certify source infeasibility.
    pub fn machine_truncate(&self, lower: f64, upper: f64) -> Result<Self> {
        self.machine_truncate_ordered(lower, upper, false)
    }

    /// Fixed-array truncation retains one zero-length segment when a short
    /// remainder emits no segment. The vector operation instead keeps a point.
    pub(super) fn machine_truncate_fixed(&self, lower: f64, upper: f64) -> Result<Self> {
        self.machine_truncate_ordered(lower, upper, true)
    }

    fn machine_truncate_ordered(&self, lower: f64, upper: f64, fixed_array: bool) -> Result<Self> {
        finite(lower, "machine truncation lower")?;
        finite(upper, "machine truncation upper")?;
        if lower > upper && !near(lower, upper)? {
            return Err(RawFunctionError::Unsupported("machine truncation geometry"));
        }
        let (start, end) = self.domain();
        if (lower > end && !near(lower, end)?) || (upper < start && !near(upper, start)?) {
            return Err(RawFunctionError::Unsupported("machine truncation bounds"));
        }
        if (lower <= start || near(lower, start)?) && (upper >= end || near(upper, end)?) {
            return Ok(self.clone());
        }
        let mut value = self.pairs[0].0;
        let mut previous = start;
        let mut first = 1;
        // A near endpoint is consumed even if it lies just beyond lower.
        // The next segment then reanchors from that endpoint, possibly using
        // a negative span. Replacing this with a point lookup changes both
        // the selected slope and the literal rounded value.
        while let Some(&(slope, endpoint)) = self.pairs.get(first) {
            if !near(endpoint, lower)? && lower < endpoint {
                break;
            }
            let span = arithmetic(endpoint - previous, "machine truncation span")?;
            let product = arithmetic(span * slope, "machine truncation multiplication")?;
            value = arithmetic(value + product, "machine truncation accumulation")?;
            previous = endpoint;
            first += 1;
        }
        if first == self.pairs.len() {
            return Self::machine_zero_length(value, 0.0, lower, lower);
        }
        // Select the requested operand on equality, preserving coordinate
        // bits just as in the ordered clipping instructions.
        let lower = if start <= lower { lower } else { start };
        let upper = if fixed_array || upper <= end {
            upper
        } else {
            end
        };
        let span = arithmetic(lower - previous, "machine truncation span")?;
        let product = arithmetic(
            span * self.pairs[first].0,
            "machine truncation multiplication",
        )?;
        let anchor = arithmetic(value + product, "machine truncation accumulation")?;
        let mut output = Vec::with_capacity(self.pairs.len() - first + 1);
        output.push((anchor, lower));
        let mut current = first;
        let mut last = lower;
        while let Some(&(slope, endpoint)) = self.pairs.get(current) {
            if near(endpoint, upper)? || upper <= endpoint {
                break;
            }
            output.push((slope, endpoint));
            last = endpoint;
            current += 1;
        }
        // A near final remainder contributes no segment. A near original
        // knot, however, keeps the active slope and requested end coordinate.
        if !near(upper, last)? && last < upper && current < self.pairs.len() {
            output.push((self.pairs[current].0, upper));
        }
        if fixed_array && output.len() == 1 {
            output.push((0., lower));
        }
        Self::from_machine_pairs(output)
    }

    /// Prepare the machine forward vector from an already truncated vector.
    /// This does not perform the reference's arc or vertex truncation and is
    /// not a source-control equivalence certificate.
    pub fn machine_forward_from_truncated(&self) -> Result<Self> {
        let mut pairs = self.pairs.clone();
        pairs[0].0 = arithmetic(pairs[0].0 + pairs[0].1, "machine forward anchor")?;
        for pair in &mut pairs[1..] {
            pair.0 = arithmetic(1.0 + pair.0, "machine forward slope")?;
        }
        Self::from_machine_pairs(pairs)
    }
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
        Ok(Self {
            pairs,
            machine_zero_length: false,
            machine_internal_zero_length: false,
        })
    }

    /// Decode a finite machine array. Duplicate coordinates and a reversed
    /// first segment remain machine-only, not source descriptors.
    pub fn from_machine_pairs(pairs: Vec<(f64, f64)>) -> Result<Self> {
        let mut function = match pairs.as_slice() {
            [(value, lower), (slope, upper)] if lower == upper => {
                Self::machine_zero_length(*value, *slope, *lower, *upper)?
            }
            [(value, coordinate), (slope, duplicate), (waiting, end)]
                if coordinate == duplicate && end > duplicate =>
            {
                Self::machine_internal_zero_length(*value, *slope, *coordinate, *waiting, *end)?
            }
            _ => {
                if !pairs.windows(2).any(|window| window[0].1 >= window[1].1) {
                    return Self::from_prepared_pairs(pairs);
                }
                let &(mut value, mut previous) = pairs
                    .first()
                    .ok_or(RawFunctionError::EmptyPreparedFunction)?;
                finite(value, "initial value")?;
                finite(previous, "initial coordinate")?;
                for (index, &(slope, end)) in pairs.iter().enumerate().skip(1) {
                    finite(slope, "slope")?;
                    finite(end, "segment end")?;
                    // The fixed-array check begins coordinate comparisons
                    // with the second segment, so retain the first literally.
                    if index > 1 && end < previous {
                        return Err(RawFunctionError::InvalidOrdering { pair: index });
                    }
                    let span = arithmetic(end - previous, "segment span")?;
                    value = arithmetic(
                        value + arithmetic(slope * span, "segment multiplication")?,
                        "segment accumulation",
                    )?;
                    previous = end;
                }
                return Ok(Self {
                    pairs,
                    machine_zero_length: true,
                    machine_internal_zero_length: true,
                });
            }
        };
        // Preserve signed-zero coordinate bits exactly as supplied.
        function.pairs = pairs;
        Ok(function)
    }

    /// Retain a generated machine array's undefined slopes.
    /// Public source and machine-input constructors remain finite-only. A raw
    /// output can contain 0/0 at an empty segment without making its earlier
    /// coordinates unavailable to the literal machine lookup and join scan.
    pub(super) fn from_machine_output_pairs(pairs: Vec<(f64, f64)>) -> Result<Self> {
        if pairs.iter().all(|pair| pair.0.is_finite()) {
            return Self::from_machine_pairs(pairs);
        }
        let &(anchor, first) = pairs
            .first()
            .ok_or(RawFunctionError::EmptyPreparedFunction)?;
        finite(anchor, "initial value")?;
        finite(first, "initial coordinate")?;
        let mut previous = first;
        for (index, &(slope, end)) in pairs.iter().enumerate().skip(1) {
            finite(end, "segment end")?;
            if index > 1 && end < previous {
                return Err(RawFunctionError::InvalidOrdering { pair: index });
            }
            if slope.is_infinite() {
                return Err(RawFunctionError::Unsupported(
                    "infinite machine output slope",
                ));
            }
            previous = end;
        }
        Ok(Self {
            pairs,
            machine_zero_length: true,
            machine_internal_zero_length: true,
        })
    }

    /// Retain a native fixed array's `(value, lower), (slope, upper)` output
    /// when its coordinates compare equal. This is an output representation,
    /// not an accepted source descriptor; downstream operations must opt in.
    pub(super) fn machine_zero_length(
        value: f64,
        slope: f64,
        lower: f64,
        upper: f64,
    ) -> Result<Self> {
        finite(value, "initial value")?;
        finite(slope, "slope")?;
        finite(lower, "initial coordinate")?;
        finite(upper, "segment end")?;
        if lower != upper {
            return Err(RawFunctionError::InvalidOrdering { pair: 1 });
        }
        Ok(Self {
            pairs: vec![(value, lower), (slope, upper)],
            machine_zero_length: true,
            machine_internal_zero_length: false,
        })
    }

    /// Preserve the recovered machine array `(v,x),(s,x),(w,end)`.
    /// This is not a source descriptor and has no general consumer contract.
    pub(super) fn machine_internal_zero_length(
        value: f64,
        slope: f64,
        coordinate: f64,
        waiting_slope: f64,
        end: f64,
    ) -> Result<Self> {
        finite(value, "initial value")?;
        finite(slope, "slope")?;
        finite(coordinate, "initial coordinate")?;
        finite(waiting_slope, "waiting slope")?;
        finite(end, "segment end")?;
        if end <= coordinate {
            return Err(RawFunctionError::InvalidOrdering { pair: 2 });
        }
        let span = arithmetic(end - coordinate, "segment span")?;
        let product = arithmetic(waiting_slope * span, "segment multiplication")?;
        arithmetic(value + product, "segment accumulation")?;
        Ok(Self {
            pairs: vec![
                (value, coordinate),
                (slope, coordinate),
                (waiting_slope, end),
            ],
            machine_zero_length: true,
            machine_internal_zero_length: true,
        })
    }

    /// Whether this output retains at least one zero-length segment.
    pub fn is_machine_zero_length(&self) -> bool {
        self.pairs
            .windows(2)
            .any(|window| window[0].1 == window[1].1)
    }

    /// Whether the stored geometry requires the machine-only contract.
    pub fn has_machine_only_geometry(&self) -> bool {
        self.machine_zero_length
    }

    pub(super) fn has_internal_zero_length(&self) -> bool {
        self.machine_internal_zero_length
    }

    pub fn constant_point(value: f64, coordinate: f64) -> Result<Self> {
        Self::from_prepared_pairs(vec![(value, coordinate)])
    }

    pub fn pairs(&self) -> &[(f64, f64)] {
        &self.pairs
    }

    /// Validate the fixed-array output contract used by the paired program.
    /// This checks machine geometry, not source-function feasibility.
    pub(crate) fn check_machine_array(&self) -> Result<bool> {
        if !(2..=12).contains(&self.pairs.len()) || self.pairs[0].0 == 1e12 {
            return Ok(false);
        }
        for index in 2..self.pairs.len() {
            let previous = self.pairs[index - 1].1;
            let current = self.pairs[index].1;
            if current <= previous || near(current, previous)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Machine lookup with the selected segment id. A near upper knot can
    /// advance the id while retaining evaluation at the unsnapped argument.
    pub(super) fn machine_value_and_segment(&self, argument: f64) -> Result<Option<(f64, usize)>> {
        self.machine_lookup_segment(argument, false)
    }

    /// Lookup that advances past a near final segment. The vector lookup and
    /// the fixed-array helper used by concatenation both have this behavior;
    /// some inlined fixed-array lookups retain the final segment instead.
    /// Both evaluate at the original argument without snapping it to the knot.
    pub(in crate::graph::coupled) fn machine_vector_value_and_segment(
        &self,
        argument: f64,
    ) -> Result<Option<(f64, usize)>> {
        self.machine_lookup_segment(argument, true)
    }

    /// Domain check used by the paired initializer for a vector input.
    pub(crate) fn check_machine_domain(&self, lower: f64, upper: f64) -> Result<bool> {
        for pair in self.pairs.windows(2) {
            let previous = pair[0].1;
            let next = pair[1].1;
            if next <= previous || near(next, previous)? {
                return Ok(false);
            }
        }
        Ok(near(self.pairs[0].1, lower)? && near(self.pairs[self.pairs.len() - 1].1, upper)?)
    }

    fn machine_lookup_segment(&self, argument: f64, vector: bool) -> Result<Option<(f64, usize)>> {
        Self::machine_lookup_pairs(&self.pairs, argument, vector)
    }

    /// Fixed-array helper lookup for finite machine intermediates. Their
    /// coordinates need not satisfy the source/prepared representation.
    pub(super) fn machine_array_value_and_segment(
        pairs: &[(f64, f64)],
        argument: f64,
    ) -> Result<Option<(f64, usize)>> {
        Self::machine_lookup_pairs(pairs, argument, true)
    }

    fn machine_lookup_pairs(
        pairs: &[(f64, f64)],
        argument: f64,
        vector: bool,
    ) -> Result<Option<(f64, usize)>> {
        finite(argument, "machine lookup argument")?;
        let lower = pairs[0].1;
        let upper = pairs[pairs.len() - 1].1;
        let mut value = pairs[0].0;
        if near(argument, lower)? {
            return Ok(Some((value, 1)));
        }
        if argument < lower {
            return Ok(None);
        }
        let mut previous = lower;
        for (index, &(slope, end)) in pairs.iter().enumerate().skip(1) {
            if argument <= end {
                let span = argument - previous;
                let product = span * slope;
                value += product;
                let selected = if (vector || index + 1 < pairs.len()) && near(end, argument)? {
                    index + 1
                } else {
                    index
                };
                return Ok(Some((value, selected)));
            }
            let span = end - previous;
            let product = span * slope;
            value += product;
            previous = end;
        }
        if near(argument, upper)? {
            Ok(Some((
                value,
                if vector { pairs.len() } else { pairs.len() - 1 },
            )))
        } else {
            Ok(None)
        }
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.pairs[0].1, self.pairs[self.pairs.len() - 1].1)
    }

    /// `None` means outside the selected boundary policy. A genuine value1e12
    /// is returned as `Some(1e12)` rather than confused with an outside sentinel.
    /// Interior knots use the segment ending at that knot. Fully traversed
    /// segments accumulate in input order; no fused multiply/add is requested.
    pub fn evaluate(&self, argument: f64, policy: EvaluationPolicy) -> Result<Option<f64>> {
        if self.machine_internal_zero_length {
            return Err(RawFunctionError::Unsupported(
                "machine internal zero-length segment evaluation",
            ));
        }
        if self.machine_zero_length {
            finite(argument, "argument")?;
            if policy != EvaluationPolicy::EndpointTolerance {
                return Err(RawFunctionError::Unsupported(
                    "machine zero-length segment strict evaluation",
                ));
            }
            return if near(argument, self.pairs[0].1)? {
                Ok(Some(self.pairs[0].0))
            } else {
                Ok(None)
            };
        }
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

#[cfg(feature = "clp")]
pub(crate) fn deserialize_machine_function<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<PreparedSlopeFunction, D::Error> {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Wire {
        pairs: Vec<(f64, f64)>,
        #[serde(default)]
        machine_zero_length: bool,
        #[serde(default)]
        machine_internal_zero_length: bool,
    }
    let wire = <Wire as serde::Deserialize>::deserialize(deserializer)?;
    let function =
        PreparedSlopeFunction::from_machine_pairs(wire.pairs).map_err(serde::de::Error::custom)?;
    if (wire.machine_zero_length && !function.machine_zero_length)
        || (wire.machine_internal_zero_length && !function.machine_internal_zero_length)
    {
        return Err(serde::de::Error::custom(
            "inconsistent machine function flags",
        ));
    }
    Ok(function)
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/descriptor/slope/tests.rs"]
mod tests;
