//! Continuous piecewise-linear arc functions with explicit finite domains.
use super::kernel::bounds::Range;
use super::*;

/// Knot pairs `[argument, value]`, with strictly increasing finite arguments.
/// Values may increase, decrease or remain constant. Evaluation outside the
/// first/last argument is infeasible; no extrapolation is performed.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "clp", serde(deny_unknown_fields))]
pub struct PiecewiseLinearFunction {
    pub points: Vec<[f64; 2]>,
}
impl PiecewiseLinearFunction {
    pub fn new(points: Vec<[f64; 2]>) -> Result<Self> {
        let value = Self { points };
        value.validate()?;
        Ok(value)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if self.points.len() < 2
            || self.points.iter().flatten().any(|x| !x.is_finite())
            || self.points.windows(2).any(|p| {
                p[0][0] >= p[1][0]
                    || !(p[1][0] - p[0][0]).is_finite()
                    || !(p[1][1] - p[0][1]).is_finite()
            })
        {
            return Err(invalid("piecewise-linear functions need at least two ordered finite knots and finite segment differences"));
        }
        Ok(())
    }
    pub fn domain(&self) -> Result<Window> {
        self.validate()?;
        Ok(self.prepared_domain())
    }
    pub(super) fn prepared_domain(&self) -> Window {
        Window {
            lower: self.points[0][0],
            upper: self.points[self.points.len() - 1][0],
        }
    }
    /// Evaluate with exact knot values and ordered interpolation operations.
    pub fn value(&self, argument: f64) -> Result<Option<f64>> {
        self.validate()?;
        self.evaluate(argument)
    }
    pub(super) fn evaluate(&self, argument: f64) -> Result<Option<f64>> {
        finite(argument)?;
        let domain = self.prepared_domain();
        if argument < domain.lower || argument > domain.upper {
            return Ok(None);
        }
        let upper = self.points.partition_point(|p| p[0] < argument);
        if self.points[upper][0] == argument {
            return Ok(Some(self.points[upper][1]));
        }
        let [x0, y0] = self.points[upper - 1];
        let [x1, y1] = self.points[upper];
        let t = finite(finite(argument - x0)? / (x1 - x0))?;
        Ok(Some(
            finite(y0 + finite(t * (y1 - y0))?)?.clamp(y0.min(y1), y0.max(y1)),
        ))
    }
    pub(super) fn range(&self, argument: Range) -> Option<Range> {
        let domain = self.prepared_domain();
        let argument = Range {
            lo: argument.lo.max(domain.lower),
            hi: argument.hi.min(domain.upper),
        };
        if argument.lo > argument.hi {
            return None;
        }
        if argument.lo == argument.hi {
            if let Ok(index) = self
                .points
                .binary_search_by(|p| p[0].total_cmp(&argument.lo))
            {
                return Some(Range::point(self.points[index][1]));
            }
        }
        let mut result = Range {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        };
        for pair in self.points.windows(2) {
            let [x0, y0] = pair[0];
            let [x1, y1] = pair[1];
            let x = Range {
                lo: argument.lo.max(x0),
                hi: argument.hi.min(x1),
            };
            if x.lo > x.hi {
                continue;
            }
            let t = x.subtract(Range::point(x0)).divide_positive(x1 - x0);
            let value = Range::point(y0).add(t.multiply(Range::point(y1 - y0)));
            result.lo = result.lo.min(value.lo.max(y0.min(y1)));
            result.hi = result.hi.max(value.hi.min(y0.max(y1)));
            // The evaluator returns literal knot values without interpolation.
            // Include those branches separately from the rounded segment map.
            if argument.lo <= x0 && argument.hi >= x0 {
                result.lo = result.lo.min(y0);
                result.hi = result.hi.max(y0);
            }
            if argument.lo <= x1 && argument.hi >= x1 {
                result.lo = result.lo.min(y1);
                result.hi = result.hi.max(y1);
            }
        }
        Some(result)
    }
    /// Conservative preimage hull, including flat/nonmonotone segments.
    /// Search uses ordered binary64 keys and the literal monotone segment
    /// evaluator. It is bounded by 64 steps per endpoint and includes all
    /// rounding plateaus; no algebraic inverse or slope division is needed.
    pub(super) fn preimage(&self, input: Range, output: Range) -> Option<Range> {
        fn key(x: f64) -> u64 {
            let b = x.to_bits();
            if b >> 63 != 0 {
                !b
            } else {
                b | (1 << 63)
            }
        }
        fn value(k: u64) -> f64 {
            f64::from_bits(if k >> 63 != 0 { k & !(1 << 63) } else { !k })
        }
        let domain = self.prepared_domain();
        let input = Range {
            lo: input.lo.max(domain.lower),
            hi: input.hi.min(domain.upper),
        };
        if input.lo > input.hi {
            return None;
        }
        let mut result: Option<Range> = None;
        for pair in self.points.windows(2) {
            let mut segment = Range {
                lo: input.lo.max(pair[0][0]),
                hi: input.hi.min(pair[1][0]),
            };
            if segment.lo > segment.hi {
                continue;
            }
            let increasing = pair[1][1] >= pair[0][1];
            let evaluate = |x: f64| {
                self.evaluate(x)
                    .ok()
                    .flatten()
                    .map(|y| if increasing { y } else { -y })
            };
            let target = if increasing {
                output
            } else {
                Range {
                    lo: -output.hi,
                    hi: -output.lo,
                }
            };
            if let (Some(lo), Some(hi)) = (evaluate(segment.lo), evaluate(segment.hi)) {
                if hi < target.lo || lo > target.hi {
                    continue;
                }
                let (first, last) = (key(segment.lo), key(segment.hi));
                let (mut low, mut high) = (first, last);
                let mut known = true;
                if lo < target.lo {
                    while low < high {
                        let middle = low + (high - low) / 2;
                        match evaluate(value(middle)) {
                            Some(y) if y < target.lo => low = middle + 1,
                            Some(_) => high = middle,
                            None => {
                                known = false;
                                break;
                            }
                        }
                    }
                    if known {
                        segment.lo = value(low);
                    }
                }
                let (mut low, mut high) = (first, last);
                if hi > target.hi {
                    while low < high {
                        let middle = low + (high - low) / 2 + (high - low) % 2;
                        match evaluate(value(middle)) {
                            Some(y) if y > target.hi => high = middle - 1,
                            Some(_) => low = middle,
                            None => {
                                known = false;
                                break;
                            }
                        }
                    }
                    if known {
                        segment.hi = value(low);
                    }
                }
            }
            if segment.lo > segment.hi {
                continue;
            }
            if let Some(range) = &mut result {
                range.lo = range.lo.min(segment.lo);
                range.hi = range.hi.max(segment.hi);
            } else {
                result = Some(segment);
            }
        }
        result
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/graph/coupled/pwl/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/graph/coupled/pwl/preimage_tests.rs"]
mod preimage_tests;
