//! Fixed-array paired-state splitting. This is machine geometry, not a source
//! function restriction or a certificate of source-control feasibility.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct SplitState {
    pub(super) pairs: Vec<(f64, f64)>,
    pub(super) waiting_bits: u64,
    pub(super) predecessor: f64,
    pub(super) key: f64,
}

fn failure(message: &'static str) -> FrontierUnknown {
    FrontierUnknown::Unsupported(message)
}

/// The caller retains the original label when the helper returns at most one
/// piece. Do that here as well, including its original key and predecessor.
pub(super) fn split_state(
    state: &SplitState,
    forward: bool,
    strategy: i32,
    halfway: f64,
    mode: i32,
    ceiling: f64,
    bucket_ends: &[f64],
    bucket_grid: [f64; 3],
    budget: FrontierBudget,
) -> FrontResult<Vec<SplitState>> {
    let mut work = Work::new(budget);
    let pairs = &state.pairs;
    if budget.max_components == 0 {
        return Err(FrontierUnknown::ComponentLimit);
    }
    if pairs.len() > budget.max_pairs {
        return Err(FrontierUnknown::PairLimit);
    }
    if pairs.is_empty() || pairs.len() > 12 || !halfway.is_finite() {
        return Err(failure("paired split state dimensions or halfway point"));
    }
    let lower = pairs[0].1;
    let upper = pairs[pairs.len() - 1].1;
    work.charge(pairs.len())?;
    if !matches!(strategy, 1 | 2)
        || pairs.len() == 1
        || (strategy == 1 && pairs.len() == 2)
        || (forward && lower > halfway + 1e-6)
        || (!forward && upper <= halfway + 1e-6)
    {
        return Ok(vec![state.clone()]);
    }
    if pairs.iter().any(|&(a, b)| !a.is_finite() || !b.is_finite())
        || pairs.windows(2).any(|p| p[0].1 > p[1].1)
        || !state.predecessor.is_finite()
        || !ceiling.is_finite()
        || state.waiting_bits & !((1 << 13) - 1) != 0
    {
        return Err(failure(
            "paired split requires finite ordered machine state",
        ));
    }
    // The reference writes sentinel keys in mode 4, then its unconditional
    // forward bucket checker compares them to numerical minima and exits(1).
    if forward && strategy == 2 && mode == 4 {
        return Err(failure(
            "paired forward bucket split has inconsistent feasibility keys",
        ));
    }
    let linear = strategy == 1;
    let aligned = |a, b| near(a, b, FrontOperationPolicy::FrontAlgebra);
    if !linear && !forward {
        for i in 2..pairs.len() {
            let same_waiting =
                ((state.waiting_bits >> (i - 1)) & 1) == ((state.waiting_bits >> i) & 1);
            if aligned(pairs[i - 1].1, pairs[i].1)?
                || (same_waiting && aligned(pairs[i - 1].0, pairs[i].0)?)
            {
                return Err(failure(
                    "paired bucket split requires distinct consecutive segments",
                ));
            }
        }
    }
    if !linear && forward && aligned(lower, halfway)? {
        let [grid_lower, grid_upper, step] = bucket_grid;
        if !grid_lower.is_finite() || !grid_upper.is_finite() || !step.is_finite() || step <= 0. {
            return Err(failure(
                "paired split requires finite positive bucket geometry",
            ));
        }
        let count = (((grid_upper + 1e-6) - grid_lower) / step).floor() + 1.;
        let source = ((lower - (grid_lower - 1e-6)) / step).floor();
        let next = source + 1.;
        let boundary = if next < count - 1. {
            grid_lower + next * step
        } else {
            grid_upper
        };
        if next < count && aligned(lower, boundary)? {
            // The initial aligned bucket boundary advances the function
            // cursor. Its unconditional consistency check rejects a changed
            // source slope before the caller can discard a singleton result.
            let mut selected = None;
            for i in 1..pairs.len() {
                if pairs[i].1 <= boundary + 1e-6 || aligned(pairs[i].1, boundary)? {
                    selected = Some((i + 1).min(pairs.len() - 1));
                }
            }
            if let Some(selected) = selected {
                if !aligned(pairs[1].0, pairs[selected].0)? {
                    return Err(failure(
                        "paired bucket split has inconsistent segment slopes",
                    ));
                }
            }
        }
    }
    let mut cuts = Vec::new();
    let mut prefix = false;
    let mut translated_backward = false;
    let mut prefix_next_index = None;
    let mut prefix_next_start = None;
    let mut skip_upper = false;
    if linear {
        if forward {
            // This is the literal anchor-value comparison in the helper,
            // distinct from the dispatcher's first-coordinate comparison.
            if aligned(pairs[0].0, halfway)? || halfway <= pairs[0].0 {
                return Ok(vec![state.clone()]);
            }
            cuts.extend(
                pairs[1..]
                    .iter()
                    .filter_map(|p| (p.1 <= halfway + 1e-6).then_some(p.1)),
            );
            if halfway < upper && cuts.last() != Some(&upper) {
                cuts.push(halfway);
            }
        } else {
            translated_backward = aligned(pairs[0].0, halfway)? || halfway <= pairs[0].0;
            if translated_backward {
                cuts.extend(pairs[1..].iter().map(|p| p.1));
            } else {
                if aligned(lower, halfway)? {
                    return Ok(vec![state.clone()]);
                }
                if lower < halfway && halfway < upper {
                    cuts.push(halfway);
                    prefix = true;
                    let mut next = pairs
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find_map(|(i, p)| (p.1 > halfway + 1e-6).then_some(i))
                        .unwrap_or(pairs.len());
                    prefix_next_start =
                        Some(if next < pairs.len() && aligned(pairs[next].1, halfway)? {
                            let start = pairs[next].1;
                            next += 1;
                            start
                        } else {
                            halfway
                        });
                    prefix_next_index = Some(next);
                    cuts.extend(pairs[next..].iter().map(|p| p.1));
                    skip_upper = true;
                } else {
                    cuts.extend(
                        pairs[1..]
                            .iter()
                            .filter_map(|p| (p.1 > halfway).then_some(p.1)),
                    );
                }
            }
        }
    } else {
        if bucket_ends.iter().any(|x| !x.is_finite())
            || bucket_ends.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(failure(
                "paired split bucket ends must be finite and ordered",
            ));
        }
        work.charge(bucket_ends.len())?;
        if forward {
            if aligned(lower, halfway)? || halfway + 1e-6 < lower {
                return Ok(vec![state.clone()]);
            }
            for &x in bucket_ends {
                if lower < x && (x <= halfway + 1e-6 || aligned(x, halfway)?) && x < upper {
                    cuts.push(x);
                }
            }
            let separate_halfway = match cuts.last() {
                Some(&x) => !aligned(x, halfway)?,
                None => true,
            };
            if lower < halfway && halfway < upper && !aligned(halfway, upper)? && separate_halfway {
                cuts.push(halfway);
            }
        } else {
            if lower < halfway && halfway < upper {
                let endpoint = pairs
                    .iter()
                    .skip(1)
                    .rev()
                    .find_map(|p| (p.1 <= halfway + 1e-6).then_some(p.1));
                let boundary = match endpoint {
                    Some(x) if aligned(x, halfway)? => x,
                    _ => halfway,
                };
                cuts.push(boundary);
                prefix = true;
                let mut next = pairs
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find_map(|(i, p)| (p.1 > boundary).then_some(i))
                    .unwrap_or(pairs.len());
                if next < pairs.len() && aligned(pairs[next].1, halfway)? {
                    next += 1;
                }
                prefix_next_index = Some(next);
                if next == pairs.len() {
                    return Ok(vec![state.clone()]);
                }
            }
            for &x in bucket_ends {
                if lower.max(halfway) < x && x < upper && !aligned(x, lower.max(halfway))? {
                    cuts.push(x);
                }
            }
        }
    }
    if !skip_upper && cuts.last() != Some(&upper) {
        cuts.push(upper);
    }
    if cuts.len() > budget.max_components {
        return Err(FrontierUnknown::ComponentLimit);
    }
    if cuts.len() <= 1 {
        return Ok(vec![state.clone()]);
    }
    let mut outputs = Vec::with_capacity(cuts.len());
    let mut start = if translated_backward { halfway } else { lower };
    let mut carry = pairs[0].0;
    let mut index = 1;
    for (position, &end) in cuts.iter().enumerate() {
        work.charge(1)?;
        if start > end {
            return Err(failure(
                "paired linear split produces decreasing coordinates",
            ));
        }
        let mut at = index;
        let mut value = carry;
        let mut coordinate = start;
        if forward
            && !linear
            && position + 1 == cuts.len()
            && lower < halfway
            && halfway < upper
            && start != halfway
            && aligned(start, halfway)?
        {
            value = arithmetic(
                value
                    + arithmetic(
                        (halfway - start)
                            * pairs[if halfway + 1e-6 < start && at > 1 && pairs[at - 1].1 == start
                            {
                                at - 1
                            } else {
                                at
                            }]
                            .0,
                        "paired split tail correction",
                    )?,
                "paired split tail anchor",
            )?;
            start = halfway;
            coordinate = halfway;
        }
        let mut output = vec![(value, start)];
        work.pair()?;
        let mut bits = 0;
        let first_segment = at;
        while at < pairs.len()
            && pairs[at].1
                <= end
                    + if prefix && position == 0 && linear {
                        1e-6
                    } else {
                        0.
                    }
        {
            work.charge(1)?;
            let (slope, endpoint) = pairs[at];
            value = arithmetic(
                value + arithmetic((endpoint - coordinate) * slope, "paired split segment")?,
                "paired split accumulation",
            )?;
            coordinate = endpoint;
            output.push(pairs[at]);
            work.pair()?;
            bits |= ((state.waiting_bits >> at) & 1) << (output.len() - 1);
            at += 1;
        }
        if output.last().unwrap().1 < end
            || output.len() == 1
            || (prefix && position == 0 && linear && output.last().unwrap().1 > end)
        {
            let slope = pairs
                .get(at)
                .ok_or_else(|| failure("paired split escaped state domain"))?
                .0;
            value = arithmetic(
                value + arithmetic((end - coordinate) * slope, "paired split partial segment")?,
                "paired split partial accumulation",
            )?;
            output.push((slope, end));
            work.pair()?;
            bits |= ((state.waiting_bits >> at) & 1) << (output.len() - 1);
        }
        carry = value;
        index = if prefix && position == 0 {
            prefix_next_index.unwrap_or(at)
        } else {
            at
        };
        if prefix && position == 0 && linear {
            if let Some(i) = pairs[1..pairs.len() - 1].iter().position(|p| p.1 == end) {
                let next = i + 2;
                output.push((pairs[next].0, end));
                work.pair()?;
                bits |= ((state.waiting_bits >> next) & 1) << (output.len() - 1);
            }
        }
        let mut predecessor = 1e12;
        if bits & 2 != 0 && (forward || (!linear && !(prefix && position == 0))) {
            predecessor = if first_segment == 1 {
                state.predecessor
            } else {
                pairs[first_segment - 1].1
            };
        }
        let (mut minimum, mut maximum) = (output[0].0, output[0].0);
        let (mut value, mut coordinate) = output[0];
        for &(slope, endpoint) in &output[1..] {
            work.charge(1)?;
            value = arithmetic(
                value + arithmetic((endpoint - coordinate) * slope, "paired split key segment")?,
                "paired split key accumulation",
            )?;
            if value < minimum {
                minimum = value;
            }
            if maximum < value {
                maximum = value;
            }
            coordinate = endpoint;
        }
        let key = if mode == 4 {
            1e12
        } else if forward {
            minimum
        } else {
            arithmetic(ceiling - maximum, "paired split backward key")?
        };
        if output.len() > 12 || output.len() > budget.max_pairs {
            return Err(FrontierUnknown::PairLimit);
        }
        outputs.push(SplitState {
            pairs: output,
            waiting_bits: bits,
            predecessor,
            key,
        });
        start = if prefix && position == 0 {
            prefix_next_start.unwrap_or(end)
        } else {
            end
        };
    }
    if forward && !linear && halfway < upper && outputs.len() > 1 && !aligned(halfway, upper)? {
        // The reference makes this coordinate-only adjustment AFTER key scans.
        let adjusted = arithmetic(halfway + 2e-6, "paired split halfway endpoint")?;
        let last = outputs.len() - 1;
        outputs[last - 1].pairs.last_mut().unwrap().1 = adjusted;
        outputs[last].pairs[0].1 = adjusted;
    }
    if outputs
        .iter()
        .any(|s| s.pairs.windows(2).any(|p| p[0].1 > p[1].1))
    {
        return Err(failure("paired split produces decreasing coordinates"));
    }
    work.charge(0)?;
    Ok(outputs)
}

#[cfg(test)]
#[path = "../../../../../tests/unit/graph/coupled/descriptor/frontier/split_tests.rs"]
mod tests;
