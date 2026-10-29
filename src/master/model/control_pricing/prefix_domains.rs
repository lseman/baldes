// Bounded-walk resource extension and paired prefix domains.
use super::*;
pub(super) fn extend_bounded_walk_resources<const D: usize>(
    graph: &crate::graph::Graph<D>,
    arc: usize,
    resources: crate::graph::ResourceValues<D>,
) -> Option<Option<crate::graph::ResourceValues<D>>> {
    Some(crate::labeling::extend_resources(
        graph,
        graph.arcs.get(arc)?,
        resources,
        true,
    ))
}

#[derive(Clone, Copy)]
pub(super) struct PairedPrefixDomain {
    pub(super) resource: usize,
    pub(super) base: Interval,
    pub(super) dependent_lower: f64,
    pub(super) dependent_upper: f64,
}

pub(super) fn paired_prefix_roots<const D: usize>(
    graph: &crate::graph::Graph<D>,
    work: &mut usize,
    deadline: Instant,
) -> Option<Vec<PairedPrefixDomain>> {
    let kernel = graph.coupled_kernel()?;
    let mut roots = Vec::new();
    for (resource, definition) in kernel.definitions().iter().enumerate() {
        let (base, dependent) = match definition {
            CoupledResource::PairedPiecewiseLinear {
                initial,
                initial_base,
                ..
            } => (
                Interval {
                    lo: initial_base.lower,
                    hi: initial_base.upper,
                },
                piecewise_value_range(
                    initial,
                    Interval {
                        lo: initial_base.lower,
                        hi: initial_base.upper,
                    },
                    work,
                    deadline,
                )?,
            ),
            CoupledResource::SlopePaired {
                initial,
                initial_base,
                policy: crate::graph::coupled::FrontOperationPolicy::StrictDomain,
                ..
            } => (
                Interval {
                    lo: initial_base.lower,
                    hi: initial_base.upper,
                },
                slope_value_range(
                    initial,
                    Interval {
                        lo: initial_base.lower,
                        hi: initial_base.upper,
                    },
                    work,
                    deadline,
                )?,
            ),
            CoupledResource::SlopePaired {
                initial,
                initial_base,
                policy: crate::graph::coupled::FrontOperationPolicy::FrontAlgebra,
                ..
            } => {
                let base = intersect_domains(
                    Interval {
                        lo: initial_base.lower,
                        hi: initial_base.upper,
                    },
                    slope_domain_with_endpoint_tolerance(initial)?,
                    slope_domain_with_endpoint_tolerance(initial)?,
                )?;
                (base, slope_function_hull(initial, work, deadline)?)
            }
            _ => continue,
        };
        roots.push(PairedPrefixDomain {
            resource,
            base,
            dependent_lower: dependent.lo,
            dependent_upper: dependent.hi,
        });
    }
    Some(roots)
}

/// Overapproximate the paired base values and lower-bound dependent values
/// reachable after one arc. The recurrence is necessary for every source
/// control: Gamma' >= max(Gamma + increment + waiting, lower), Gamma' <=
/// upper. Function hulls relax base/Gamma correlations; an empty relaxed
/// intersection therefore proves infeasibility. Failed arithmetic disables
/// only this branch's filter.
pub(super) fn extend_paired_prefix_domains<const D: usize>(
    graph: &crate::graph::Graph<D>,
    arc_index: usize,
    current: &[PairedPrefixDomain],
    arc_hulls: &mut HashMap<(usize, usize), [Interval; 4]>,
    work: &mut usize,
    deadline: Instant,
) -> Option<Option<Vec<PairedPrefixDomain>>> {
    let kernel = graph.coupled_kernel()?;
    let mut next = Vec::with_capacity(current.len());
    for &domain in current {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let definition = kernel.definitions().get(domain.resource)?;
        let (
            next_base_lower,
            next_base_upper,
            increment,
            lower,
            upper,
            waiting_lower,
            waiting_upper,
            unbounded_upper,
        ) = match definition {
            CoupledResource::PairedPiecewiseLinear { arcs, mode, .. } => {
                let arc = arcs.get(arc_index)?;
                let departure = intersect_domains(
                    domain.base,
                    piecewise_domain(&arc.arrival)?,
                    piecewise_domain(&arc.increment)?,
                );
                let Some(departure) = departure else {
                    return Some(None);
                };
                let head = intersect_head_domains(
                    piecewise_domain(&arc.lower)?,
                    piecewise_domain(&arc.upper)?,
                );
                let Some(head) = head else {
                    return Some(None);
                };
                let Some(arrival_at_lower_departure) =
                    piecewise_eval_validated(&arc.arrival, departure.lo)
                else {
                    return None;
                };
                let reachable_head = Interval {
                    lo: arrival_at_lower_departure.max(head.lo),
                    hi: head.hi,
                };
                if reachable_head.lo > reachable_head.hi {
                    return Some(None);
                }
                let increment = piecewise_value_range(&arc.increment, departure, work, deadline)?;
                let lower = piecewise_value_range(&arc.lower, reachable_head, work, deadline)?;
                let upper = piecewise_value_range(&arc.upper, reachable_head, work, deadline)?;
                let maximum_wait =
                    crate::core::rounding::add_up(head.hi, -arrival_at_lower_departure)?.max(0.);
                let waiting_lower = if arc.waiting < 0 {
                    crate::core::rounding::multiply_down(f64::from(arc.waiting), maximum_wait)?
                } else {
                    0.0
                };
                let waiting_upper = if arc.waiting > 0 {
                    multiply_upper_nonnegative(f64::from(arc.waiting), maximum_wait)?
                } else {
                    0.0
                };
                let unbounded_upper = *mode
                    == crate::graph::coupled::PairedResourceMode::CostUnbounded
                    && arc
                        .upper
                        .points
                        .first()
                        .is_some_and(|point| point[1] >= 1e12);
                (
                    arrival_at_lower_departure,
                    head.hi,
                    increment,
                    lower,
                    upper,
                    waiting_lower,
                    waiting_upper,
                    unbounded_upper,
                )
            }
            CoupledResource::SlopePaired {
                arcs,
                policy: crate::graph::coupled::FrontOperationPolicy::StrictDomain,
                ..
            } => {
                let arc = arcs.get(arc_index)?;
                let departure = intersect_domains(
                    domain.base,
                    slope_domain(&arc.arrival)?,
                    slope_domain(&arc.increment)?,
                );
                let Some(departure) = departure else {
                    return Some(None);
                };
                let head =
                    intersect_head_domains(slope_domain(&arc.lower)?, slope_domain(&arc.upper)?);
                let Some(head) = head else {
                    return Some(None);
                };
                let Some(arrival_at_lower_departure) = slope_value_range(
                    &arc.arrival,
                    Interval {
                        lo: departure.lo,
                        hi: departure.lo,
                    },
                    work,
                    deadline,
                )
                .map(|range| range.lo) else {
                    return None;
                };
                let reachable_head = Interval {
                    lo: arrival_at_lower_departure.max(head.lo),
                    hi: head.hi,
                };
                if reachable_head.lo > reachable_head.hi {
                    return Some(None);
                }
                let increment = slope_value_range(&arc.increment, departure, work, deadline)?;
                let lower = slope_value_range(&arc.lower, reachable_head, work, deadline)?;
                let upper = slope_value_range(&arc.upper, reachable_head, work, deadline)?;
                let unbounded_upper = false;
                let maximum_wait =
                    crate::core::rounding::add_up(head.hi, -arrival_at_lower_departure)?.max(0.);
                (
                    arrival_at_lower_departure.max(head.lo),
                    head.hi,
                    increment,
                    lower,
                    upper,
                    if arc.waiting < 0 {
                        crate::core::rounding::multiply_down(f64::from(arc.waiting), maximum_wait)?
                    } else {
                        0.
                    },
                    if arc.waiting > 0 {
                        multiply_upper_nonnegative(f64::from(arc.waiting), maximum_wait)?
                    } else {
                        0.
                    },
                    unbounded_upper,
                )
            }
            CoupledResource::SlopePaired {
                initial: _,
                arcs,
                mode,
                policy: crate::graph::coupled::FrontOperationPolicy::FrontAlgebra,
                ..
            } => {
                let arc = arcs.get(arc_index)?;
                let departure = intersect_domains(
                    domain.base,
                    slope_domain_with_endpoint_tolerance(&arc.arrival)?,
                    slope_domain_with_endpoint_tolerance(&arc.increment)?,
                );
                let Some(_departure) = departure else {
                    return Some(None);
                };
                let head = intersect_head_domains(
                    slope_domain_with_endpoint_tolerance(&arc.lower)?,
                    slope_domain_with_endpoint_tolerance(&arc.upper)?,
                );
                let Some(head) = head else {
                    return Some(None);
                };
                // Endpoint snapping can accept a head just below arrival. Do not
                // impose the strict head >= arrival relation here; use the full
                // interval difference for a conservative waiting enclosure.
                // These four full-domain hulls depend only on the immutable
                // resource and physical arc, not on this prefix. Cyclic walks
                // can revisit the same arc with many different domains.
                // Retain only complete successful results and bound memory per
                // pricing call; all prefix-dependent arithmetic remains below.
                let key = (domain.resource, arc_index);
                let [arrival, increment, lower, upper] = if let Some(&cached) = arc_hulls.get(&key)
                {
                    if Instant::now() >= deadline {
                        return None;
                    }
                    cached
                } else {
                    let hulls = [
                        slope_function_hull(&arc.arrival, work, deadline)?,
                        slope_function_hull(&arc.increment, work, deadline)?,
                        slope_function_hull(&arc.lower, work, deadline)?,
                        slope_function_hull(&arc.upper, work, deadline)?,
                    ];
                    if arc_hulls.len() < 4096 {
                        arc_hulls.insert(key, hulls);
                    }
                    hulls
                };
                let head_minus_arrival = Interval {
                    lo: crate::core::rounding::add_down(head.lo, -arrival.hi)?,
                    hi: crate::core::rounding::add_up(head.hi, -arrival.lo)?,
                };
                let waiting =
                    Interval::point(f64::from(arc.waiting)).multiply(head_minus_arrival)?;
                let unbounded_upper = *mode
                    == crate::graph::coupled::PairedResourceMode::CostUnbounded
                    && arc
                        .upper
                        .pairs()
                        .first()
                        .is_some_and(|point| point.0 >= 1e12);
                (
                    head.lo,
                    head.hi,
                    increment,
                    lower,
                    upper,
                    waiting.lo,
                    waiting.hi,
                    unbounded_upper,
                )
            }
            // The root constructor only installs supported paired domains.
            _ => return None,
        };
        let after_increment =
            crate::core::rounding::add_down(domain.dependent_lower, increment.lo)?;
        let recurrence_lower = crate::core::rounding::add_down(after_increment, waiting_lower)?;
        let dependent_lower = recurrence_lower.max(lower.lo);
        if !unbounded_upper && dependent_lower > upper.hi {
            return Some(None);
        }
        let recurrence_upper = crate::core::rounding::add_up(
            crate::core::rounding::add_up(domain.dependent_upper, increment.hi)?,
            waiting_upper,
        )?
        .max(lower.hi);
        let dependent_upper = if unbounded_upper {
            recurrence_upper
        } else {
            recurrence_upper.min(upper.hi)
        };
        next.push(PairedPrefixDomain {
            resource: domain.resource,
            base: Interval {
                lo: next_base_lower,
                hi: next_base_upper,
            },
            dependent_lower,
            dependent_upper,
        });
    }
    Some(Some(next))
}

pub(super) fn intersect_domains(
    current: Interval,
    first: (f64, f64),
    second: (f64, f64),
) -> Option<Interval> {
    let lower = current.lo.max(first.0).max(second.0);
    let upper = current.hi.min(first.1).min(second.1);
    (lower <= upper).then_some(Interval {
        lo: lower,
        hi: upper,
    })
}

pub(super) fn intersect_head_domains(first: (f64, f64), second: (f64, f64)) -> Option<Interval> {
    let lower = first.0.max(second.0);
    let upper = first.1.min(second.1);
    (lower <= upper).then_some(Interval {
        lo: lower,
        hi: upper,
    })
}

pub(super) fn piecewise_domain(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
) -> Option<(f64, f64)> {
    Some((
        function.points.first()?.first().copied()?,
        function.points.last()?.first().copied()?,
    ))
}

pub(super) fn piecewise_eval_validated(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
    argument: f64,
) -> Option<f64> {
    let points = &function.points;
    let first = points.first()?;
    let last = points.last()?;
    if !argument.is_finite() || argument < first[0] || argument > last[0] {
        return None;
    }
    let upper = points.partition_point(|point| point[0] < argument);
    let point = points.get(upper)?;
    if point[0] == argument {
        return Some(point[1]);
    }
    let left = points.get(upper.checked_sub(1)?)?;
    let span = argument - left[0];
    let width = point[0] - left[0];
    let rise = point[1] - left[1];
    if !span.is_finite() || !width.is_finite() || !rise.is_finite() {
        return None;
    }
    let fraction = span / width;
    let scaled = fraction * rise;
    let value = left[1] + scaled;
    (fraction.is_finite() && scaled.is_finite() && value.is_finite())
        .then(|| value.clamp(left[1].min(point[1]), left[1].max(point[1])))
}

/// First knot at or above `coordinate`, or strictly above when `after` is set.
/// Coordinates have already been validated as strictly increasing and finite.
pub(super) fn piecewise_knot_bound(
    points: &[[f64; 2]],
    coordinate: f64,
    after: bool,
    work: &mut usize,
    deadline: Instant,
) -> Option<usize> {
    let mut left = 0;
    let mut right = points.len();
    while left < right {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let middle = left + (right - left) / 2;
        let precedes = if after {
            points[middle][0] <= coordinate
        } else {
            points[middle][0] < coordinate
        };
        if precedes {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    Some(left)
}

/// Exact hull of the validated knot evaluator on a closed subdomain. The
/// evaluator is monotone on each segment and clamps rounded interpolation to
/// its segment endpoints, so extrema occur at the query endpoints or knots.
pub(super) fn piecewise_value_range(
    function: &crate::graph::coupled::PiecewiseLinearFunction,
    domain: Interval,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let (lower, upper) = piecewise_domain(function)?;
    if domain.lo < lower || domain.hi > upper || domain.lo > domain.hi {
        return None;
    }
    *work = work.checked_sub(2)?;
    if Instant::now() >= deadline {
        return None;
    }
    let mut lo = piecewise_eval_validated(function, domain.lo)?;
    let mut hi = piecewise_eval_validated(function, domain.hi)?;
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    // Small functions and full-domain queries already need the whole scan.
    // For longer partial-domain queries, binary bounds retain exactly the
    // same included knots and order while skipping unrelated coordinates.
    let (start, end) = if function.points.len() <= 16 || (domain.lo == lower && domain.hi == upper)
    {
        (0, function.points.len())
    } else {
        (
            piecewise_knot_bound(&function.points, domain.lo, false, work, deadline)?,
            piecewise_knot_bound(&function.points, domain.hi, true, work, deadline)?,
        )
    };
    for point in &function.points[start..end] {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        if domain.lo <= point[0] && point[0] <= domain.hi {
            lo = lo.min(point[1]);
            hi = hi.max(point[1]);
        }
    }
    Some(Interval { lo, hi })
}

/// Directed hull of a prepared raw-slope evaluator over a subdomain. The
/// function is affine between stored coordinates; interval replay preserves
/// each sequential segment operation and encloses rounded endpoint values.
pub(super) fn slope_value_range(
    function: &crate::graph::coupled::PreparedSlopeFunction,
    domain: Interval,
    work: &mut usize,
    deadline: Instant,
) -> Option<Interval> {
    let pairs = function.pairs();
    let &(initial, start) = pairs.first()?;
    let end = pairs.last()?.1;
    if domain.lo < start || domain.hi > end || domain.lo > domain.hi {
        return None;
    }
    if pairs.len() == 1 {
        return (domain.lo == start && domain.hi == start).then_some(Interval::point(initial));
    }
    let mut value = Interval::point(initial);
    let mut previous = start;
    let mut hull = Interval {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for &(slope, segment_end) in pairs.iter().skip(1) {
        *work = work.checked_sub(1)?;
        if Instant::now() >= deadline {
            return None;
        }
        let clipped_lo = domain.lo.max(previous);
        let clipped_hi = domain.hi.min(segment_end);
        if clipped_lo <= clipped_hi {
            let lo_span = clipped_lo - previous;
            let hi_span = clipped_hi - previous;
            if !lo_span.is_finite() || !hi_span.is_finite() {
                return None;
            }
            let lo_value = value.add(Interval::point(slope).multiply(Interval::point(lo_span))?)?;
            let hi_value = value.add(Interval::point(slope).multiply(Interval::point(hi_span))?)?;
            hull.lo = hull.lo.min(lo_value.lo).min(hi_value.lo);
            hull.hi = hull.hi.max(lo_value.hi).max(hi_value.hi);
            if domain.hi <= segment_end {
                break;
            }
        }
        let span = segment_end - previous;
        if !span.is_finite() || span <= 0. {
            return None;
        }
        value = value.add(Interval::point(slope).multiply(Interval::point(span))?)?;
        previous = segment_end;
    }
    (hull.lo.is_finite() && hull.hi.is_finite()).then_some(hull)
}

pub(super) fn slope_domain(
    function: &crate::graph::coupled::PreparedSlopeFunction,
) -> Option<(f64, f64)> {
    Some(function.domain())
}
