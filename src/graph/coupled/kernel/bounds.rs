//! Outward bounds on the actual resource join, including signed coefficients.
use super::*;
use crate::core::rounding::{add_down, add_up, multiply_down};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::graph::coupled) struct Range {
    pub lo: f64,
    pub hi: f64,
}
impl Default for Range {
    fn default() -> Self {
        Self::point(0.)
    }
}
impl Range {
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    pub fn unknown() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }
    pub fn finite(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }
    pub fn add(self, b: Self) -> Self {
        match (add_down(self.lo, b.lo), add_up(self.hi, b.hi)) {
            (Some(lo), Some(hi)) => Self { lo, hi },
            _ => Self::unknown(),
        }
    }
    pub fn subtract(self, b: Self) -> Self {
        self.add(Self {
            lo: -b.hi,
            hi: -b.lo,
        })
    }
    pub fn multiply(self, b: Self) -> Self {
        if self == Self::point(0.) || b == Self::point(0.) {
            return Self::point(0.);
        }
        if !self.finite() || !b.finite() {
            return Self::unknown();
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for x in [self.lo, self.hi] {
            for y in [b.lo, b.hi] {
                let (Some(l), Some(h)) = (multiply_down(x, y), multiply_down(-x, y).map(|v| -v))
                else {
                    return Self::unknown();
                };
                lo = lo.min(l);
                hi = hi.max(h);
            }
        }
        Self { lo, hi }
    }
    pub fn divide_positive(self, divisor: f64) -> Self {
        fn next_down(x: f64) -> f64 {
            if x == 0. {
                return -f64::from_bits(1);
            }
            f64::from_bits(if x > 0. {
                x.to_bits() - 1
            } else {
                x.to_bits() + 1
            })
        }
        if !self.finite() || !divisor.is_finite() || divisor <= 0. {
            return Self::unknown();
        }
        let (lo, hi) = (self.lo / divisor, self.hi / divisor);
        if !lo.is_finite() || !hi.is_finite() {
            return Self::unknown();
        }
        Self {
            lo: next_down(lo),
            hi: -next_down(-hi),
        }
    }
    fn include(&mut self, x: f64) {
        self.lo = self.lo.min(x);
        self.hi = self.hi.max(x);
    }
}
/// Bounds an arbitrary ordered floating sum of state-dependent arc charges.
/// The exact-sum interval composes in O(1); a separate absolute sum and term
/// count bound roundoff independently of the order used by suffix replay.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CostEnvelope {
    sum: Range,
    absolute: f64,
    terms: Option<usize>,
}
impl Default for CostEnvelope {
    fn default() -> Self {
        Self {
            sum: Range::point(0.),
            absolute: 0.,
            terms: Some(0),
        }
    }
}
impl CostEnvelope {
    pub(super) fn charge(range: Range) -> Self {
        if range == Range::point(0.) {
            return Self::default();
        }
        Self {
            sum: range,
            absolute: range.lo.abs().max(range.hi.abs()),
            terms: Some(1),
        }
    }
    pub(super) fn combine(&self, other: &Self) -> Self {
        Self {
            sum: self.sum.add(other.sum),
            absolute: add_up(self.absolute, other.absolute).unwrap_or(f64::INFINITY),
            terms: self
                .terms
                .and_then(|a| other.terms.and_then(|b| a.checked_add(b))),
        }
    }
    pub(super) fn range(&self) -> Range {
        let Some(terms) = self.terms else {
            return Range::unknown();
        };
        if terms == 0 {
            return Range::point(0.);
        }
        if terms as u128 > (1u128 << 52) || !self.sum.finite() || !self.absolute.is_finite() {
            return Range::unknown();
        }
        let n = Range::point(terms as f64);
        let relative = n.multiply(Range::point(f64::EPSILON));
        if relative.hi >= 0.5 {
            return Range::unknown();
        }
        // gamma_n <= 2*n*epsilon. The additive term also covers gradual
        // underflow; no reassociation of the actual objective is assumed.
        let error = Range::point(self.absolute)
            .multiply(relative.multiply(Range::point(2.)))
            .add(n.multiply(Range::point(f64::from_bits(1))));
        self.sum.add(Range {
            lo: -error.hi,
            hi: error.hi,
        })
    }
}

fn dot_range(a: &[Range], b: &[Range]) -> Range {
    if a.len() != b.len() {
        return Range::unknown();
    }
    a.iter()
        .zip(b)
        .fold(Range::point(0.), |sum, (&a, &b)| sum.add(a.multiply(b)))
}
// Mixed point/interval dots retain the exact ordered interval arithmetic while
// avoiding temporary Vec<Range> allocations in every completion query.
fn dot_values(coefficients: &[Range], values: &[f64]) -> Range {
    if coefficients.len() != values.len() {
        return Range::unknown();
    }
    coefficients
        .iter()
        .zip(values)
        .fold(Range::point(0.), |sum, (&a, &x)| {
            sum.add(a.multiply(Range::point(x)))
        })
}
fn dot_coefficients(coefficients: &[f64], values: &[Range]) -> Range {
    if coefficients.len() != values.len() {
        return Range::unknown();
    }
    coefficients
        .iter()
        .zip(values)
        .fold(Range::point(0.), |sum, (&a, &x)| {
            sum.add(Range::point(a).multiply(x))
        })
}
fn correction(coefficients: &[Range], numeric: &[Range], initial: &[f64]) -> Range {
    if numeric.len() != initial.len() {
        return Range::unknown();
    }
    if numeric
        .iter()
        .zip(initial)
        .all(|(x, &i)| *x == Range::point(i))
    {
        return Range::point(0.);
    }
    // Enclose the two dots separately: this also bounds their actual rounded
    // evaluations, rather than assuming dot(c,x-i) has identical rounding.
    dot_range(coefficients, numeric).subtract(dot_values(coefficients, initial))
}

/// Direction-wide hulls of retained labels. No path-length/sign assumption is
/// needed for completion bounds; these enclose actual stored floating states.
#[derive(Clone, Debug, Default)]
pub(crate) struct CoupledStateBounds {
    numeric: Vec<Range>,
    coefficients: Vec<Range>,
    selection: Vec<u64>,
    loads: Vec<[Range; 3]>,
    state_costs: Range,
    initial_dot: Option<Range>,
    direction: Option<bool>,
    count: usize,
    invalid: bool,
}
impl CoupledStateBounds {
    pub(crate) fn include(&mut self, state: Option<&CoupledLabelState>) {
        self.include_directed(state, false);
    }
    pub(crate) fn include_reversed(&mut self, state: Option<&CoupledLabelState>) {
        self.include_directed(state, true);
    }
    fn include_directed(&mut self, state: Option<&CoupledLabelState>, reversed: bool) {
        self.initial_dot = None;
        let Some(state) = state else {
            self.invalid = true;
            return;
        };
        if reversed && state.suffix.is_some() {
            self.invalid = true;
            return;
        }
        // A union prefix can represent either orientation. A one-orientation
        // hull cannot certify its infeasibility in both roles.
        if state.reversal_peaks.is_some()
            || !state.slope_fronts.is_empty()
            || !state.slope_relations.is_empty()
        {
            self.invalid = true;
            return;
        }
        let forward = state.suffix.is_none() && !reversed;
        let state_costs = state
            .suffix
            .as_ref()
            .map_or(Range::point(0.), |suffix| suffix.state_costs.range());
        let coefficients = state
            .suffix
            .as_ref()
            .map_or(&[][..], |s| s.coefficients.as_slice());
        if self.count == 0 {
            self.direction = Some(forward);
            self.state_costs = state_costs;
            self.selection = state.selection.to_vec();
            self.loads = state
                .loads
                .iter()
                .map(|values| values.map(Range::point))
                .collect();
            self.numeric = state.numeric.iter().copied().map(Range::point).collect();
            self.coefficients = coefficients.iter().copied().map(Range::point).collect();
        } else if self.direction != Some(forward)
            || self.numeric.len() != state.numeric.len()
            || self.coefficients.len() != coefficients.len()
            || self.selection.len() != state.selection.len()
            || self.loads.len() != state.loads.len()
        {
            self.invalid = true;
            return;
        } else {
            self.state_costs.lo = self.state_costs.lo.min(state_costs.lo);
            self.state_costs.hi = self.state_costs.hi.max(state_costs.hi);
            for (union, &domain) in self.selection.iter_mut().zip(&state.selection) {
                *union |= domain;
            }
            for (ranges, values) in self.loads.iter_mut().zip(&state.loads) {
                for (range, &value) in ranges.iter_mut().zip(values) {
                    range.include(value);
                }
            }
            for (range, &x) in self.numeric.iter_mut().zip(&state.numeric) {
                range.include(x);
            }
            for (range, &x) in self.coefficients.iter_mut().zip(coefficients) {
                range.include(x);
            }
        }
        self.invalid |= state
            .numeric
            .iter()
            .chain(coefficients)
            .chain(state.loads.iter().flatten())
            .any(|x| !x.is_finite());
        self.count += 1;
    }
}
impl CoupledStateBounds {
    /// Merge already prepared bucket extrema instead of visiting every label
    /// again (including recomputing nonlinear cost envelopes).
    pub(crate) fn include_hull(&mut self, other: &Self) {
        self.initial_dot = None;
        self.invalid |= other.invalid;
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            let invalid = self.invalid;
            *self = other.clone();
            self.invalid |= invalid;
            self.initial_dot = None;
            return;
        }
        if self.direction != other.direction
            || self.numeric.len() != other.numeric.len()
            || self.coefficients.len() != other.coefficients.len()
            || self.loads.len() != other.loads.len()
            || self.selection.len() != other.selection.len()
        {
            self.invalid = true;
            return;
        }
        for (a, b) in self
            .numeric
            .iter_mut()
            .zip(&other.numeric)
            .chain(self.coefficients.iter_mut().zip(&other.coefficients))
            .chain(
                self.loads
                    .iter_mut()
                    .flatten()
                    .zip(other.loads.iter().flatten()),
            )
            .chain(std::iter::once((&mut self.state_costs, &other.state_costs)))
        {
            a.lo = a.lo.min(b.lo);
            a.hi = a.hi.max(b.hi);
        }
        for (a, b) in self.selection.iter_mut().zip(&other.selection) {
            *a |= b;
        }
        self.count = self.count.saturating_add(other.count);
    }
    pub(crate) fn label_count(&self) -> usize {
        self.count
    }
}

impl CoupledKernel {
    /// Necessary compatibility with an entire opposite-label hull. False is
    /// proof that no label represented by this hull can join this query.
    pub(crate) fn completion_feasible(
        &self,
        forward: bool,
        state: Option<&CoupledLabelState>,
        opposite: &CoupledStateBounds,
    ) -> bool {
        let Some(state) = state else {
            return true;
        };
        if state.reversal_peaks.is_some()
            || self.has_slope_fronts()
            || opposite.invalid
            || opposite.count == 0
            || opposite.direction != Some(!forward)
            || state.suffix.is_none() != forward
            || self.check_state(state).is_err()
            || state.selection.len() != opposite.selection.len()
            || state.loads.len() != opposite.loads.len()
        {
            return true;
        }
        if state
            .selection
            .iter()
            .zip(&opposite.selection)
            .any(|(domain, union)| domain & union == 0)
        {
            return false;
        }
        let mut index = 0;
        for definition in self.definitions.iter() {
            if let CoupledResource::SimultaneousPickupDelivery {
                capacity,
                tolerance,
                ..
            } = definition
            {
                let own = state.loads[index].map(Range::point);
                let other = opposite.loads[index];
                let (f, b) = if forward { (own, other) } else { (other, own) };
                let deliveries = f[0].add(b[0]);
                let net_peak = f[1].add(b[2]);
                let lower_peak = f[2].lo.max(net_peak.lo);
                if add_down(deliveries.lo, lower_peak)
                    .is_some_and(|lower| lower > capacity + tolerance)
                {
                    return false;
                }
                index += 1;
            }
        }
        if !forward {
            let suffix = state.suffix.as_ref().unwrap();
            for bound in &suffix.bounds {
                let value = Range::point(bound.offset)
                    .add(dot_coefficients(&bound.coefficients, &opposite.numeric));
                if value.hi < bound.bounds.lower || value.lo > bound.bounds.upper {
                    return false;
                }
            }
        }
        true
    }
    pub(crate) fn completion_correction_lower(
        &self,
        forward: bool,
        state: Option<&CoupledLabelState>,
        opposite: &CoupledStateBounds,
    ) -> f64 {
        if !self.front_indices.is_empty()
            || self.has_slope_fronts()
            || !self.relation_indices.is_empty()
            || !self.plugins.is_empty()
        {
            return f64::NEG_INFINITY;
        }
        if !self.has_costs() {
            return 0.;
        }
        let Some(state) = state else {
            return f64::NEG_INFINITY;
        };
        if opposite.invalid
            || opposite.count == 0
            || opposite.direction != Some(!forward)
            || state.suffix.is_none() != forward
            || self.check_state(state).is_err()
        {
            return f64::NEG_INFINITY;
        }
        let range = if forward {
            if state.numeric.as_slice() == self.initial {
                Range::point(0.)
            } else {
                dot_values(&opposite.coefficients, &state.numeric).subtract(
                    opposite
                        .initial_dot
                        .unwrap_or_else(|| dot_values(&opposite.coefficients, &self.initial)),
                )
            }
        } else if opposite.numeric.len() == self.initial.len()
            && opposite
                .numeric
                .iter()
                .zip(&self.initial)
                .all(|(x, &i)| *x == Range::point(i))
        {
            Range::point(0.)
        } else {
            let coefficients = &state.suffix.as_ref().unwrap().coefficients;
            let initial = coefficients
                .iter()
                .zip(&self.initial)
                .fold(Range::point(0.), |sum, (&a, &x)| {
                    sum.add(Range::point(a).multiply(Range::point(x)))
                });
            dot_coefficients(coefficients, &opposite.numeric).subtract(initial)
        };
        let nonlinear = if forward {
            opposite.state_costs
        } else {
            state.suffix.as_ref().unwrap().state_costs.range()
        };
        let range = range.add(nonlinear);
        if range.lo.is_finite() {
            range.lo
        } else {
            f64::NEG_INFINITY
        }
    }
    pub(super) fn backward_penalty(&self, a: &Suffix, b: &Suffix) -> Option<f64> {
        if a.coefficients == b.coefficients {
            return Some(0.);
        }
        let numeric = &self.optimization.numeric;
        let ca: Vec<_> = a.coefficients.iter().copied().map(Range::point).collect();
        let cb: Vec<_> = b.coefficients.iter().copied().map(Range::point).collect();
        let separate = correction(&ca, numeric, &self.initial)
            .subtract(correction(&cb, numeric, &self.initial))
            .hi;
        // Both suffixes see the same entry state x, so in real arithmetic
        // the gap is sum((a_i - b_i) * (x_i - initial_i)): the reference's
        // max(0, x_a - x_b) * bound for a cumulative source bound. Each
        // suffix's binary64 dot product adds at most (n + 2) * eps * sum|c x|.
        let correlated = || {
            let mut gap = Range::point(0.);
            let mut magnitude = Range::point(0.);
            for (((&ai, &bi), x), &init) in a
                .coefficients
                .iter()
                .zip(&b.coefficients)
                .zip(numeric)
                .zip(&self.initial)
            {
                let delta = Range::point(ai).subtract(Range::point(bi));
                gap = gap.add(delta.multiply(x.subtract(Range::point(init))));
                let size = Range::point(x.lo.abs().max(x.hi.abs()).max(init.abs()));
                magnitude = magnitude
                    .add(Range::point(ai.abs()).multiply(size))
                    .add(Range::point(bi.abs()).multiply(size));
            }
            let eps = (numeric.len() as f64 + 2.) * f64::EPSILON;
            gap.add(magnitude.multiply(Range::point(2. * eps))).hi
        };
        let gap = separate.min(correlated());
        gap.is_finite().then_some(gap.max(0.))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PreparedBounds {
    pub numeric: Vec<Range>,
    pub weights: Vec<Range>,
    pub dependency_order: super::dependency_order::DependencyOrder,
    pub exact_costs: bool,
}
impl PreparedBounds {
    pub fn new<const D: usize>(
        g: &Graph<D>,
        definitions: &[CoupledResource],
        slots: &[Option<usize>],
        initial: &[f64],
    ) -> Self {
        // Full elementarity only bounds physical visits when every nonterminal
        // owns an element. Group-free vertices may otherwise repeat forever.
        let needs_hop_bounds = definitions.iter().any(|definition| {
            matches!(
                definition,
                CoupledResource::CumulativeCost { .. } | CoupledResource::CustomResource { .. }
            )
        });
        let simple_hops = g.vertices.len().saturating_sub(1);
        let elementary_hops = (needs_hop_bounds
            && g.elementary
            && g.vertices
                .iter()
                .enumerate()
                .all(|(v, x)| v == g.source || v == g.sink || !x.elements.is_empty()))
        .then_some(simple_hops);
        let mut hops = [g.max_route_arcs(), elementary_hops]
            .into_iter()
            .flatten()
            .reduce(usize::min);
        // Only scan the physical topology when cumulative resources need a
        // finite hop proof and the existing bound could be improved by a DAG.
        if needs_hop_bounds && hops.map_or(true, |bound| bound > simple_hops) && g.is_acyclic() {
            hops = Some(simple_hops);
        }
        // Use a floating count only when it is exact. Larger route caps retain
        // conservative unbounded domains instead of underestimating a visit
        // count after an integer-to-f64 conversion.
        let hop_count = hops.and_then(|count| {
            let value = count as f64;
            ((value as u128) == count as u128).then_some(value)
        });
        let mut numeric: Vec<_> = initial.iter().copied().map(Range::point).collect();
        let mut weights = vec![Range::point(0.); initial.len()];
        let mut exact_costs = true;
        let mut total_cost = Range::point(0.);
        for (r, definition) in definitions.iter().enumerate() {
            match definition {
                CoupledResource::CumulativeCost {
                    increments,
                    weights: w,
                    source_upper,
                    ..
                }
                | CoupledResource::CustomResource {
                    increments,
                    weights: w,
                    backward_bound: source_upper,
                    ..
                } => {
                    let i = slots[r].unwrap();
                    let initial = initial[i];
                    let (min, max) = increments
                        .iter()
                        .fold((0f64, 0f64), |(min, max), &x| (min.min(x), max.max(x)));
                    let integral =
                        initial.fract() == 0. && increments.iter().all(|x| x.fract() == 0.);
                    let domain = hop_count.map(|count| {
                        let exact_sum = Range::point(initial)
                            .add(Range { lo: min, hi: max }.multiply(Range::point(count)));
                        let absolute =
                            Range::point(min.abs().max(max.abs())).multiply(Range::point(count));
                        let relative = Range::point(count).multiply(Range::point(f64::EPSILON));
                        if relative.hi >= 0.5 || !absolute.finite() {
                            return Range::unknown();
                        }
                        // Enclose the h rounded state additions as well as the
                        // exact interval sum. The gamma bound is deliberately
                        // symmetric; it is a pruning domain, not an objective.
                        let error = absolute
                            .multiply(relative.multiply(Range::point(2.)))
                            .add(Range::point(count).multiply(Range::point(f64::from_bits(1))));
                        exact_sum.add(Range {
                            lo: -error.hi,
                            hi: error.hi,
                        })
                    });
                    let finite_domain = domain.filter(|range| range.finite());
                    let exact = integral
                        && finite_domain
                            .is_some_and(|d| d.lo.abs().max(d.hi.abs()) <= 4_503_599_627_370_496.);
                    numeric[i] = if let Some(domain) = finite_domain {
                        domain
                    } else {
                        Range {
                            lo: if min >= 0. {
                                initial
                            } else {
                                f64::NEG_INFINITY
                            },
                            hi: if max <= 0. { initial } else { f64::INFINITY },
                        }
                    };
                    // The source bound promises that no prefix adds more than
                    // it, which bounds a backward label's entry state (the
                    // reference's backward max(0, x_a - x_b) * bound).
                    if let Some(k) = source_upper {
                        if min >= 0. && w.iter().all(|&x| x >= 0.) {
                            if let Some(hi) = crate::core::rounding::add_up(initial, *k) {
                                numeric[i].hi = numeric[i].hi.min(hi);
                            }
                        }
                    }
                    let (min, max) = w
                        .iter()
                        .fold((0f64, 0f64), |(min, max), &x| (min.min(x), max.max(x)));
                    weights[i] = hop_count.map_or(
                        Range {
                            lo: if min == 0. { 0. } else { f64::NEG_INFINITY },
                            hi: if max == 0. { 0. } else { f64::INFINITY },
                        },
                        |h| Range { lo: min, hi: max }.multiply(Range::point(h)),
                    );
                    let exact = exact && w.iter().all(|x| x.fract() == 0.);
                    exact_costs &= exact;
                    if let (true, Some(h)) = (exact, hop_count) {
                        let abs = numeric[i].lo.abs().max(numeric[i].hi.abs());
                        total_cost = total_cost.add(
                            Range::point(abs)
                                .multiply(Range::point(min.abs().max(max.abs())))
                                .multiply(Range::point(h)),
                        );
                    }
                }
                CoupledResource::Dependent { bounds, .. }
                | CoupledResource::PiecewiseLinearDependent { bounds, .. }
                | CoupledResource::PiecewiseLinearResource { bounds, .. }
                | CoupledResource::EncodedPiecewiseLinearResource { bounds, .. }
                | CoupledResource::EncodedPiecewiseLinearDependent { bounds, .. } => {
                    numeric[slots[r].unwrap()] = Range {
                        lo: bounds.lower,
                        hi: bounds.upper,
                    };
                }
                _ => {}
            }
        }
        exact_costs &= total_cost.finite() && total_cost.hi <= 4_503_599_627_370_496.;
        let dependency_order =
            super::dependency_order::DependencyOrder::prepare(definitions, slots, &numeric);
        Self {
            numeric,
            weights,
            dependency_order,
            exact_costs,
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/bounds/coupled_hull_tests.rs"]
mod coupled_hull_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/bounds/cost_envelope_tests.rs"]
mod cost_envelope_tests;

/// A necessary resource relation for all pairs represented by two hulls. It
/// cannot certify a route and is usable only for a query contained in its hull.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoupledPairSummary {
    pub(crate) possible: bool,
    pub(crate) correction_lower: f64,
}
impl Default for CoupledPairSummary {
    fn default() -> Self {
        Self {
            possible: true,
            correction_lower: f64::NEG_INFINITY,
        }
    }
}
impl CoupledStateBounds {
    pub(crate) fn prepare(&mut self, kernel: &CoupledKernel) {
        self.initial_dot = Some(dot_values(&self.coefficients, &kernel.initial));
    }
    pub(crate) fn usable(&self) -> bool {
        !self.invalid && self.count > 0
    }
    pub(crate) fn contains(&self, state: Option<&CoupledLabelState>) -> bool {
        let Some(state) = state else {
            return false;
        };
        let coefficients = state
            .suffix
            .as_ref()
            .map_or(&[][..], |s| s.coefficients.as_slice());
        let costs = state
            .suffix
            .as_ref()
            .map_or(Range::point(0.), |s| s.state_costs.range());
        let within = |r: &Range, x: f64| x.is_finite() && r.lo <= x && x <= r.hi;
        self.usable()
            && self.direction == Some(state.suffix.is_none())
            && self.numeric.len() == state.numeric.len()
            && self.coefficients.len() == coefficients.len()
            && self.selection.len() == state.selection.len()
            && self.loads.len() == state.loads.len()
            && self
                .numeric
                .iter()
                .zip(&state.numeric)
                .all(|(r, &x)| within(r, x))
            && self
                .coefficients
                .iter()
                .zip(coefficients)
                .all(|(r, &x)| within(r, x))
            && self
                .selection
                .iter()
                .zip(&state.selection)
                .all(|(&union, &x)| x & !union == 0)
            && self
                .loads
                .iter()
                .zip(&state.loads)
                .all(|(r, x)| r.iter().zip(x).all(|(r, &x)| within(r, x)))
            && costs.lo >= self.state_costs.lo
            && costs.hi <= self.state_costs.hi
    }
}
impl CoupledKernel {
    pub(crate) fn pairing_summary(
        &self,
        a: &CoupledStateBounds,
        b: &CoupledStateBounds,
    ) -> CoupledPairSummary {
        if !self.front_indices.is_empty()
            || self.has_slope_fronts()
            || !self.relation_indices.is_empty()
            || !self.plugins.is_empty()
        {
            return CoupledPairSummary::default();
        }
        if !a.usable() || !b.usable() || a.direction == b.direction {
            return CoupledPairSummary::default();
        }
        let (f, b) = if a.direction == Some(true) {
            (a, b)
        } else {
            (b, a)
        };
        if f.direction != Some(true)
            || b.direction != Some(false)
            || f.selection.len() != b.selection.len()
            || f.loads.len() != b.loads.len()
        {
            return CoupledPairSummary::default();
        }
        let mut possible = !f
            .selection
            .iter()
            .zip(&b.selection)
            .any(|(&x, &y)| x & y == 0);
        let mut index = 0;
        for definition in self.definitions.iter() {
            if let CoupledResource::SimultaneousPickupDelivery {
                capacity,
                tolerance,
                ..
            } = definition
            {
                let (Some(f), Some(b)) = (f.loads.get(index), b.loads.get(index)) else {
                    return CoupledPairSummary::default();
                };
                let deliveries = f[0].add(b[0]);
                let peak = f[2].lo.max(f[1].add(b[2]).lo);
                if add_down(deliveries.lo, peak).is_some_and(|lower| lower > capacity + tolerance) {
                    possible = false;
                }
                index += 1;
            }
        }
        let correction_lower = if self.has_costs() {
            let range = correction(&b.coefficients, &f.numeric, &self.initial).add(b.state_costs);
            if range.lo.is_finite() {
                range.lo
            } else {
                f64::NEG_INFINITY
            }
        } else {
            0.
        };
        CoupledPairSummary {
            possible,
            correction_lower,
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/bounds/pairing_tests.rs"]
mod pairing_tests;

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/bounds/merge_tests.rs"]
mod merge_tests;
