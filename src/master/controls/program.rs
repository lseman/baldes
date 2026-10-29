//! Exact finite source regions for paired PWL and max/clamp programs.
use super::{exact_add, exact_mul, exact_sub, Budget};
use crate::graph::coupled::{CoupledResource, PiecewiseLinearFunction, PreparedSlopeFunction};
use crate::graph::Window;
use crate::master::{
    preprocessing::{Bounds, Constraint},
    RowSense,
};
/// (slope*x+intercept)/denominator on a closed segment. Knot slopes need
/// not be representable: positive denominator scaling preserves their relation.
#[derive(Clone, Copy)]
struct Segment {
    lo: f64,
    hi: f64,
    slope: f64,
    intercept: f64,
    denominator: f64,
}
impl Segment {
    fn image(self, x: f64) -> Option<(f64, f64)> {
        let lo = crate::core::rounding::add_down(
            crate::core::rounding::multiply_down(self.slope, x)?,
            self.intercept,
        )?;
        let hi = crate::core::rounding::add_up(
            -crate::core::rounding::multiply_down(-self.slope, x)?,
            self.intercept,
        )?;
        // Division is used only for an outer variable box, never a row.
        let down = lo / self.denominator;
        let up = hi / self.denominator;
        if !down.is_finite() || !up.is_finite() {
            return None;
        }
        let next_down = |x: f64| {
            if x == 0. {
                -f64::from_bits(1)
            } else {
                f64::from_bits(if x > 0. {
                    x.to_bits() - 1
                } else {
                    x.to_bits() + 1
                })
            }
        };
        Some((
            if self.denominator == 1. || exact_mul(down, self.denominator) == Some(lo) {
                down
            } else {
                next_down(down)
            },
            if self.denominator == 1. || exact_mul(up, self.denominator) == Some(hi) {
                up
            } else {
                -next_down(-up)
            },
        ))
    }
    fn range(self) -> Option<Bounds> {
        let (a, b) = self.image(self.lo)?;
        let (c, d) = self.image(self.hi)?;
        Some(Bounds {
            lower: a.min(c),
            upper: b.max(d),
        })
    }
}
fn knots(function: &PiecewiseLinearFunction, budget: &mut Budget) -> Option<Vec<Segment>> {
    budget.take(function.points.len().checked_mul(8)?)?;
    function
        .points
        .windows(2)
        .map(|pair| {
            let ([x0, y0], [x1, y1]) = (pair[0], pair[1]);
            let dx = exact_sub(x1, x0)?;
            let dy = exact_sub(y1, y0)?;
            Some(Segment {
                lo: x0,
                hi: x1,
                slope: dy,
                intercept: exact_sub(exact_mul(dx, y0)?, exact_mul(dy, x0)?)?,
                denominator: dx,
            })
        })
        .collect()
}
fn slopes(function: &PreparedSlopeFunction, budget: &mut Budget) -> Option<Vec<Segment>> {
    budget.take(function.pairs().len().checked_mul(8)?)?;
    let (mut value, mut start) = function.pairs()[0];
    let mut result = vec![];
    if function.pairs().len() == 1 {
        result.push(Segment {
            lo: start,
            hi: start,
            slope: 0.,
            intercept: value,
            denominator: 1.,
        });
    }
    for &(slope, end) in &function.pairs()[1..] {
        result.push(Segment {
            lo: start,
            hi: end,
            slope,
            intercept: exact_sub(value, exact_mul(slope, start)?)?,
            denominator: 1.,
        });
        value = exact_add(value, exact_mul(slope, exact_sub(end, start)?)?)?;
        start = end;
    }
    Some(result)
}
pub(super) struct Program {
    functions: Vec<Vec<Segment>>,
    initial: Window,
    waiting: Vec<i32>,
}
pub(super) fn program(
    definition: &CoupledResource,
    path: &[usize],
    budget: &mut Budget,
) -> Option<Program> {
    let mut functions = vec![];
    let mut waiting = vec![];
    let initial = match definition {
        CoupledResource::PairedPiecewiseLinear {
            initial,
            initial_base,
            arcs,
            ..
        } => {
            functions.push(knots(initial, budget)?);
            for &index in path {
                let arc = arcs.get(index)?;
                for f in [&arc.arrival, &arc.increment, &arc.lower, &arc.upper] {
                    functions.push(knots(f, budget)?);
                }
                waiting.push(arc.waiting);
            }
            *initial_base
        }
        CoupledResource::SlopePaired {
            initial,
            initial_base,
            arcs,
            ..
        } => {
            functions.push(slopes(initial, budget)?);
            for &index in path {
                let arc = arcs.get(index)?;
                for f in [&arc.arrival, &arc.increment, &arc.lower, &arc.upper] {
                    functions.push(slopes(f, budget)?);
                }
                waiting.push(arc.waiting);
            }
            *initial_base
        }
        _ => return None,
    };
    functions.iter().all(|f| !f.is_empty()).then_some(Program {
        functions,
        initial,
        waiting,
    })
}
pub(super) fn repair_arrivals(
    program: &Program,
    choices: &[usize],
    bounds: &[Bounds],
    mut base: Vec<f64>,
    budget: &mut Budget,
) -> Option<Vec<f64>> {
    for i in 0..program.waiting.len() {
        budget.take(8)?;
        let arrival = program.functions[1 + 4 * i][choices[1 + 4 * i]]
            .image(base[i])?
            .1;
        base[i + 1] = base[i + 1]
            .max(arrival)
            .max(bounds[i + 1].lower)
            .min(bounds[i + 1].upper);
    }
    Some(base)
}

/// One advisory control using the full function domains, without selecting an
/// LP region. It may miss controls behind nonmonotone dependent bounds; failure
/// is unknown and the caller must independently certify the complete proposal.
pub(super) fn earliest_base_proposal(program: &Program, budget: &mut Budget) -> Option<Vec<f64>> {
    let n = program.waiting.len();
    budget.take(n.checked_mul(8)?.checked_add(4)?)?;
    let mut bounds = vec![
        Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        };
        n.checked_add(1)?
    ];
    let domain = |function: &[Segment]| -> Option<(f64, f64)> {
        Some((function.first()?.lo, function.last()?.hi))
    };
    let (lo, hi) = domain(&program.functions[0])?;
    restrict(
        &mut bounds[0],
        lo.max(program.initial.lower),
        hi.min(program.initial.upper),
    );
    for i in 0..n {
        for function in &program.functions[1 + 4 * i..3 + 4 * i] {
            let (lo, hi) = domain(function)?;
            restrict(&mut bounds[i], lo, hi);
        }
        for function in &program.functions[3 + 4 * i..5 + 4 * i] {
            let (lo, hi) = domain(function)?;
            restrict(&mut bounds[i + 1], lo, hi);
        }
    }
    if bounds
        .iter()
        .any(|b| !b.lower.is_finite() || !b.upper.is_finite() || b.lower > b.upper)
    {
        return None;
    }
    let mut base = Vec::with_capacity(bounds.len());
    base.push(bounds[0].lower);
    for i in 0..n {
        let arrival = &program.functions[1 + 4 * i];
        budget.take(arrival.len().checked_add(2)?)?;
        let segment = arrival
            .iter()
            .find(|s| s.lo <= base[i] && base[i] <= s.hi)?;
        let head = segment.image(base[i])?.1.max(bounds[i + 1].lower);
        if !head.is_finite() || head > bounds[i + 1].upper {
            return None;
        }
        base.push(head);
    }
    Some(base)
}

pub(super) struct Cell {
    pub(super) bounds: Vec<Bounds>,
    pub(super) rows: Vec<Constraint>,
    pub(super) base_count: usize,
}
fn restrict(bounds: &mut Bounds, lo: f64, hi: f64) {
    bounds.lower = bounds.lower.max(lo);
    bounds.upper = bounds.upper.min(hi);
}
pub(super) fn row(entries: Vec<(usize, f64)>, sense: RowSense, rhs: f64) -> Constraint {
    Constraint {
        entries: entries.into_iter().filter(|(_, a)| *a != 0.).collect(),
        sense,
        rhs,
    }
}
/// Err means unsupported coefficient arithmetic; None proves this cell empty.
pub(super) fn cell(
    program: &Program,
    choices: &[usize],
    terminal: Option<Window>,
    budget: &mut Budget,
) -> std::result::Result<Option<Cell>, ()> {
    let n = program.waiting.len();
    let count = n + 1;
    budget.take(count.checked_mul(48).ok_or(())?).ok_or(())?;
    let mut bounds = vec![
        Bounds {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY
        };
        count * 2
    ];
    let selected: Vec<_> = program
        .functions
        .iter()
        .zip(choices)
        .map(|(f, &i)| f[i])
        .collect();
    let initial = selected[0];
    restrict(
        &mut bounds[0],
        program.initial.lower.max(initial.lo),
        program.initial.upper.min(initial.hi),
    );
    bounds[count] = initial.range().ok_or(())?;
    let mut rows = vec![row(
        vec![(count, initial.denominator), (0, -initial.slope)],
        RowSense::Equal,
        initial.intercept,
    )];
    for i in 0..n {
        let [a, inc, lower, upper]: [Segment; 4] =
            selected[1 + 4 * i..5 + 4 * i].try_into().map_err(|_| ())?;
        restrict(&mut bounds[i], a.lo.max(inc.lo), a.hi.min(inc.hi));
        restrict(
            &mut bounds[i + 1],
            lower.lo.max(upper.lo),
            lower.hi.min(upper.hi),
        );
        bounds[count + i + 1] = Bounds {
            lower: lower.range().ok_or(())?.lower,
            upper: upper.range().ok_or(())?.upper,
        };
        let growth = (|| {
            let w = f64::from(program.waiting[i]);
            let scale = exact_mul(a.denominator, inc.denominator)?;
            let departure = exact_sub(
                exact_mul(inc.slope, a.denominator)?,
                exact_mul(exact_mul(w, a.slope)?, inc.denominator)?,
            )?;
            let rhs = exact_sub(
                exact_mul(inc.intercept, a.denominator)?,
                exact_mul(exact_mul(w, a.intercept)?, inc.denominator)?,
            )?;
            Some(row(
                vec![
                    (count + i + 1, scale),
                    (count + i, -scale),
                    (i + 1, -exact_mul(scale, w)?),
                    (i, -departure),
                ],
                if choices[program.functions.len() + i] == 0 {
                    RowSense::Equal
                } else {
                    RowSense::GreaterEqual
                },
                rhs,
            ))
        })()
        .ok_or(())?;
        rows.push(growth);
        rows.push(row(
            vec![(count + i + 1, lower.denominator), (i + 1, -lower.slope)],
            if choices[program.functions.len() + i] == 1 {
                RowSense::Equal
            } else {
                RowSense::GreaterEqual
            },
            lower.intercept,
        ));
        rows.push(row(
            vec![(count + i + 1, upper.denominator), (i + 1, -upper.slope)],
            RowSense::LessEqual,
            upper.intercept,
        ));
        rows.push(row(
            vec![(i + 1, a.denominator), (i, -a.slope)],
            RowSense::GreaterEqual,
            a.intercept,
        ));
    }
    if let Some(terminal) = terminal {
        restrict(&mut bounds[2 * count - 1], terminal.lower, terminal.upper);
    }
    if bounds.iter().any(|b| b.lower > b.upper) {
        return Ok(None);
    }
    if bounds
        .iter()
        .any(|b| !b.lower.is_finite() || !b.upper.is_finite())
    {
        return Err(());
    }
    Ok(Some(Cell {
        bounds,
        rows,
        base_count: count,
    }))
}

pub(super) fn shifted(cell: &Cell, budget: &mut Budget) -> Option<(Vec<Constraint>, Vec<f64>)> {
    budget.take(
        cell.rows
            .len()
            .checked_mul(cell.bounds.len().checked_add(1)?)?,
    )?;
    let mut rows = vec![];
    let mut widths = vec![];
    for (v, bound) in cell.bounds.iter().enumerate() {
        let width = exact_sub(bound.upper, bound.lower)?;
        widths.push(width);
        rows.push(row(vec![(v, 1.)], RowSense::LessEqual, width));
    }
    for source in &cell.rows {
        let mut rhs = source.rhs;
        for &(v, a) in &source.entries {
            rhs = exact_sub(rhs, exact_mul(a, cell.bounds[v].lower)?)?;
        }
        rows.push(row(source.entries.clone(), source.sense, rhs));
    }
    Some((rows, widths))
}

impl Program {
    pub(super) fn radices(&self) -> Vec<usize> {
        self.functions
            .iter()
            .map(Vec::len)
            .chain(std::iter::repeat(2).take(self.waiting.len()))
            .collect()
    }
}
