//! Ordered residual-column selection criteria recovered from the selection dispatcher.
use super::*;

pub(super) fn validate(criteria: &[i32]) -> Result<()> {
    if criteria
        .iter()
        .any(|token| !matches!(token, 0..=13 | 15..=17))
    {
        return Err(invalid(
            "column selection criteria must be strategy tokens 0..13 or 15..17",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Column {
    tag: Variable,
    value: f64,
    cost: f64,
    priority: f64,
    reduced: f64,
    guidance: f64,
    greedy: f64,
    terminal_resource: Option<f64>,
    down: bool,
    up: bool,
}

fn less(a: f64, b: f64) -> bool {
    a < b - (1e-6 + 1e-10 * a.abs().max(b.abs()))
}
fn equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-6 * a.abs().max(b.abs())
}

fn initial(token: i32) -> f64 {
    match token {
        0..=2 | 15..=17 => 0.,
        3 | 4 | 6..=8 => 2.,
        _ => 1e12,
    }
}

impl Column {
    fn score(&self, token: i32, quantity: f64) -> Option<f64> {
        let delta = quantity - self.value;
        match token {
            0 | 1 => Some(-1.),
            2 => Some(-self.priority),
            3 => Some(1. - delta.abs()),
            4 => Some(delta.abs()),
            5 | 7 | 12 | 16 => Some(delta.abs() * self.cost),
            6 if quantity >= self.value => Some(delta),
            6 => Some(2.),
            8 if (delta < 0. && !less(self.value, self.guidance))
                || (delta > 0. && !less(self.guidance, self.value)) =>
            {
                Some((self.value - self.guidance).abs())
            }
            9 => Some(delta * self.cost),
            10 => Some(delta * self.reduced),
            11 => Some(if delta < 0. {
                -self.greedy
            } else {
                self.greedy
            }),
            15 if delta > 0. => self.terminal_resource.map(|r| -self.value.sqrt() * r),
            // The shipped comparator has no selectable branch for these tokens.
            13 | 17 => None,
            _ => None,
        }
    }

    fn proposal(&self, token: i32) -> Option<(f64, f64)> {
        // suitableToFixValue rejects a zero contribution. Bounds and original
        // route feasibility are checked again by the child-fixing operation.
        let floor = (self.value + self.value * 1e-10 + 1e-6).floor();
        let ceil = (self.value - self.value * 1e-10 - 1e-6).ceil();
        let mut best: Option<(f64, f64)> = None;
        for (quantity, allowed) in [(floor, self.down), (ceil, self.up)] {
            if !allowed || quantity.abs() <= 1e-6 {
                continue;
            }
            let Some(score) = self.score(token, quantity) else {
                continue;
            };
            if best.as_ref().map_or(true, |b| less(score, b.0)) {
                best = Some((score, quantity));
            }
        }
        best
    }
}

fn select(columns: &[Column], criteria: &[i32]) -> Option<(Variable, f64)> {
    let mut scores: Vec<_> = criteria.iter().map(|&token| initial(token)).collect();
    let mut selected = None;
    for column in columns {
        for (index, &token) in criteria.iter().enumerate() {
            let Some((score, quantity)) = column.proposal(token) else {
                break;
            };
            // Closest2RoundUp updates its up score, but a suitable down
            // contribution returns a tie with the floor, even after that update.
            if token == 6 && column.down {
                if less(score, scores[index]) {
                    scores[index] = score;
                }
                continue;
            }
            if less(score, scores[index]) {
                scores[index] = score;
                selected = Some((column.tag, quantity));
                for next in index + 1..criteria.len() {
                    scores[next] = column
                        .score(criteria[next], quantity)
                        .unwrap_or_else(|| initial(criteria[next]));
                }
                break;
            }
            if !equal(score, scores[index]) {
                break;
            }
        }
    }
    selected
}

/// Residual coverage at quantity one, including active nonlinear rows already
/// materialized in the master column. Equality/upper oversatisfaction rejects
/// the column; lower rows cap their contribution at the remaining demand.
pub(super) fn rank<const D: usize>(
    node: &NodeSnapshot<D>,
    candidates: Vec<(Variable, f64)>,
    config: &DivingOptions,
    guidance: Option<&IntegerWitness>,
    residuals: &Residuals,
) -> Result<Vec<(Variable, f64)>> {
    let criteria = &config.column_selection_criteria;
    validate(criteria)?;
    let mut columns = Vec::with_capacity(candidates.len());
    for (tag, value) in candidates {
        let (position, metadata) = node
            .warm
            .columns
            .iter()
            .enumerate()
            .filter_map(|(p, c)| c.as_ref().map(|c| (p, c)))
            .find(|(_, c)| match tag {
                Variable::Route(tag) => c.tag == tag as i32,
                Variable::Master(index) => master_variables::decode(c.tag) == Some((index, 1.)),
            })
            .ok_or_else(|| invalid("diving selection column identity"))?;
        let (incumbent, fixed, priority, terminal_resource) = match tag {
            Variable::Route(tag) => {
                let route = &node.state.routes[tag];
                let identity = route.route.identity();
                let incumbent = guidance.map_or(0., |w| {
                    w.1.iter()
                        .filter(|r| {
                            r.route.network == route.network && r.route.route.identity() == identity
                        })
                        .map(|r| r.quantity)
                        .sum::<f64>()
                });
                let fixed = node
                    .state
                    .fixed_routes
                    .iter()
                    .filter(|r| {
                        r.route.network == route.network && r.route.route.identity() == identity
                    })
                    .map(|r| r.quantity)
                    .sum();
                (
                    incumbent,
                    fixed,
                    config
                        .network_column_priorities
                        .get(&route.network)
                        .copied()
                        .unwrap_or(1.),
                    Some(
                        route
                            .route
                            .resource_profile
                            .last()
                            .and_then(|r| r.first())
                            .copied()
                            .unwrap_or(0.),
                    ),
                )
            }
            Variable::Master(index) => {
                let variable = &node.state.master_variables[index];
                let incumbent = guidance
                    .and_then(|w| w.2.iter().find(|&&(id, _)| id == variable.id))
                    .map_or(0., |&(_, x)| x);
                (
                    incumbent,
                    residuals.get(&index).copied().unwrap_or(0.),
                    config
                        .master_variable_priorities
                        .get(&variable.id)
                        .copied()
                        .unwrap_or(1.),
                    None,
                )
            }
        };
        let suitable = |quantity| {
            let mut residuals = residuals.clone();
            fix_variable(node, tag, quantity, &mut residuals).is_ok()
        };
        columns.push(Column {
            tag,
            value,
            cost: node.master.costs[position],
            priority,
            reduced: node.master.reduced_costs()[position],
            guidance: incumbent - fixed,
            greedy: if criteria.contains(&11) {
                residual_greedy_cost(node, position, &metadata.entries, residuals)
            } else {
                0.
            },
            terminal_resource,
            down: suitable(value.floor()),
            up: suitable(value.ceil()),
        });
    }
    let mut ranked = Vec::new();
    while let Some((tag, quantity)) = select(&columns, criteria) {
        ranked.push((tag, quantity));
        columns.retain(|c| c.tag != tag);
    }
    Ok(ranked)
}

fn residual_greedy_cost<const D: usize>(
    node: &NodeSnapshot<D>,
    column: usize,
    entries: &[(usize, f64)],
    residuals: &Residuals,
) -> f64 {
    let mut contribution = 0.;
    for &(row, coefficient) in entries {
        if node.state.retired_rows.contains(&row)
            || node.state.master_variable_bound_rows.contains(&row)
        {
            continue;
        }
        let mut rhs = node.state.rhs[row];
        if let Some(shared) = row.checked_sub(node.state.shared_offset) {
            for (&index, &offset) in residuals {
                for &(r, a) in &node.state.master_variables[index].entries {
                    if r == shared {
                        rhs -= a * offset;
                    }
                }
            }
        }
        match node.master.senses[row] {
            RowSense::Equal if less(rhs, coefficient) => return 1e12,
            RowSense::Equal => contribution += coefficient,
            RowSense::LessEqual if less(rhs, coefficient) => return 1e12,
            RowSense::LessEqual => {}
            RowSense::GreaterEqual if less(rhs, coefficient) => contribution += rhs.max(0.),
            RowSense::GreaterEqual => contribution += coefficient,
        }
    }
    if contribution > 1e-6 {
        node.master.costs[column] / contribution
    } else {
        1e12
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/master/tree/column_selection_tests.rs"]
mod tests;
