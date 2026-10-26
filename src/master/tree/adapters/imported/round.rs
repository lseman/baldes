//! Source batches at a common stored pricing point, with atomic tree rollback.
use super::*;

pub(super) struct Collected {
    pub status: Option<TreeStatus>,
    pub batches: usize,
    pub added: usize,
    pub completed: usize,
    pub numerical_bound: Option<f64>,
}
pub(super) fn collect<const D: usize, F>(
    pending: &mut Pending<D>,
    broadcast: &mut impl FnMut(&Pending<D>),
    point: &ImportPricing<D>,
    max_batches: usize,
    deadline: Instant,
    tolerance: f64,
    all: bool,
    numerical: bool,
    source: &mut F,
) -> Result<Collected>
where
    F: FnMut(&ImportPricing<D>, Duration) -> Result<ImportOracleReply>,
{
    let mut groups = vec![];
    for offset in 0..pending.session.order.len() {
        let id =
            pending.session.order[(pending.session.cursor + offset) % pending.session.order.len()];
        if point.open_group_ids().contains(&id) {
            groups.push(id);
            if !all {
                break;
            }
        }
    }
    let mut result = Collected {
        status: None,
        batches: 0,
        added: 0,
        completed: 0,
        numerical_bound: None,
    };
    let mut minima = vec![];
    for group in groups {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            result.status = Some(TreeStatus::TimeLimit);
            break;
        }
        if result.batches >= max_batches {
            result.status = Some(TreeStatus::BatchLimit);
            break;
        }
        let request = point.for_subproblem(group)?;
        let reply = source(&request, remaining)?;
        request.validate_reply(&reply, tolerance)?;
        if reply.batch.columns.iter().any(|c| c.bounds.lower != 0.) {
            return Err(invalid("import tree requires optional future columns"));
        }
        let added = reply.batch.columns.len();
        let complete = reply.batch.complete;
        let accepted = pending.session.view(group)?.accept(reply.batch)?;
        pending.session.definition = accepted.definition;
        pending.session.pending = accepted.open_groups;
        pending.session.cursor = (pending
            .session
            .order
            .iter()
            .position(|id| *id == group)
            .unwrap()
            + 1)
            % pending.session.order.len();
        broadcast(pending);
        result.batches += 1;
        result.added += added;
        result.completed += usize::from(complete);
        if let Some(minimum) = reply.minimum {
            minima.push(minimum);
        }
    }
    if numerical {
        result.numerical_bound = point.numerical_bound(&minima)?;
    }
    Ok(result)
}
