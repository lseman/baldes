//! Shared tree lifecycle; adapters interpret solved nodes and create children.
use super::{frontier::QueuedNode, Evaluation, NodeInfo, TreeFrontier};
use crate::{invalid, Result};
use std::{collections::VecDeque, time::Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    Exhausted,
    OverflowOnly,
    TimeLimit,
    NodeLimit,
    Incomplete,
    Requested,
}

pub(crate) enum Decision {
    Continue,
    Stop { reason: Stop, restore_ready: bool },
}

pub(crate) struct Limits {
    pub deadline: Instant,
    pub remaining_nodes: usize,
    pub batch_size: usize,
    pub time_first: bool,
}

pub(crate) trait Model {
    type Node: NodeInfo;
    type Output: Evaluation;
    fn limits(&self) -> Limits;
    fn prune(&mut self, _frontier: &mut TreeFrontier<Self::Node, Self::Output>) {}
    /// Selection accounting and pruning before an expensive evaluation.
    fn admit(&mut self, _entry: &mut QueuedNode<Self::Node, Self::Output>) -> bool {
        true
    }
    fn evaluate(
        &mut self,
        entries: &mut [QueuedNode<Self::Node, Self::Output>],
        frontier: &mut TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Vec<Result<Self::Output>>>;
    fn consume(
        &mut self,
        entry: Option<QueuedNode<Self::Node, Self::Output>>,
        output: Result<Self::Output>,
        reused: bool,
        frontier: &mut TreeFrontier<Self::Node, Self::Output>,
    ) -> Result<Decision>;
}

struct Ready<N, E> {
    entry: Option<QueuedNode<N, E>>,
    output: Result<E>,
    reused: bool,
}

/// Own batch selection, prepared-result reuse, budgets and evaluation dispatch.
/// A prepared root enters before frontier selection, just like an already
/// evaluated child. It is accounted by the model's consume hook.
pub(crate) fn run<M: Model>(
    model: &mut M,
    frontier: &mut TreeFrontier<M::Node, M::Output>,
    root: Option<M::Output>,
) -> Result<Stop> {
    let mut ready = VecDeque::new();
    if let Some(root) = root {
        ready.push_back(Ready {
            entry: None,
            output: Ok(root),
            reused: false,
        });
    }
    loop {
        model.prune(frontier);
        if ready.is_empty() {
            if frontier.is_empty() {
                return Ok(Stop::Exhausted);
            }
            // A live overflow channel does not establish exhaustion of the
            // primary search. Retain it for final bound/open-node reporting.
            if frontier.primary_is_empty() {
                return Ok(Stop::OverflowOnly);
            }
            let limits = model.limits();
            if limits.time_first && Instant::now() >= limits.deadline {
                return Ok(Stop::TimeLimit);
            }
            if limits.remaining_nodes == 0 {
                return Ok(Stop::NodeLimit);
            }
            if Instant::now() >= limits.deadline {
                return Ok(Stop::TimeLimit);
            }
            if limits.batch_size == 0 {
                return Err(invalid("tree batch size must be positive"));
            }
            let count = limits
                .batch_size
                .min(frontier.len())
                .min(limits.remaining_nodes);
            let mut batch = Vec::with_capacity(count);
            for _ in 0..count {
                let mut entry = frontier
                    .pop()?
                    .ok_or_else(|| invalid("tree frontier changed during selection"))?;
                if model.admit(&mut entry) {
                    batch.push(entry);
                }
            }
            if batch.iter().all(|entry| entry.evaluated.is_none()) {
                if !batch.is_empty() {
                    let outputs = model.evaluate(&mut batch, frontier)?;
                    if outputs.len() != batch.len() {
                        return Err(invalid("tree evaluation batch dimensions"));
                    }
                    ready.extend(batch.into_iter().zip(outputs).map(|(entry, output)| Ready {
                        entry: Some(entry),
                        output,
                        reused: false,
                    }));
                }
            } else {
                // Prepared entries keep their treatment rank and never incur
                // a second evaluation. Mixed batches preserve selection order.
                for mut entry in batch {
                    let reused = entry.evaluated.is_some();
                    let output = if let Some(output) = entry.evaluated.take() {
                        Ok(output)
                    } else {
                        let mut outputs =
                            model.evaluate(std::slice::from_mut(&mut entry), frontier)?;
                        if outputs.len() != 1 {
                            return Err(invalid("tree evaluation batch dimensions"));
                        }
                        outputs.pop().unwrap()
                    };
                    ready.push_back(Ready {
                        entry: Some(entry),
                        output,
                        reused,
                    });
                }
            }
        }
        let Some(next) = ready.pop_front() else {
            continue;
        };
        match model.consume(next.entry, next.output, next.reused, frontier)? {
            Decision::Continue => {}
            Decision::Stop {
                reason,
                restore_ready,
            } => {
                if restore_ready {
                    for next in ready {
                        if let Some(entry) = next.entry {
                            frontier.push_evaluated(entry.node, entry.parent_branch_order, None)?;
                        }
                    }
                }
                return Ok(reason);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/tree/driver_tests.rs"]
mod tests;
