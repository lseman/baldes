//! Shared queue policies, prepared-node ranks and bounded overflow channels.
use super::{Frontier, Priority};
use crate::{invalid, Result};

pub(crate) trait NodeInfo {
    fn depth(&self) -> usize;
    fn numerical_bound(&self) -> f64;
}
pub(crate) trait Evaluation {
    fn proven_infeasible(&self) -> bool;
}

/// Assignment-tree frontier order, independent of which branch family is used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TreeSearchPolicy {
    /// Native strategy 1: depth, numerical LP bound, then stable creation order.
    #[default]
    Default,
    /// Process the open node with the lowest inherited numerical LP bound first.
    BestBound,
    /// Native strategy 2: inherited numerical dual bound with declared rounding,
    /// queued-child primal sentinels, depth, parent branch-and-price order.
    #[serde(rename = "bound-priority")]
    BoundPriority,
    /// Native strategy 3: depth, then higher numerical LP bound, then creation.
    #[serde(rename = "depth-bound-priority")]
    DepthBoundPriority,
}

impl TreeSearchPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "default" => Ok(Self::Default),
            "best-bound" => Ok(Self::BestBound),
            "bound-priority" => Ok(Self::BoundPriority),
            "depth-bound-priority" => Ok(Self::DepthBoundPriority),
            _ => Err(invalid(
                "network tree search must be default, best-bound, bound-priority or depth-bound-priority",
            )),
        }
    }
}

/// Integer conversion used for native strategy-2 dual-bound ranking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntegerBoundRounding {
    /// Native nonpositive control with a continuous-objective bound tag.
    #[default]
    Unrounded,
    /// Native nonpositive control with an integer-objective bound tag.
    GuardedTolerance,
    /// Native positive rounding control: ceil the numerical bound directly.
    ExactCeil,
}

pub(crate) struct QueuedNode<N, E> {
    pub(crate) node: N,
    pub(crate) evaluated: Option<E>,
    pub(crate) serial: u64,
    pub(crate) branch_order: i32,
    pub(crate) parent_branch_order: i32,
    pub(crate) strategy2_dual_bound: f64,
}

impl<N: NodeInfo, E: Evaluation> QueuedNode<N, E> {
    pub(crate) fn proven_infeasible(&self) -> bool {
        self.evaluated
            .as_ref()
            .is_some_and(Evaluation::proven_infeasible)
    }

    pub(crate) fn numerical_bound(&self) -> f64 {
        if self.proven_infeasible() {
            f64::INFINITY
        } else {
            self.node.numerical_bound()
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum HeapOrder {
    Primary,
    Secondary,
    BoundPriority,
    DepthBoundPriority,
    BestBound,
    ExactBestBound,
}

pub(crate) fn queue_bound(value: f64) -> f64 {
    if value == f64::NEG_INFINITY {
        -1e12
    } else if value == f64::INFINITY {
        1e12
    } else {
        value
    }
}

fn bound_less(left: f64, right: f64) -> bool {
    let left = queue_bound(left);
    let right = queue_bound(right);
    let epsilon = 1e-6 + 1e-10 * left.abs().max(right.abs());
    left < right - epsilon
}

/// Strategy 2 uses the native node's secondary dual bound, whose rounding
/// depends on the declared objective contract.
/// BALDES keeps the underlying numerical LP bound separately from its
/// controller-certified bound, so the native queue key is derived from the
/// numerical channel and never from certification metadata.
pub(crate) fn bound_priority_bound<N: NodeInfo>(node: &N, rounding: IntegerBoundRounding) -> f64 {
    bound_priority_values(node.numerical_bound(), rounding)
}

pub(crate) fn bound_priority_values(numerical_bound: f64, rounding: IntegerBoundRounding) -> f64 {
    // Native children copy the secondary bound unconditionally. In particular,
    // an unknown numerical channel must not acquire a controller-certified key.
    let bound = numerical_bound;
    if !bound.is_finite() {
        return queue_bound(bound);
    }
    if bound.abs() >= 4_503_599_627_370_496. {
        return bound;
    }
    match rounding {
        IntegerBoundRounding::Unrounded => bound,
        IntegerBoundRounding::GuardedTolerance => {
            let epsilon = 1e-6 + 1e-10 * bound.abs();
            (bound - epsilon).ceil()
        }
        IntegerBoundRounding::ExactCeil => bound.ceil(),
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BoundPriorityKey {
    pub(crate) dual_bound: f64,
    pub(crate) depth: usize,
    pub(crate) parent_branch_order: i32,
    pub(crate) serial: u64,
}

/// Native operator< reports lower queue priority first.
pub(crate) fn bound_priority_less(left: BoundPriorityKey, right: BoundPriorityKey) -> bool {
    if bound_less(left.dual_bound, right.dual_bound) {
        return false;
    }
    if bound_less(right.dual_bound, left.dual_bound) {
        return true;
    }
    if left.depth != right.depth {
        return left.depth < right.depth;
    }
    if left.parent_branch_order != right.parent_branch_order {
        return left.parent_branch_order > right.parent_branch_order;
    }
    left.serial >= right.serial
}

type FrontierHeap<N, E> = Frontier<QueuedNode<N, E>, HeapOrder>;

impl<N: NodeInfo, E: Evaluation> Priority<QueuedNode<N, E>> for HeapOrder {
    /// Returns true when `left` has lower queue priority than `right`, exactly
    /// following the native pairwise bound predicate and short-circuit order.
    fn less(&self, left: &QueuedNode<N, E>, right: &QueuedNode<N, E>) -> bool {
        if matches!(self, HeapOrder::ExactBestBound) {
            return left
                .node
                .numerical_bound()
                .total_cmp(&right.node.numerical_bound())
                .then(left.serial.cmp(&right.serial))
                .is_gt();
        }
        let left_depth = left.node.depth();
        let right_depth = right.node.depth();
        if matches!(*self, HeapOrder::BoundPriority) {
            // A native child initializes both primal-bound tie fields to the
            // objective-sense sentinel and they stay equal until evaluation.
            return bound_priority_less(
                BoundPriorityKey {
                    dual_bound: left.strategy2_dual_bound,
                    depth: left_depth,
                    parent_branch_order: left.parent_branch_order,
                    serial: left.serial,
                },
                BoundPriorityKey {
                    dual_bound: right.strategy2_dual_bound,
                    depth: right_depth,
                    parent_branch_order: right.parent_branch_order,
                    serial: right.serial,
                },
            );
        }
        if !matches!(*self, HeapOrder::BestBound) && left_depth != right_depth {
            return left_depth < right_depth;
        }
        // Native primary/overflow comparators read the ordinary LP dual
        // channel. The selected bound may instead be a controller certificate
        // and belongs to pruning, not native queue ranking.
        let left_bound = left.numerical_bound();
        let right_bound = right.numerical_bound();
        if bound_less(left_bound, right_bound) {
            return matches!(*self, HeapOrder::DepthBoundPriority);
        }
        if bound_less(right_bound, left_bound) {
            return !matches!(*self, HeapOrder::DepthBoundPriority);
        }
        if matches!(*self, HeapOrder::BestBound) && left_depth != right_depth {
            return left_depth < right_depth;
        }
        match *self {
            HeapOrder::Secondary => left.serial > right.serial,
            HeapOrder::Primary
            | HeapOrder::BoundPriority
            | HeapOrder::DepthBoundPriority
            | HeapOrder::BestBound
            | HeapOrder::ExactBestBound => left.serial >= right.serial,
        }
    }
}

pub(crate) enum TreeFrontier<N, E> {
    BestBound {
        nodes: FrontierHeap<N, E>,
        overflow: FrontierHeap<N, E>,
        next_serial: u64,
        next_branch_order: i32,
        open_node_limit: usize,
        strategy2_bound_rounding: IntegerBoundRounding,
    },
    Bounded {
        primary: FrontierHeap<N, E>,
        overflow: FrontierHeap<N, E>,
        next_serial: u64,
        next_branch_order: i32,
        open_node_limit: usize,
        strategy2_bound_rounding: IntegerBoundRounding,
    },
}

impl<N: NodeInfo, E: Evaluation> TreeFrontier<N, E> {
    pub(crate) fn new(
        policy: TreeSearchPolicy,
        open_node_limit: usize,
        strategy2_bound_rounding: IntegerBoundRounding,
    ) -> Self {
        match policy {
            TreeSearchPolicy::BestBound => Self::BestBound {
                nodes: FrontierHeap::new(HeapOrder::BestBound),
                overflow: FrontierHeap::new(HeapOrder::Secondary),
                next_serial: 0,
                next_branch_order: 2,
                open_node_limit,
                strategy2_bound_rounding,
            },
            TreeSearchPolicy::BoundPriority => Self::Bounded {
                primary: FrontierHeap::new(HeapOrder::BoundPriority),
                overflow: FrontierHeap::new(HeapOrder::Secondary),
                next_serial: 0,
                next_branch_order: 2,
                open_node_limit,
                strategy2_bound_rounding,
            },
            TreeSearchPolicy::Default => Self::Bounded {
                primary: FrontierHeap::new(HeapOrder::Primary),
                overflow: FrontierHeap::new(HeapOrder::Secondary),
                next_serial: 0,
                next_branch_order: 2,
                open_node_limit,
                strategy2_bound_rounding,
            },
            TreeSearchPolicy::DepthBoundPriority => Self::Bounded {
                primary: FrontierHeap::new(HeapOrder::DepthBoundPriority),
                overflow: FrontierHeap::new(HeapOrder::Secondary),
                next_serial: 0,
                next_branch_order: 2,
                open_node_limit,
                strategy2_bound_rounding,
            },
        }
    }

    pub(crate) fn exact_best_bound() -> Self {
        Self::BestBound {
            nodes: FrontierHeap::new(HeapOrder::ExactBestBound),
            overflow: FrontierHeap::new(HeapOrder::Secondary),
            next_serial: 0,
            next_branch_order: 2,
            open_node_limit: usize::MAX,
            strategy2_bound_rounding: IntegerBoundRounding::Unrounded,
        }
    }
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::BestBound {
                nodes, overflow, ..
            } => nodes.is_empty() && overflow.is_empty(),
            Self::Bounded {
                primary, overflow, ..
            } => primary.is_empty() && overflow.is_empty(),
        }
    }

    pub(crate) fn primary_is_empty(&self) -> bool {
        match self {
            Self::BestBound { nodes, .. } => nodes.is_empty(),
            Self::Bounded { primary, .. } => primary.is_empty(),
        }
    }

    #[cfg(test)]
    pub(crate) fn push(&mut self, node: N, parent_branch_order: i32) -> Result<()> {
        self.push_evaluated(node, parent_branch_order, None)
    }

    pub(crate) fn push_evaluated(
        &mut self,
        node: N,
        parent_branch_order: i32,
        evaluated: Option<E>,
    ) -> Result<()> {
        // Native assigns a branch-and-price rank while admitting a prepared
        // son, before it reaches the frontier. Ordinary sons receive a rank
        // when selected. Exact strong-branching outputs are the prepared-son
        // representation here.
        let branch_order = if evaluated.is_some() {
            let next = match self {
                Self::BestBound {
                    next_branch_order, ..
                }
                | Self::Bounded {
                    next_branch_order, ..
                } => next_branch_order,
            };
            let rank = *next;
            *next = next
                .checked_add(1)
                .ok_or_else(|| invalid("branch-and-price order overflow"))?;
            rank
        } else {
            0
        };
        let strategy2_bound_rounding = match self {
            Self::BestBound {
                strategy2_bound_rounding,
                ..
            }
            | Self::Bounded {
                strategy2_bound_rounding,
                ..
            } => *strategy2_bound_rounding,
        };
        let strategy2_dual_bound = if evaluated
            .as_ref()
            .is_some_and(Evaluation::proven_infeasible)
        {
            bound_priority_values(f64::INFINITY, strategy2_bound_rounding)
        } else {
            bound_priority_bound(&node, strategy2_bound_rounding)
        };
        match self {
            Self::BestBound {
                nodes,
                overflow,
                next_serial,
                open_node_limit,
                ..
            } => {
                let queued = QueuedNode {
                    node,
                    evaluated,
                    serial: *next_serial,
                    branch_order,
                    parent_branch_order,
                    strategy2_dual_bound,
                };
                *next_serial = next_serial
                    .checked_add(1)
                    .ok_or_else(|| invalid("network tree creation serial overflow"))?;
                if nodes.nodes.len() < *open_node_limit {
                    nodes.push(queued);
                } else {
                    overflow.push(queued);
                }
            }
            Self::Bounded {
                primary,
                overflow,
                next_serial,
                open_node_limit,
                ..
            } => {
                let queued = QueuedNode {
                    node,
                    evaluated,
                    serial: *next_serial,
                    branch_order,
                    parent_branch_order,
                    strategy2_dual_bound,
                };
                *next_serial = next_serial
                    .checked_add(1)
                    .ok_or_else(|| invalid("tree creation serial overflow"))?;
                if primary.nodes.len() < *open_node_limit {
                    primary.push(queued);
                } else {
                    overflow.push(queued);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn pop(&mut self) -> Result<Option<QueuedNode<N, E>>> {
        let (queued, next_branch_order) = match self {
            Self::BestBound {
                nodes,
                overflow,
                next_branch_order,
                ..
            } => (overflow.pop().or_else(|| nodes.pop()), next_branch_order),
            Self::Bounded {
                primary,
                overflow,
                next_branch_order,
                ..
            } => (overflow.pop().or_else(|| primary.pop()), next_branch_order),
        };
        let Some(mut queued) = queued else {
            return Ok(None);
        };
        // Creation serial and treatment rank are distinct native fields. The
        // root has rank 1. Ordinary sons get a rank on selection; prepared
        // sons retain the rank assigned at admission.
        if queued.branch_order == 0 {
            queued.branch_order = *next_branch_order;
            *next_branch_order = next_branch_order
                .checked_add(1)
                .ok_or_else(|| invalid("branch-and-price order overflow"))?;
        }
        Ok(Some(queued))
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::BestBound {
                nodes, overflow, ..
            } => nodes.len() + overflow.len(),
            Self::Bounded {
                primary, overflow, ..
            } => primary.len() + overflow.len(),
        }
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = &QueuedNode<N, E>> {
        let (primary, overflow) = match self {
            Self::BestBound {
                nodes, overflow, ..
            } => (nodes, overflow),
            Self::Bounded {
                primary, overflow, ..
            } => (primary, overflow),
        };
        primary.iter().chain(overflow.iter())
    }
    /// Metadata edits must preserve priority keys; rebuild the frontier when changing keys.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut QueuedNode<N, E>> {
        let (primary, overflow) = match self {
            Self::BestBound {
                nodes, overflow, ..
            } => (nodes, overflow),
            Self::Bounded {
                primary, overflow, ..
            } => (primary, overflow),
        };
        primary.nodes.iter_mut().chain(overflow.nodes.iter_mut())
    }
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&N) -> bool) {
        let (primary, overflow) = match self {
            Self::BestBound {
                nodes, overflow, ..
            } => (nodes, overflow),
            Self::Bounded {
                primary, overflow, ..
            } => (primary, overflow),
        };
        primary.retain(|entry| keep(&entry.node));
        overflow.retain(|entry| keep(&entry.node));
    }
    pub(crate) fn into_nodes(self) -> Vec<QueuedNode<N, E>> {
        match self {
            Self::BestBound {
                nodes, overflow, ..
            } => nodes.nodes.into_iter().chain(overflow.nodes).collect(),
            Self::Bounded {
                primary, overflow, ..
            } => primary.nodes.into_iter().chain(overflow.nodes).collect(),
        }
    }
}
