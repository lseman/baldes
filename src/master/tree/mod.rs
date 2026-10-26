//! Shared branch-and-price lifecycle and policies; model registrations are adapters.
#[path = "adapters/routing/mod.rs"]
mod routing;
pub use routing::{
    candidates, clusters, history, strong, BranchConstraint, BranchEvaluationHistory,
    BranchHistoryOptions, BranchHistoryPolicy, BranchingOptions, DivingOptions,
    StrongBranchingOptions, StrongPhase, TreeOptions, TreeOutput,
};
mod driver;
mod frontier;
pub(crate) use driver::{run, Decision, Limits, Model, Stop};
pub(crate) use frontier::queue_bound;
pub(crate) use frontier::QueuedNode;
#[cfg(test)]
pub(crate) use frontier::{
    bound_priority_bound, bound_priority_less, bound_priority_values, BoundPriorityKey,
};
pub(crate) use frontier::{Evaluation, NodeInfo, TreeFrontier};
pub use frontier::{IntegerBoundRounding, TreeSearchPolicy};

/// Priority is model policy; heap insertion, removal, and rebuilding are shared.
pub(crate) trait Priority<T> {
    /// True if left should be treated later than right.
    fn less(&self, left: &T, right: &T) -> bool;
}

pub(crate) struct Frontier<T, P> {
    pub(crate) nodes: Vec<T>,
    priority: P,
}
impl<T, P: Priority<T>> Frontier<T, P> {
    pub fn new(priority: P) -> Self {
        Self {
            nodes: vec![],
            priority,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.nodes.iter()
    }
    pub fn less(&self, left: &T, right: &T) -> bool {
        self.priority.less(left, right)
    }
    pub fn push(&mut self, node: T) {
        self.nodes.push(node);
        let mut hole = self.nodes.len() - 1;
        while hole > 0 {
            let parent = (hole - 1) / 2;
            if !self.less(&self.nodes[parent], &self.nodes[hole]) {
                break;
            }
            self.nodes.swap(parent, hole);
            hole = parent;
        }
    }
    /// Match the recovered heap's child selection before sifting upward.
    /// This matters for tolerance-based priority predicates with equivalent keys.
    pub fn pop(&mut self) -> Option<T> {
        if self.nodes.is_empty() {
            return None;
        }
        let result = self.nodes.swap_remove(0);
        let len = self.nodes.len();
        if len == 0 {
            return Some(result);
        }
        let mut hole = 0;
        while 2 * hole + 2 < len {
            let right = 2 * hole + 2;
            let left = right - 1;
            let child = if self.less(&self.nodes[right], &self.nodes[left]) {
                left
            } else {
                right
            };
            self.nodes.swap(hole, child);
            hole = child;
        }
        if 2 * hole + 1 < len {
            self.nodes.swap(hole, 2 * hole + 1);
            hole = 2 * hole + 1;
        }
        while hole > 0 {
            let parent = (hole - 1) / 2;
            if !self.less(&self.nodes[parent], &self.nodes[hole]) {
                break;
            }
            self.nodes.swap(parent, hole);
            hole = parent;
        }
        Some(result)
    }
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        // Rebuild after arbitrary removals; retaining a Vec in-place alone
        // would invalidate heap ordering when surviving nodes change parents.
        let nodes = std::mem::take(&mut self.nodes);
        for node in nodes {
            if keep(&node) {
                self.push(node);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/tree/frontier_tests.rs"]
mod tests;

mod status;
pub use status::TreeStatus;

/// Compile model-specific registrations and run the shared tree lifecycle.
pub trait TreeInput {
    type Options;
    type Output;
    fn run_tree(self, options: Self::Options) -> crate::Result<Self::Output>;
}
pub fn solve<I: TreeInput>(input: I, options: I::Options) -> crate::Result<I::Output> {
    input.run_tree(options)
}

pub use crate::master::model::tree_adapter::{
    ModelTreeOptions, ModelTreeOutput, NetworkBranchDirection, NetworkBranchPriorities,
    NetworkBranchingPolicy,
};
