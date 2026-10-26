use super::*;
use crate::master::tree::frontier::QueuedNode;

struct Node {
    id: usize,
    bound: f64,
}
impl NodeInfo for Node {
    fn depth(&self) -> usize {
        0
    }
    fn numerical_bound(&self) -> f64 {
        self.bound
    }
}
struct Output(usize);
impl Evaluation for Output {
    fn proven_infeasible(&self) -> bool {
        false
    }
}
struct Hooks {
    selected: usize,
    evaluated: Vec<usize>,
    consumed: Vec<usize>,
    deadline: Instant,
    limit: usize,
    time_first: bool,
    spawn: bool,
    suspend: bool,
}
impl Hooks {
    fn new() -> Self {
        Self {
            selected: 0,
            evaluated: vec![],
            consumed: vec![],
            deadline: Instant::now() + std::time::Duration::from_secs(10),
            limit: 10,
            time_first: false,
            spawn: false,
            suspend: false,
        }
    }
}
impl Model for Hooks {
    type Node = Node;
    type Output = Output;
    fn limits(&self) -> Limits {
        Limits {
            deadline: self.deadline,
            remaining_nodes: self.limit.saturating_sub(self.selected),
            batch_size: 2,
            time_first: self.time_first,
        }
    }
    fn admit(&mut self, _entry: &mut QueuedNode<Node, Output>) -> bool {
        self.selected += 1;
        true
    }
    fn evaluate(
        &mut self,
        entries: &mut [QueuedNode<Node, Output>],
        _frontier: &mut TreeFrontier<Node, Output>,
    ) -> Result<Vec<Result<Output>>> {
        Ok(entries
            .iter()
            .map(|entry| {
                self.evaluated.push(entry.node.id);
                Ok(Output(entry.node.id))
            })
            .collect())
    }
    fn consume(
        &mut self,
        entry: Option<QueuedNode<Node, Output>>,
        output: Result<Output>,
        _reused: bool,
        frontier: &mut TreeFrontier<Node, Output>,
    ) -> Result<Decision> {
        let output = output?;
        if entry.is_none() {
            self.selected += 1;
        }
        if self.suspend {
            let entry = entry.unwrap();
            frontier.push_evaluated(entry.node, entry.parent_branch_order, None)?;
            return Ok(Decision::Stop {
                reason: Stop::Incomplete,
                restore_ready: true,
            });
        }
        self.consumed.push(output.0);
        if self.spawn && output.0 == 0 {
            frontier.push_evaluated(Node { id: 1, bound: 1. }, 1, Some(Output(1)))?;
            frontier.push_evaluated(Node { id: 2, bound: 2. }, 1, None)?;
        }
        Ok(Decision::Continue)
    }
}

#[test]
fn prepared_root_and_children_use_one_driver_without_repeating_evaluations() {
    let mut hooks = Hooks::new();
    hooks.spawn = true;
    let mut frontier = TreeFrontier::exact_best_bound();
    assert_eq!(
        run(&mut hooks, &mut frontier, Some(Output(0))).unwrap(),
        Stop::Exhausted
    );
    assert_eq!(hooks.consumed, vec![0, 1, 2]);
    assert_eq!(hooks.evaluated, vec![2]);
    assert_eq!(hooks.selected, 3);
}

#[test]
fn interrupted_batch_restores_current_and_ready_nodes_for_open_bound_accounting() {
    let mut hooks = Hooks::new();
    hooks.suspend = true;
    let mut frontier = TreeFrontier::exact_best_bound();
    for id in 0..2 {
        frontier
            .push_evaluated(
                Node {
                    id,
                    bound: id as f64,
                },
                0,
                None,
            )
            .unwrap();
    }
    assert_eq!(
        run(&mut hooks, &mut frontier, None).unwrap(),
        Stop::Incomplete
    );
    assert_eq!(hooks.evaluated, vec![0, 1]);
    assert_eq!(frontier.len(), 2);
    let mut ids: Vec<_> = frontier.iter().map(|n| n.node.id).collect();
    ids.sort();
    assert_eq!(ids, vec![0, 1]);
    assert!(frontier.iter().all(|n| n.evaluated.is_none()));
}

#[test]
fn overflow_only_frontier_remains_unresolved_and_limits_keep_policy_order() {
    let mut hooks = Hooks::new();
    let mut frontier = TreeFrontier::new(
        super::super::TreeSearchPolicy::BestBound,
        0,
        super::super::IntegerBoundRounding::Unrounded,
    );
    frontier
        .push_evaluated(Node { id: 1, bound: 0. }, 0, None)
        .unwrap();
    assert_eq!(
        run(&mut hooks, &mut frontier, None).unwrap(),
        Stop::OverflowOnly
    );
    assert_eq!(frontier.len(), 1);
    assert!(hooks.evaluated.is_empty());
    let mut frontier = TreeFrontier::exact_best_bound();
    frontier
        .push_evaluated(Node { id: 1, bound: 0. }, 0, None)
        .unwrap();
    hooks.deadline = Instant::now();
    hooks.limit = 0;
    assert_eq!(
        run(&mut hooks, &mut frontier, None).unwrap(),
        Stop::NodeLimit
    );
    hooks.time_first = true;
    assert_eq!(
        run(&mut hooks, &mut frontier, None).unwrap(),
        Stop::TimeLimit
    );
}
