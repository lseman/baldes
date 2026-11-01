//! Indexed collections for tolerant comparisons.
use super::*;
pub(super) use crate::master::ordering::{stable, Work};
use std::collections::HashMap;

struct Node {
    key: usize,
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
    red: bool,
}

/// Indexed red-black nodes retain the original multiset/map insertion behavior
/// even where tolerant comparisons do not define a transitive order.
pub(super) struct Tree {
    nodes: Vec<Node>,
    root: Option<usize>,
    locations: HashMap<usize, usize>,
}
impl Tree {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            root: None,
            locations: HashMap::new(),
        }
    }
    fn red(&self, node: Option<usize>) -> bool {
        node.is_some_and(|n| self.nodes[n].red)
    }
    fn rotate_left(&mut self, x: usize) {
        let y = self.nodes[x].right.unwrap();
        let child = self.nodes[y].left;
        self.nodes[x].right = child;
        if let Some(child) = child {
            self.nodes[child].parent = Some(x);
        }
        let parent = self.nodes[x].parent;
        self.nodes[y].parent = parent;
        if let Some(parent) = parent {
            if self.nodes[parent].left == Some(x) {
                self.nodes[parent].left = Some(y);
            } else {
                self.nodes[parent].right = Some(y);
            }
        } else {
            self.root = Some(y);
        }
        self.nodes[y].left = Some(x);
        self.nodes[x].parent = Some(y);
    }
    fn rotate_right(&mut self, x: usize) {
        let y = self.nodes[x].left.unwrap();
        let child = self.nodes[y].right;
        self.nodes[x].left = child;
        if let Some(child) = child {
            self.nodes[child].parent = Some(x);
        }
        let parent = self.nodes[x].parent;
        self.nodes[y].parent = parent;
        if let Some(parent) = parent {
            if self.nodes[parent].right == Some(x) {
                self.nodes[parent].right = Some(y);
            } else {
                self.nodes[parent].left = Some(y);
            }
        } else {
            self.root = Some(y);
        }
        self.nodes[y].right = Some(x);
        self.nodes[x].parent = Some(y);
    }
    fn rebalance(&mut self, mut x: usize) {
        while let Some(parent) = self.nodes[x].parent.filter(|&p| self.nodes[p].red) {
            let grand = self.nodes[parent].parent.unwrap();
            if self.nodes[grand].left == Some(parent) {
                let uncle = self.nodes[grand].right;
                if self.red(uncle) {
                    self.nodes[parent].red = false;
                    self.nodes[uncle.unwrap()].red = false;
                    self.nodes[grand].red = true;
                    x = grand;
                } else {
                    if self.nodes[parent].right == Some(x) {
                        x = parent;
                        self.rotate_left(x);
                    }
                    let parent = self.nodes[x].parent.unwrap();
                    let grand = self.nodes[parent].parent.unwrap();
                    self.nodes[parent].red = false;
                    self.nodes[grand].red = true;
                    self.rotate_right(grand);
                }
            } else {
                let uncle = self.nodes[grand].left;
                if self.red(uncle) {
                    self.nodes[parent].red = false;
                    self.nodes[uncle.unwrap()].red = false;
                    self.nodes[grand].red = true;
                    x = grand;
                } else {
                    if self.nodes[parent].left == Some(x) {
                        x = parent;
                        self.rotate_right(x);
                    }
                    let parent = self.nodes[x].parent.unwrap();
                    let grand = self.nodes[parent].parent.unwrap();
                    self.nodes[parent].red = false;
                    self.nodes[grand].red = true;
                    self.rotate_left(grand);
                }
            }
        }
        self.nodes[self.root.unwrap()].red = false;
    }
    fn attach(&mut self, key: usize, parent: Option<usize>, left: bool) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node {
            key,
            parent,
            left: None,
            right: None,
            red: true,
        });
        self.locations.insert(key, index);
        if let Some(parent) = parent {
            if left {
                self.nodes[parent].left = Some(index);
            } else {
                self.nodes[parent].right = Some(index);
            }
        } else {
            self.root = Some(index);
        }
        self.rebalance(index);
        index
    }
    pub fn insert_equal(
        &mut self,
        key: usize,
        before: &impl Fn(usize, usize) -> bool,
        work: &mut Work,
    ) -> Option<usize> {
        let (mut node, mut parent, mut left) = (self.root, None, true);
        while let Some(index) = node {
            parent = node;
            left = work.compare(key, self.nodes[index].key, before)?;
            node = if left {
                self.nodes[index].left
            } else {
                self.nodes[index].right
            };
        }
        Some(self.attach(key, parent, left))
    }
    pub fn insert_unique(
        &mut self,
        key: usize,
        before: &impl Fn(usize, usize) -> bool,
        work: &mut Work,
    ) -> Option<(usize, bool)> {
        let (mut node, mut parent, mut left, mut lower) = (self.root, None, true, None);
        while let Some(index) = node {
            parent = node;
            left = !work.compare(self.nodes[index].key, key, before)?;
            if left {
                lower = node;
                node = self.nodes[index].left;
            } else {
                node = self.nodes[index].right;
            }
        }
        if let Some(lower) = lower {
            if !work.compare(key, self.nodes[lower].key, before)? {
                return Some((self.nodes[lower].key, false));
            }
        }
        self.attach(key, parent, left);
        Some((key, true))
    }
    pub fn ordered(&self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.nodes.len());
        let (mut pending, mut node) = (vec![], self.root);
        while node.is_some() || !pending.is_empty() {
            if let Some(index) = node {
                pending.push(index);
                node = self.nodes[index].left;
            } else {
                let index = pending.pop().unwrap();
                result.push(self.nodes[index].key);
                node = self.nodes[index].right;
            }
        }
        result
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }

    fn replace(&mut self, old: usize, new: Option<usize>) {
        let parent = self.nodes[old].parent;
        if let Some(parent) = parent {
            if self.nodes[parent].left == Some(old) {
                self.nodes[parent].left = new;
            } else {
                self.nodes[parent].right = new;
            }
        } else {
            self.root = new;
        }
        if let Some(new) = new {
            self.nodes[new].parent = parent;
        }
    }
    fn black(&mut self, node: Option<usize>) {
        if let Some(node) = node {
            self.nodes[node].red = false;
        }
    }
    fn repair_erase(&mut self, mut x: Option<usize>, mut parent: Option<usize>) {
        while x != self.root && !self.red(x) {
            let p = parent.unwrap();
            if x == self.nodes[p].left {
                let mut sibling = self.nodes[p].right.unwrap();
                if self.nodes[sibling].red {
                    self.nodes[sibling].red = false;
                    self.nodes[p].red = true;
                    self.rotate_left(p);
                    sibling = self.nodes[p].right.unwrap();
                }
                if !self.red(self.nodes[sibling].left) && !self.red(self.nodes[sibling].right) {
                    self.nodes[sibling].red = true;
                    x = Some(p);
                    parent = self.nodes[p].parent;
                } else {
                    if !self.red(self.nodes[sibling].right) {
                        self.black(self.nodes[sibling].left);
                        self.nodes[sibling].red = true;
                        self.rotate_right(sibling);
                        sibling = self.nodes[p].right.unwrap();
                    }
                    self.nodes[sibling].red = self.nodes[p].red;
                    self.nodes[p].red = false;
                    self.black(self.nodes[sibling].right);
                    self.rotate_left(p);
                    x = self.root;
                }
            } else {
                let mut sibling = self.nodes[p].left.unwrap();
                if self.nodes[sibling].red {
                    self.nodes[sibling].red = false;
                    self.nodes[p].red = true;
                    self.rotate_right(p);
                    sibling = self.nodes[p].left.unwrap();
                }
                if !self.red(self.nodes[sibling].right) && !self.red(self.nodes[sibling].left) {
                    self.nodes[sibling].red = true;
                    x = Some(p);
                    parent = self.nodes[p].parent;
                } else {
                    if !self.red(self.nodes[sibling].left) {
                        self.black(self.nodes[sibling].right);
                        self.nodes[sibling].red = true;
                        self.rotate_left(sibling);
                        sibling = self.nodes[p].left.unwrap();
                    }
                    self.nodes[sibling].red = self.nodes[p].red;
                    self.nodes[p].red = false;
                    self.black(self.nodes[sibling].left);
                    self.rotate_right(p);
                    x = self.root;
                }
            }
        }
        self.black(x);
    }
    pub fn remove(&mut self, key: usize) {
        let z = self.locations.remove(&key).unwrap();
        let (x, parent, red);
        if self.nodes[z].left.is_none() {
            x = self.nodes[z].right;
            parent = self.nodes[z].parent;
            red = self.nodes[z].red;
            self.replace(z, x);
        } else if self.nodes[z].right.is_none() {
            x = self.nodes[z].left;
            parent = self.nodes[z].parent;
            red = self.nodes[z].red;
            self.replace(z, x);
        } else {
            let mut successor = self.nodes[z].right.unwrap();
            while let Some(left) = self.nodes[successor].left {
                successor = left;
            }
            red = self.nodes[successor].red;
            x = self.nodes[successor].right;
            if self.nodes[successor].parent == Some(z) {
                parent = Some(successor);
                if let Some(x) = x {
                    self.nodes[x].parent = Some(successor);
                }
            } else {
                parent = self.nodes[successor].parent;
                self.replace(successor, x);
                self.nodes[successor].right = self.nodes[z].right;
                let right = self.nodes[successor].right.unwrap();
                self.nodes[right].parent = Some(successor);
            }
            self.replace(z, Some(successor));
            self.nodes[successor].left = self.nodes[z].left;
            let left = self.nodes[successor].left.unwrap();
            self.nodes[left].parent = Some(successor);
            self.nodes[successor].red = self.nodes[z].red;
        }
        if !red {
            self.repair_erase(x, parent);
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PriorityKey {
    pub family: usize,
    pub family_priority: f64,
    pub rule: u8,
    pub variable_priority: f64,
    pub added_level: i32,
}

fn selection_before(
    a: &PriorityKey,
    b: &PriorityKey,
    creation_a: usize,
    creation_b: usize,
) -> bool {
    if less(b.family_priority, a.family_priority) {
        return true;
    }
    if less(a.family_priority, b.family_priority) {
        return false;
    }
    if a.rule == 2 && b.rule == 2 {
        if less(b.variable_priority, a.variable_priority) {
            return true;
        }
        if less(a.variable_priority, b.variable_priority) {
            return false;
        }
    }
    let (a, b) = (a.added_level as f64, b.added_level as f64);
    let slack = a.abs().max(b.abs()) * 1e-6 + 1e-6;
    if !(a < b - slack || b < a - slack) {
        creation_a < creation_b
    } else {
        less(a, b)
    }
}

fn filter_family(
    tree: &mut Tree,
    keys: &[PriorityKey],
    family_key: usize,
    work: &mut Work,
) -> Option<()> {
    let key = keys[family_key];
    if key.rule != 2 {
        return Some(());
    }
    let mut best = 0.;
    for index in tree.ordered() {
        work.step()?;
        if keys[index].family == key.family && less(best, keys[index].variable_priority) {
            best = keys[index].variable_priority;
        }
    }
    for index in tree.ordered() {
        work.step()?;
        if keys[index].family == key.family && less(keys[index].variable_priority, best) {
            tree.remove(index);
        }
    }
    Some(())
}

pub(super) fn publish(keys: &[PriorityKey], quota: usize, work: &mut Work) -> Option<Vec<usize>> {
    if !work.alive() {
        return None;
    }
    let mut tree = Tree::new();
    for (index, key) in keys.iter().enumerate() {
        work.step()?;
        if index > 0 && key.family != keys[index - 1].family {
            filter_family(&mut tree, keys, index - 1, work)?;
        }
        tree.insert_unique(
            index,
            &|a, b| selection_before(&keys[a], &keys[b], a, b),
            work,
        )?;
        if tree.len() > quota {
            let last = *tree.ordered().last().unwrap();
            tree.remove(last);
        }
    }
    if !keys.is_empty() {
        filter_family(&mut tree, keys, keys.len() - 1, work)?;
    }
    work.alive().then(|| tree.ordered())
}
