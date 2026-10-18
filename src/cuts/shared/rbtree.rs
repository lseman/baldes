//! The ordered cut candidate set (`std::set` of (score, cut) pairs) as the
//! separators fill it. The comparator has a 1e-6 tolerance and is therefore
//! not transitive; the insertion position depends on the tree shape, so this
//! reproduces libstdc++'s `_M_get_insert_unique_pos` and
//! `_Rb_tree_insert_and_rebalance` rather than using a sorted vector.

const NIL: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
struct Node {
    score: f64,
    id: i32,
    value: usize,
    left: usize,
    right: usize,
    parent: usize,
    red: bool,
}

/// Candidate set ordered by score ascending beyond
/// 1e-6, ties by cut id. `value` is the caller's cut handle.
#[derive(Clone, Debug, Default)]
pub struct CandidateSet {
    nodes: Vec<Node>,
    root: usize,
    leftmost: usize,
    rightmost: usize,
}

fn less(a: (f64, i32), b: (f64, i32)) -> bool {
    const EPS: f64 = 1e-6;
    if a.0 < b.0 - EPS {
        return true;
    }
    if a.0 > b.0 + EPS {
        return false;
    }
    a.1 < b.1
}

impl CandidateSet {
    pub fn new() -> Self {
        Self {
            nodes: vec![],
            root: NIL,
            leftmost: NIL,
            rightmost: NIL,
        }
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn clear(&mut self) {
        *self = Self::new();
    }
    fn key(&self, n: usize) -> (f64, i32) {
        (self.nodes[n].score, self.nodes[n].id)
    }
    fn predecessor(&self, mut x: usize) -> usize {
        if self.nodes[x].left != NIL {
            x = self.nodes[x].left;
            while self.nodes[x].right != NIL {
                x = self.nodes[x].right;
            }
            return x;
        }
        let mut y = self.nodes[x].parent;
        while y != NIL && x == self.nodes[y].left {
            x = y;
            y = self.nodes[y].parent;
        }
        y
    }
    /// Returns false when an equivalent element is present (not inserted).
    pub fn insert(&mut self, score: f64, id: i32, value: usize) -> bool {
        let k = (score, id);
        let mut x = self.root;
        let mut y = NIL;
        let mut comp = true;
        while x != NIL {
            y = x;
            comp = less(k, self.key(x));
            x = if comp {
                self.nodes[x].left
            } else {
                self.nodes[x].right
            };
        }
        let mut j = y;
        if comp {
            if j == self.leftmost {
                return self.insert_at(y, k, value);
            }
            j = self.predecessor(j);
        }
        if less(self.key(j), k) {
            return self.insert_at(y, k, value);
        }
        false
    }
    fn insert_at(&mut self, p: usize, k: (f64, i32), value: usize) -> bool {
        // _M_insert_ with x == 0: left when p is the header or k < key(p).
        let insert_left = p == NIL || less(k, self.key(p));
        let z = self.nodes.len();
        self.nodes.push(Node {
            score: k.0,
            id: k.1,
            value,
            left: NIL,
            right: NIL,
            parent: p,
            red: true,
        });
        if insert_left {
            if p == NIL {
                self.root = z;
                self.leftmost = z;
                self.rightmost = z;
            } else {
                self.nodes[p].left = z;
                if p == self.leftmost {
                    self.leftmost = z;
                }
            }
        } else {
            self.nodes[p].right = z;
            if p == self.rightmost {
                self.rightmost = z;
            }
        }
        self.rebalance(z);
        true
    }
    fn rotate_left(&mut self, x: usize) {
        let y = self.nodes[x].right;
        self.nodes[x].right = self.nodes[y].left;
        if self.nodes[y].left != NIL {
            let l = self.nodes[y].left;
            self.nodes[l].parent = x;
        }
        self.nodes[y].parent = self.nodes[x].parent;
        let p = self.nodes[x].parent;
        if x == self.root {
            self.root = y;
        } else if x == self.nodes[p].left {
            self.nodes[p].left = y;
        } else {
            self.nodes[p].right = y;
        }
        self.nodes[y].left = x;
        self.nodes[x].parent = y;
    }
    fn rotate_right(&mut self, x: usize) {
        let y = self.nodes[x].left;
        self.nodes[x].left = self.nodes[y].right;
        if self.nodes[y].right != NIL {
            let r = self.nodes[y].right;
            self.nodes[r].parent = x;
        }
        self.nodes[y].parent = self.nodes[x].parent;
        let p = self.nodes[x].parent;
        if x == self.root {
            self.root = y;
        } else if x == self.nodes[p].right {
            self.nodes[p].right = y;
        } else {
            self.nodes[p].left = y;
        }
        self.nodes[y].right = x;
        self.nodes[x].parent = y;
    }
    fn red(&self, x: usize) -> bool {
        x != NIL && self.nodes[x].red
    }
    fn rebalance(&mut self, mut x: usize) {
        while x != self.root && self.red(self.nodes[x].parent) {
            let p = self.nodes[x].parent;
            let xpp = self.nodes[p].parent;
            if p == self.nodes[xpp].left {
                let y = self.nodes[xpp].right;
                if self.red(y) {
                    self.nodes[p].red = false;
                    self.nodes[y].red = false;
                    self.nodes[xpp].red = true;
                    x = xpp;
                } else {
                    if x == self.nodes[p].right {
                        x = p;
                        self.rotate_left(x);
                    }
                    let p = self.nodes[x].parent;
                    self.nodes[p].red = false;
                    self.nodes[xpp].red = true;
                    self.rotate_right(xpp);
                }
            } else {
                let y = self.nodes[xpp].left;
                if self.red(y) {
                    self.nodes[p].red = false;
                    self.nodes[y].red = false;
                    self.nodes[xpp].red = true;
                    x = xpp;
                } else {
                    if x == self.nodes[p].left {
                        x = p;
                        self.rotate_right(x);
                    }
                    let p = self.nodes[x].parent;
                    self.nodes[p].red = false;
                    self.nodes[xpp].red = true;
                    self.rotate_left(xpp);
                }
            }
        }
        let r = self.root;
        self.nodes[r].red = false;
    }
    /// In-order (score, cut handle) pairs.
    pub fn ordered(&self) -> Vec<(f64, usize)> {
        let mut out = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![];
        let mut x = self.root;
        while x != NIL || !stack.is_empty() {
            while x != NIL {
                stack.push(x);
                x = self.nodes[x].left;
            }
            let n = stack.pop().unwrap();
            out.push((self.nodes[n].score, self.nodes[n].value));
            x = self.nodes[n].right;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orders_by_score_then_id() {
        let mut s = CandidateSet::new();
        for (i, score) in [-0.3, -0.5, -0.1, -0.5, -0.2].into_iter().enumerate() {
            assert!(s.insert(score, i as i32, i));
        }
        let order: Vec<_> = s.ordered().into_iter().map(|(_, v)| v).collect();
        assert_eq!(order, vec![1, 3, 0, 4, 2]);
        assert!(!s.insert(-0.5, 1, 9));
    }
}
