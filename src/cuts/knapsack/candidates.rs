use super::*;

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub(super) row: usize,
    pub(super) score: f64,
    pub(super) kind: KnapsackKind,
    pub(super) numerator: i32,
    pub(super) denominator: i32,
    pub(super) rhs: i32,
    pub(super) multiplier: f64,
    pub(super) thresholds: Vec<i32>,
    pub(super) breakpoints: Vec<i32>,
    pub(super) coefficients: BTreeMap<i32, i32>,
}

pub(super) fn candidate_less(lhs: &Candidate, rhs: &Candidate) -> bool {
    if lhs.score < rhs.score {
        return true;
    }
    if rhs.score < lhs.score {
        return false;
    }
    if lhs.multiplier < rhs.multiplier {
        return true;
    }
    if lhs.rhs < rhs.rhs {
        return true;
    }
    lhs.breakpoints < rhs.breakpoints
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Color {
    Red,
    Black,
}

pub(super) struct RbNode {
    pub(super) val: Candidate,
    pub(super) color: Color,
    pub(super) parent: Option<usize>,
    pub(super) left: Option<usize>,
    pub(super) right: Option<usize>,
}

#[derive(Default)]
pub(super) struct CandidateTree {
    pub(super) nodes: Vec<RbNode>,
    pub(super) free: Vec<usize>,
    pub(super) root: Option<usize>,
    pub(super) len: usize,
}

impl CandidateTree {
    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn alloc(&mut self, val: Candidate) -> usize {
        let node = RbNode {
            val,
            color: Color::Red,
            parent: None,
            left: None,
            right: None,
        };
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = node;
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(node);
            idx
        }
    }

    pub(super) fn minimum(&self, mut x: usize) -> usize {
        while let Some(left) = self.nodes[x].left {
            x = left;
        }
        x
    }

    pub(super) fn maximum(&self, mut x: usize) -> usize {
        while let Some(right) = self.nodes[x].right {
            x = right;
        }
        x
    }

    pub(super) fn predecessor(&self, mut x: usize) -> Option<usize> {
        if let Some(left) = self.nodes[x].left {
            return Some(self.maximum(left));
        }
        let mut y = self.nodes[x].parent;
        while let Some(p) = y {
            if self.nodes[p].left == Some(x) {
                x = p;
                y = self.nodes[p].parent;
            } else {
                break;
            }
        }
        y
    }

    pub(super) fn insert(&mut self, val: Candidate) -> bool {
        let mut x = self.root;
        let mut y = None;
        let mut comp = true;
        while let Some(curr) = x {
            y = Some(curr);
            comp = candidate_less(&val, &self.nodes[curr].val);
            x = if comp {
                self.nodes[curr].left
            } else {
                self.nodes[curr].right
            };
        }
        let mut j = y;
        if comp {
            if let Some(y_idx) = y {
                if Some(y_idx) == self.root.map(|r| self.minimum(r)) {
                    self.insert_at(true, y_idx, val);
                    return true;
                }
                j = self.predecessor(y_idx);
            } else {
                let z = self.alloc(val);
                self.nodes[z].color = Color::Black;
                self.root = Some(z);
                self.len += 1;
                return true;
            }
        }
        if let Some(j_idx) = j {
            if candidate_less(&self.nodes[j_idx].val, &val) {
                let insert_left = x.is_some()
                    || y.map_or(true, |y_idx| candidate_less(&val, &self.nodes[y_idx].val));
                self.insert_at(insert_left, y.unwrap(), val);
                return true;
            }
        }
        false
    }

    pub(super) fn insert_at(&mut self, insert_left: bool, p: usize, val: Candidate) {
        let z = self.alloc(val);
        self.nodes[z].parent = Some(p);
        if insert_left {
            self.nodes[p].left = Some(z);
        } else {
            self.nodes[p].right = Some(z);
        }
        self.insert_fixup(z);
        self.len += 1;
    }

    pub(super) fn rotate_left(&mut self, x: usize) {
        let y = self.nodes[x].right.unwrap();
        let y_left = self.nodes[y].left;
        self.nodes[x].right = y_left;
        if let Some(yl) = y_left {
            self.nodes[yl].parent = Some(x);
        }
        let x_parent = self.nodes[x].parent;
        self.nodes[y].parent = x_parent;
        if let Some(xp) = x_parent {
            if self.nodes[xp].left == Some(x) {
                self.nodes[xp].left = Some(y);
            } else {
                self.nodes[xp].right = Some(y);
            }
        } else {
            self.root = Some(y);
        }
        self.nodes[y].left = Some(x);
        self.nodes[x].parent = Some(y);
    }

    pub(super) fn rotate_right(&mut self, x: usize) {
        let y = self.nodes[x].left.unwrap();
        let y_right = self.nodes[y].right;
        self.nodes[x].left = y_right;
        if let Some(yr) = y_right {
            self.nodes[yr].parent = Some(x);
        }
        let x_parent = self.nodes[x].parent;
        self.nodes[y].parent = x_parent;
        if let Some(xp) = x_parent {
            if self.nodes[xp].right == Some(x) {
                self.nodes[xp].right = Some(y);
            } else {
                self.nodes[xp].left = Some(y);
            }
        } else {
            self.root = Some(y);
        }
        self.nodes[y].right = Some(x);
        self.nodes[x].parent = Some(y);
    }

    pub(super) fn color_of(&self, node: Option<usize>) -> Color {
        node.map_or(Color::Black, |idx| self.nodes[idx].color)
    }

    pub(super) fn insert_fixup(&mut self, mut z: usize) {
        while let Some(p) = self.nodes[z].parent {
            if self.nodes[p].color != Color::Red {
                break;
            }
            let gp = self.nodes[p].parent.unwrap();
            if Some(p) == self.nodes[gp].left {
                let y = self.nodes[gp].right;
                if self.color_of(y) == Color::Red {
                    self.nodes[p].color = Color::Black;
                    self.nodes[y.unwrap()].color = Color::Black;
                    self.nodes[gp].color = Color::Red;
                    z = gp;
                } else {
                    if Some(z) == self.nodes[p].right {
                        z = p;
                        self.rotate_left(z);
                    }
                    let p2 = self.nodes[z].parent.unwrap();
                    let gp2 = self.nodes[p2].parent.unwrap();
                    self.nodes[p2].color = Color::Black;
                    self.nodes[gp2].color = Color::Red;
                    self.rotate_right(gp2);
                }
            } else {
                let y = self.nodes[gp].left;
                if self.color_of(y) == Color::Red {
                    self.nodes[p].color = Color::Black;
                    self.nodes[y.unwrap()].color = Color::Black;
                    self.nodes[gp].color = Color::Red;
                    z = gp;
                } else {
                    if Some(z) == self.nodes[p].left {
                        z = p;
                        self.rotate_right(z);
                    }
                    let p2 = self.nodes[z].parent.unwrap();
                    let gp2 = self.nodes[p2].parent.unwrap();
                    self.nodes[p2].color = Color::Black;
                    self.nodes[gp2].color = Color::Red;
                    self.rotate_left(gp2);
                }
            }
        }
        if let Some(r) = self.root {
            self.nodes[r].color = Color::Black;
        }
    }

    pub(super) fn transplant(&mut self, u: usize, v: Option<usize>) {
        let up = self.nodes[u].parent;
        if let Some(p) = up {
            if self.nodes[p].left == Some(u) {
                self.nodes[p].left = v;
            } else {
                self.nodes[p].right = v;
            }
        } else {
            self.root = v;
        }
        if let Some(v_idx) = v {
            self.nodes[v_idx].parent = up;
        }
    }

    pub(super) fn pop_last(&mut self) {
        let Some(r) = self.root else { return };
        let z = self.maximum(r);
        let y_original_color = self.nodes[z].color;
        let x = self.nodes[z].left;
        let x_parent = self.nodes[z].parent;
        self.transplant(z, x);
        if y_original_color == Color::Black {
            self.delete_fixup(x, x_parent);
        }
        self.free.push(z);
        self.len -= 1;
    }

    pub(super) fn delete_fixup(&mut self, mut x: Option<usize>, mut x_parent: Option<usize>) {
        while x != self.root && self.color_of(x) == Color::Black {
            let Some(p) = x_parent else { break };
            if x == self.nodes[p].left {
                let mut w = self.nodes[p].right.unwrap();
                if self.nodes[w].color == Color::Red {
                    self.nodes[w].color = Color::Black;
                    self.nodes[p].color = Color::Red;
                    self.rotate_left(p);
                    w = self.nodes[p].right.unwrap();
                }
                if self.color_of(self.nodes[w].left) == Color::Black
                    && self.color_of(self.nodes[w].right) == Color::Black
                {
                    self.nodes[w].color = Color::Red;
                    x = Some(p);
                    x_parent = self.nodes[p].parent;
                } else {
                    if self.color_of(self.nodes[w].right) == Color::Black {
                        if let Some(wl) = self.nodes[w].left {
                            self.nodes[wl].color = Color::Black;
                        }
                        self.nodes[w].color = Color::Red;
                        self.rotate_right(w);
                        w = self.nodes[p].right.unwrap();
                    }
                    self.nodes[w].color = self.nodes[p].color;
                    self.nodes[p].color = Color::Black;
                    if let Some(wr) = self.nodes[w].right {
                        self.nodes[wr].color = Color::Black;
                    }
                    self.rotate_left(p);
                    x = self.root;
                    x_parent = None;
                }
            } else {
                let mut w = self.nodes[p].left.unwrap();
                if self.nodes[w].color == Color::Red {
                    self.nodes[w].color = Color::Black;
                    self.nodes[p].color = Color::Red;
                    self.rotate_right(p);
                    w = self.nodes[p].left.unwrap();
                }
                if self.color_of(self.nodes[w].right) == Color::Black
                    && self.color_of(self.nodes[w].left) == Color::Black
                {
                    self.nodes[w].color = Color::Red;
                    x = Some(p);
                    x_parent = self.nodes[p].parent;
                } else {
                    if self.color_of(self.nodes[w].left) == Color::Black {
                        if let Some(wr) = self.nodes[w].right {
                            self.nodes[wr].color = Color::Black;
                        }
                        self.nodes[w].color = Color::Red;
                        self.rotate_left(w);
                        w = self.nodes[p].left.unwrap();
                    }
                    self.nodes[w].color = self.nodes[p].color;
                    self.nodes[p].color = Color::Black;
                    if let Some(wl) = self.nodes[w].left {
                        self.nodes[wl].color = Color::Black;
                    }
                    self.rotate_right(p);
                    x = self.root;
                    x_parent = None;
                }
            }
        }
        if let Some(x_idx) = x {
            self.nodes[x_idx].color = Color::Black;
        }
    }

    pub(super) fn into_vec(self) -> Vec<Candidate> {
        let mut out = Vec::with_capacity(self.len);
        self.collect(self.root, &mut out);
        out
    }

    pub(super) fn collect(&self, node: Option<usize>, out: &mut Vec<Candidate>) {
        if let Some(idx) = node {
            self.collect(self.nodes[idx].left, out);
            out.push(self.nodes[idx].val.clone());
            self.collect(self.nodes[idx].right, out);
        }
    }
}

pub(super) fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
