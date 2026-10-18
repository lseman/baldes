use std::collections::{BTreeMap, BTreeSet};

use crate::core::options::Options;
use crate::{invalid, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackParameters {
    pub max_cuts_per_round: usize,
    pub use_rounding: bool,
    pub max_one_k: i32,
    pub violation_tolerance: f64,
}

impl Default for KnapsackParameters {
    fn default() -> Self {
        Self {
            max_cuts_per_round: 100,
            use_rounding: false,
            max_one_k: 1,
            violation_tolerance: 0.02,
        }
    }
}

pub fn knapsack_parameters(
    options: &Options,
    initial: KnapsackParameters,
) -> Result<KnapsackParameters> {
    let mut parameters = initial;
    let mode = options.get("RCSPresConsKnapsackCutsMode")?;
    parameters.use_rounding = mode > 0.0;
    parameters.max_one_k = mode.abs() as i32;
    let per_round = options.get("RCSPresConsKnapsackCutsMaxNumPerRound")?;
    if per_round < 1.0 {
        return Err(invalid(
            "RCSPresConsKnapsackCutsMaxNumPerRound must be positive",
        ));
    }
    parameters.max_cuts_per_round = per_round as usize;
    validate_parameters(&parameters)?;
    Ok(parameters)
}

fn validate_parameters(parameters: &KnapsackParameters) -> Result<()> {
    if parameters.max_cuts_per_round < 1 || parameters.max_cuts_per_round > i32::MAX as usize {
        return Err(invalid("knapsack max_cuts_per_round must be at least 1"));
    }
    if parameters.max_one_k < 0 {
        return Err(invalid("knapsack max_one_k must be non-negative"));
    }
    if !parameters.violation_tolerance.is_finite() || parameters.violation_tolerance < 0.0 {
        return Err(invalid("knapsack violation_tolerance must be non-negative"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnapsackKind {
    Rounding,
    Steps,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteLoadKnapsackCut {
    pub kind: KnapsackKind,
    pub numerator: i32,
    pub denominator: i32,
    pub rhs: i32,
    pub lhs: f64,
    pub violation: f64,
    pub thresholds: Vec<i32>,
    pub coefficients: BTreeMap<i32, i32>,
}

impl RouteLoadKnapsackCut {
    pub fn coefficient(&self, load: i32) -> i32 {
        if load < 1 {
            return 0;
        }
        match self.coefficients.range(..=load).next_back() {
            Some((_, &val)) => val,
            None => 0,
        }
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    row: usize,
    score: f64,
    kind: KnapsackKind,
    numerator: i32,
    denominator: i32,
    rhs: i32,
    multiplier: f64,
    thresholds: Vec<i32>,
    breakpoints: Vec<i32>,
    coefficients: BTreeMap<i32, i32>,
}

fn candidate_less(lhs: &Candidate, rhs: &Candidate) -> bool {
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
enum Color {
    Red,
    Black,
}

struct RbNode {
    val: Candidate,
    color: Color,
    parent: Option<usize>,
    left: Option<usize>,
    right: Option<usize>,
}

#[derive(Default)]
struct CandidateTree {
    nodes: Vec<RbNode>,
    free: Vec<usize>,
    root: Option<usize>,
    len: usize,
}

impl CandidateTree {
    fn len(&self) -> usize {
        self.len
    }

    fn alloc(&mut self, val: Candidate) -> usize {
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

    fn minimum(&self, mut x: usize) -> usize {
        while let Some(left) = self.nodes[x].left {
            x = left;
        }
        x
    }

    fn maximum(&self, mut x: usize) -> usize {
        while let Some(right) = self.nodes[x].right {
            x = right;
        }
        x
    }

    fn predecessor(&self, mut x: usize) -> Option<usize> {
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

    fn insert(&mut self, val: Candidate) -> bool {
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

    fn insert_at(&mut self, insert_left: bool, p: usize, val: Candidate) {
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

    fn rotate_left(&mut self, x: usize) {
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

    fn rotate_right(&mut self, x: usize) {
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

    fn color_of(&self, node: Option<usize>) -> Color {
        node.map_or(Color::Black, |idx| self.nodes[idx].color)
    }

    fn insert_fixup(&mut self, mut z: usize) {
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

    fn transplant(&mut self, u: usize, v: Option<usize>) {
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

    fn pop_last(&mut self) {
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

    fn delete_fixup(&mut self, mut x: Option<usize>, mut x_parent: Option<usize>) {
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

    fn into_vec(self) -> Vec<Candidate> {
        let mut out = Vec::with_capacity(self.len);
        self.collect(self.root, &mut out);
        out
    }

    fn collect(&self, node: Option<usize>, out: &mut Vec<Candidate>) {
        if let Some(idx) = node {
            self.collect(self.nodes[idx].left, out);
            out.push(self.nodes[idx].val.clone());
            self.collect(self.nodes[idx].right, out);
        }
    }
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

#[derive(Clone, Debug, Default)]
pub struct KnapsackSeparator {
    pub parameters: KnapsackParameters,
}

impl KnapsackSeparator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parameters(parameters: KnapsackParameters) -> Result<Self> {
        validate_parameters(&parameters)?;
        Ok(Self { parameters })
    }

    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(invalid("knapsack parameter must be finite"));
        }
        match name {
            "max_cuts_per_round" => {
                if !value.is_finite() || value < 1.0 || value > i32::MAX as f64 {
                    return Err(invalid("knapsack max_cuts_per_round must be at least 1"));
                }
                self.parameters.max_cuts_per_round = value as usize;
            }
            "use_rounding" => {
                self.parameters.use_rounding = value != 0.0;
            }
            "max_one_k" => {
                if !value.is_finite() || value < 0.0 || value > i32::MAX as f64 {
                    return Err(invalid("knapsack max_one_k must be non-negative"));
                }
                self.parameters.max_one_k = value as i32;
            }
            "violation_tolerance" => {
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid("knapsack violation_tolerance must be non-negative"));
                }
                self.parameters.violation_tolerance = value;
            }
            _ => return Err(invalid(format!("unknown knapsack parameter: {name}"))),
        }
        Ok(())
    }

    pub fn get_parameter(&self, name: &str) -> Result<f64> {
        match name {
            "max_cuts_per_round" => Ok(self.parameters.max_cuts_per_round as f64),
            "use_rounding" => Ok(if self.parameters.use_rounding {
                1.0
            } else {
                0.0
            }),
            "max_one_k" => Ok(self.parameters.max_one_k as f64),
            "violation_tolerance" => Ok(self.parameters.violation_tolerance),
            _ => Err(invalid(format!("unknown knapsack parameter: {name}"))),
        }
    }

    pub fn configure(&mut self, options: &Options) -> Result<()> {
        self.parameters = knapsack_parameters(options, self.parameters.clone())?;
        Ok(())
    }

    pub fn separate(
        &self,
        loads: &[i32],
        values: &[f64],
        rhs: i32,
    ) -> Result<Vec<RouteLoadKnapsackCut>> {
        if loads.len() != values.len() {
            return Err(invalid("loads and values must have the same length"));
        }
        if rhs < 0 || rhs > i32::MAX / 8 {
            return Err(invalid("knapsack rhs is out of range"));
        }
        validate_parameters(&self.parameters)?;
        for &value in values {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid("knapsack values must be finite and non-negative"));
            }
        }
        for &load in loads {
            if load < 0 || load > i32::MAX / 8 {
                return Err(invalid(
                    "knapsack loads must be non-negative and within range",
                ));
            }
        }

        let mut totals: BTreeMap<i32, f64> = BTreeMap::new();
        for (&load, &val) in loads.iter().zip(values.iter()) {
            *totals.entry(load).or_insert(0.0) += (val * 1e8 + 0.5).floor() / 1e8;
        }

        let mut result: Vec<_> = self
            .separate_totals(&[(totals, rhs)])
            .into_iter()
            .map(|(_, cut)| cut)
            .collect();
        // The standalone adapter reports LHS in original column order,
        // independently of the load aggregation used by separation.
        for cut in &mut result {
            cut.lhs = 0.;
            for (&load, &value) in loads.iter().zip(values) {
                cut.lhs += ((value * 1e8 + 0.5).floor() / 1e8) * cut.coefficient(load) as f64;
            }
            cut.violation = cut.lhs - cut.rhs as f64;
        }
        Ok(result)
    }

    // The original bounded candidate tree is shared by all original rows.
    fn separate_totals(
        &self,
        rows: &[(BTreeMap<i32, f64>, i32)],
    ) -> Vec<(usize, RouteLoadKnapsackCut)> {
        let mut candidates = CandidateTree::default();
        for (row, (totals, rhs)) in rows.iter().enumerate() {
            if self.parameters.use_rounding {
                self.separation_by_rounding(totals, *rhs, row, &mut candidates);
            }
            if self.parameters.max_one_k > 0 {
                self.separation_of_one_k_inequalities(totals, *rhs, row, &mut candidates);
            }
        }
        let ordered = candidates.into_vec();
        let mut result = Vec::with_capacity(ordered.len());
        for cand in ordered {
            let mut cut = RouteLoadKnapsackCut {
                kind: cand.kind,
                numerator: cand.numerator,
                denominator: cand.denominator,
                rhs: cand.rhs,
                lhs: 0.0,
                violation: 0.0,
                thresholds: cand.thresholds,
                coefficients: cand.coefficients,
            };
            for (&load, &value) in &rows[cand.row].0 {
                cut.lhs += value * cut.coefficient(load) as f64;
            }
            cut.violation = cut.lhs - (cut.rhs as f64);
            result.push((cand.row, cut));
        }
        result
    }

    fn separation_by_rounding(
        &self,
        totals: &BTreeMap<i32, f64>,
        rhs: i32,
        row: usize,
        candidates: &mut CandidateTree,
    ) {
        for &q in totals.keys() {
            if q >= 1 {
                for j in 1..=q {
                    let multiplier = (j as f64) / (q as f64);
                    let target = (multiplier * (rhs as f64) + 1e-6).floor() as i32;
                    let mut lhs = 0.0;
                    for (&load, &val) in totals {
                        lhs += ((load as f64) * multiplier + 1e-6).floor() * val;
                    }
                    if (target as f64) * (lhs - target as f64) > self.parameters.violation_tolerance
                    {
                        let mut coefficients = BTreeMap::new();
                        coefficients.insert(0, 0);
                        for k in 1..=target {
                            let step = ((k as f64) / multiplier - 1e-6).ceil() as i32;
                            coefficients.entry(step).or_insert(k);
                        }
                        let g = gcd(j, q);
                        let score = -(lhs - target as f64) / (target as f64);
                        candidates.insert(Candidate {
                            row,
                            score,
                            kind: KnapsackKind::Rounding,
                            numerator: j / g,
                            denominator: q / g,
                            rhs: target,
                            multiplier,
                            thresholds: Vec::new(),
                            breakpoints: Vec::new(),
                            coefficients,
                        });
                        if candidates.len() > self.parameters.max_cuts_per_round {
                            candidates.pop_last();
                        }
                    }
                }
            }
        }
    }

    fn separation_of_one_k_inequalities(
        &self,
        totals: &BTreeMap<i32, f64>,
        rhs: i32,
        row: usize,
        candidates: &mut CandidateTree,
    ) {
        if totals.is_empty() {
            return;
        }
        let mut unique_vals = BTreeSet::new();
        for &load in totals.keys() {
            let folded = if (rhs + 1) / 2 < load {
                rhs + 1 - load
            } else {
                load
            };
            unique_vals.insert(folded);
        }
        let vals: Vec<i32> = unique_vals.into_iter().collect();

        let mut emit = |low: &[i32]| {
            let cut_rhs = (low.len() * 2) as i32;
            let mut breakpoints = Vec::with_capacity(cut_rhs as usize + 1);
            breakpoints.push(0);
            breakpoints.extend_from_slice(low);
            for &v in low.iter().rev() {
                breakpoints.push(rhs + 1 - v);
            }

            let mut count = 0usize;
            let mut lhs = 0.0;
            for (&load, &val) in totals {
                while count < cut_rhs as usize && breakpoints[count + 1] <= load {
                    count += 1;
                }
                lhs += (count as f64) * val;
            }
            if self.parameters.violation_tolerance + (cut_rhs as f64) < lhs {
                let mut coefficients = BTreeMap::new();
                coefficients.insert(0, 0);
                for i in 1..=(cut_rhs as usize) {
                    if i == cut_rhs as usize || breakpoints[i] < breakpoints[i + 1] {
                        coefficients.entry(breakpoints[i]).or_insert(i as i32);
                    }
                }
                let thresholds = breakpoints[1..].to_vec();
                let score = ((cut_rhs as f64) - lhs) / (cut_rhs as f64);
                candidates.insert(Candidate {
                    row,
                    score,
                    kind: KnapsackKind::Steps,
                    numerator: 1,
                    denominator: 1,
                    rhs: cut_rhs,
                    multiplier: 0.0,
                    thresholds,
                    breakpoints,
                    coefficients,
                });
                if candidates.len() > self.parameters.max_cuts_per_round {
                    candidates.pop_last();
                }
            }
        };

        let size = vals.len();
        for i in 0..size {
            for j in i..size {
                if vals[j] > 2 * vals[i] {
                    break;
                }
                for k in j..size {
                    let (a, b, c) = (vals[i], vals[j], vals[k]);
                    if c <= a + b && rhs < a + 2 * c && rhs < c + 2 * b {
                        emit(&[a, b, c]);
                    }
                }
            }
        }
        if self.parameters.max_one_k > 7 {
            for i in 0..size {
                for j in i..size {
                    if vals[j] > 2 * vals[i] {
                        break;
                    }
                    for k in j..size {
                        let (a, b, c) = (vals[i], vals[j], vals[k]);
                        if c > a + b || !(rhs < 3 * c) {
                            continue;
                        }
                        for l in k..size {
                            let d = vals[l];
                            if d <= a + c && d <= 2 * b && rhs < a + 2 * d && rhs < b + c + d {
                                emit(&[a, b, c, d]);
                            }
                        }
                    }
                }
            }
        }
        if self.parameters.max_one_k > 9 {
            for i in 0..size {
                for j in i..size {
                    if vals[j] > 2 * vals[i] {
                        break;
                    }
                    for k in j..size {
                        let (a, b, c) = (vals[i], vals[j], vals[k]);
                        if c > a + b {
                            continue;
                        }
                        for l in k..size {
                            let d = vals[l];
                            if d > a + c || d > 2 * b || !(rhs < c + 2 * d) {
                                continue;
                            }
                            for m in l..size {
                                let e = vals[m];
                                if e <= a + d
                                    && e <= b + c
                                    && rhs < a + 2 * e
                                    && rhs < b + d + e
                                    && rhs < 2 * c + e
                                {
                                    emit(&[a, b, c, d, e]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// An original integer master row and its graph-specific resource/scaling.
#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackRow {
    pub rhs: i32,
    pub graph_resources: BTreeMap<i32, (usize, f64)>,
    pub load_totals: BTreeMap<i32, f64>,
}
impl KnapsackRow {
    pub fn new(graph: i32, resource: usize, scale: f64, rhs: i32) -> Self {
        Self {
            rhs,
            graph_resources: BTreeMap::from([(graph, (resource, scale))]),
            load_totals: BTreeMap::new(),
        }
    }
}
/// Original graph resource metadata used by `prepareSeparation`.
/// IDs are independent of the records' positions and solver coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnapsackResource {
    pub id: usize,
    pub variable_ids: Vec<i32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnapsackGraph {
    pub id: i32,
    pub resources: Vec<KnapsackResource>,
}
/// Preparation associations use graph positions in `graphs`; `None` models
/// a null original graph pointer. Ordered positions preserve native map order.
#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackPreparationRow {
    pub rhs: i32,
    pub load_totals: BTreeMap<i32, f64>,
    pub graph_resources: BTreeMap<Option<usize>, (usize, f64)>,
}
#[derive(Clone, Debug)]
pub struct KnapsackColumn {
    pub graph_id: i32,
    pub resources: Vec<f64>,
    pub value: f64,
}
/// Native RouteLoadKnapsackCut, including its original row association.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourceKnapsackCut {
    pub id: i32,
    pub row: usize,
    pub cut_class: i32,
    pub graph_resources: BTreeMap<i32, (usize, f64)>,
    pub cut: RouteLoadKnapsackCut,
    /// Master dual; pricing uses its negative, rounded to 1e-8.
    pub dual: f64,
}
impl ResourceKnapsackCut {
    pub fn coefficient(&self, graph: i32, resources: &[f64]) -> i32 {
        let Some(&(resource, scale)) = self.graph_resources.get(&graph) else {
            return 0;
        };
        resources.get(resource).map_or(0, |q| {
            self.cut.coefficient((q * scale + 1e-6).floor() as i32)
        })
    }
}
#[derive(Clone, Debug, Default)]
pub struct ResourceKnapsackSeparator {
    pub parameters: KnapsackParameters,
    pub rows: Vec<KnapsackRow>,
    pub next_cut_id: i32,
    pub num_separate_calls: usize,
}
impl ResourceKnapsackSeparator {
    /// failures retain completed rows and the partially prepared current row.
    /// Cut identifiers and separation-call counters survive preparation.
    pub fn prepare(
        &mut self,
        graphs: &[KnapsackGraph],
        rows: &[KnapsackPreparationRow],
    ) -> Result<()> {
        self.rows.clear();
        for row in rows {
            self.rows.push(KnapsackRow {
                rhs: row.rhs,
                load_totals: row.load_totals.clone(),
                graph_resources: BTreeMap::new(),
            });
            for (&graph, &(resource, scale)) in &row.graph_resources {
                let graph = graph
                    .and_then(|index| graphs.get(index))
                    .ok_or_else(|| invalid("knapsack preparation graph"))?;
                let mut exists = false;
                for entry in &graph.resources {
                    if entry.id != resource {
                        continue;
                    }
                    if entry.variable_ids.first().is_none_or(|&id| id < 0) {
                        return Err(invalid("knapsack preparation resource variable"));
                    }
                    exists = true;
                }
                if !exists {
                    return Err(invalid("knapsack preparation resource ID"));
                }
                self.rows
                    .last_mut()
                    .unwrap()
                    .graph_resources
                    .entry(graph.id)
                    .or_insert((resource, scale));
            }
        }
        Ok(())
    }
    pub fn separate(&mut self, columns: &[KnapsackColumn]) -> Result<Vec<ResourceKnapsackCut>> {
        validate_parameters(&self.parameters)?;
        self.num_separate_calls += 1;
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            if row.rhs < 0 || row.rhs > i32::MAX / 8 {
                return Err(invalid("knapsack original rhs out of range"));
            }
            let mut totals = BTreeMap::new();
            for (&load, &value) in &row.load_totals {
                if !value.is_finite() || value < 0. {
                    return Err(invalid("knapsack load totals"));
                }
                totals.insert(load, (value * 1e8 + 0.5).floor() / 1e8);
            }
            for column in columns {
                if !column.value.is_finite() || column.value < 0. {
                    return Err(invalid("knapsack column value"));
                }
                let Some(&(resource, scale)) = row.graph_resources.get(&column.graph_id) else {
                    continue;
                };
                let q = *column
                    .resources
                    .get(resource)
                    .ok_or_else(|| invalid("knapsack sink resource missing"))?;
                let load = (q * scale + 1e-6).floor();
                if !load.is_finite() || load < 0. || load > (i32::MAX / 8) as f64 {
                    return Err(invalid("knapsack scaled load out of range"));
                }
                *totals.entry(load as i32).or_insert(0.) +=
                    (column.value * 1e8 + 0.5).floor() / 1e8;
            }
            rows.push((totals, row.rhs));
        }
        let separator = KnapsackSeparator::with_parameters(self.parameters.clone())?;
        let mut result = Vec::new();
        for (row, cut) in separator.separate_totals(&rows) {
            let b = &cut.thresholds;
            let rhs = cut.rhs as usize;
            let class = if b.is_empty() {
                0
            } else if (1..rhs).all(|i| i == rhs / 2 || b[i - 1] == b[i]) {
                2
            } else if rhs == 6 && b[0] == b[1] && b[4] == b[5] {
                3
            } else if rhs == 8 && b[0] == b[1] && b[2] == b[3] && b[4] == b[5] && b[6] == b[7] {
                4
            } else if rhs == 10 && b[0] == b[1] && b[2] == b[3] && b[6] == b[7] && b[8] == b[9] {
                5
            } else {
                cut.rhs
            };
            result.push(ResourceKnapsackCut {
                id: self.next_cut_id,
                row,
                cut_class: class,
                graph_resources: self.rows[row].graph_resources.clone(),
                cut,
                dual: 0.,
            });
            self.next_cut_id += 1;
        }
        Ok(result)
    }
}

/// retrieveActiveRouteLoadKnapCuts / updateResConsKnapCutsCoeffsMap
/// . Native pricing keys are the integer cut
/// breakpoints; scaling is applied by separation and master coefficients.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KnapsackPricing {
    pub active: bool,
    pub resources: BTreeMap<usize, BTreeMap<i32, f64>>,
}
impl KnapsackPricing {
    pub fn prepare(
        cuts: &[ResourceKnapsackCut],
        graph: i32,
        resource_count: usize,
    ) -> Result<Self> {
        let mapping = (0..resource_count).map(|r| (r, r)).collect();
        let mut result = Self::default();
        result.retrieve(cuts, graph, &mapping)?;
        Ok(result)
    }
    /// Refresh native cache slots in place. Resource IDs in cut associations
    /// are external IDs; `resource_indices` maps them to solver coordinates.
    /// a failed retrieval leaves the already processed state.
    pub fn retrieve(
        &mut self,
        cuts: &[ResourceKnapsackCut],
        graph: i32,
        resource_indices: &BTreeMap<usize, usize>,
    ) -> Result<()> {
        for &resource in resource_indices.values() {
            self.resources.entry(resource).or_default();
        }
        for steps in self.resources.values_mut() {
            steps.clear();
        }
        self.active = false;
        for cut in cuts {
            if !cut.dual.is_finite() || cut.dual > 1e-6 {
                return Err(invalid("knapsack master dual must be non-positive"));
            }
            if -cut.dual <= 1e-6 {
                continue;
            }
            let Some(&(resource, _)) = cut.graph_resources.get(&graph) else {
                continue;
            };
            self.active = true;
            let Some(&resource) = resource_indices.get(&resource) else {
                return Err(invalid("knapsack pricing resource index"));
            };
            let dual = (-cut.dual * 1e8 + 0.5).floor() / 1e8;
            let steps = self.resources.entry(resource).or_default();
            let mut previous = 0;
            for (&load, &coefficient) in &cut.cut.coefficients {
                let value = steps.range(..=load).next_back().map_or(0., |(_, &v)| v);
                steps.entry(load).or_insert(value);
                let delta = (coefficient - previous) as f64 * dual;
                for (_, cost) in steps.range_mut(load..) {
                    *cost += delta;
                }
                previous = coefficient;
            }
        }
        Ok(())
    }
    pub fn cost(&self, resources: &[f64]) -> f64 {
        let mut result = 0.;
        for (&resource, steps) in &self.resources {
            let load = (resources[resource] + 1e-6).floor() as i32;
            if let Some((_, &cost)) = steps.range(..=load).next_back() {
                result += cost;
            }
        }
        result
    }
    /// resConsKnapsackCutsDualsCost: forward consumption, or sink upper
    /// minus the backward label's remaining-resource coordinate.
    pub fn partial_cost(&self, g: &crate::Graph, resources: &[f64], forward: bool) -> f64 {
        if forward {
            return self.cost(resources);
        }
        let mut consumption = [0.; 2];
        for &r in self.resources.keys() {
            consumption[r] = g.vertices[g.sink].bounds[r].upper - resources[r];
        }
        self.cost(&consumption)
    }
    /// Native join consumption: forward + (sink upper - backward), or
    /// forward + forward for the symmetric consumption search.
    pub fn join_cost(
        &self,
        g: &crate::Graph,
        forward: &[f64],
        backward: &[f64],
        symmetric: bool,
    ) -> f64 {
        let mut consumption = [0.; 2];
        for &r in self.resources.keys() {
            consumption[r] = if symmetric {
                forward[r] + backward[r]
            } else {
                (g.vertices[g.sink].bounds[r].upper - backward[r]) + forward[r]
            };
        }
        self.cost(&consumption)
    }
}
