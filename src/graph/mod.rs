//! Physical graph input. Physical IDs never index bucket or label storage.
pub mod buckets;
pub mod relaxation;
pub mod symmetry;
use crate::{invalid, Result};
use std::collections::HashSet;
pub mod ng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Disposable,
    Equality,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub lower: f64,
    pub upper: f64,
}
impl Window {
    pub fn new(lower: f64, upper: f64) -> Result<Self> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(invalid("finite ordered resource bounds required"));
        }
        Ok(Self { lower, upper })
    }
}
impl Default for Window {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: 1e5,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vertex {
    pub bounds: [Window; 2],
    pub elements: Vec<usize>,
    pub memory: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Arc {
    pub id: i32,
    pub tail: usize,
    pub head: usize,
    pub cost: f64,
    pub reduced_cost: f64,
    pub variables: Vec<(usize, f64)>,
    pub consumption: [f64; 2],
    pub bounds: [Option<Window>; 2],
    pub elements: Vec<usize>,
    pub memory: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub vertices: Vec<Vertex>,
    pub arcs: Vec<Arc>,
    pub resources: Vec<ResourceKind>,
    pub source: usize,
    pub sink: usize,
    pub element_count: usize,
    pub elementary: bool,
    pub initial: [Option<f64>; 2],
    pub graph_id: i32,
    pub main_resource: Option<usize>,
    grouped: bool,
}
impl Graph {
    pub fn new(vertices: usize, resources: usize) -> Result<Self> {
        if vertices == 0 || vertices > 1024 || resources > 2 {
            return Err(invalid(
                "pricing accepts 1..1024 vertices and 0..2 resources",
            ));
        }
        Ok(Self {
            vertices: (0..vertices)
                .map(|v| Vertex {
                    bounds: [Window::default(); 2],
                    elements: vec![v],
                    memory: vec![v],
                })
                .collect(),
            arcs: vec![],
            resources: vec![ResourceKind::Disposable; resources],
            source: 0,
            sink: vertices - 1,
            element_count: vertices,
            elementary: false,
            initial: [None; 2],
            graph_id: -1,
            main_resource: None,
            grouped: false,
        })
    }
    pub fn add_arc(&mut self, tail: usize, head: usize, cost: f64) -> Result<i32> {
        let id = self
            .arcs
            .iter()
            .map(|a| a.id)
            .max()
            .map_or(Some(0), |id| id.checked_add(1))
            .ok_or_else(|| invalid("physical arc ID overflow"))?;
        self.add_arc_with_id(id, tail, head, cost)?;
        Ok(id)
    }
    pub fn add_arc_with_id(&mut self, id: i32, tail: usize, head: usize, cost: f64) -> Result<()> {
        if id < 0
            || self.arcs.iter().any(|a| a.id == id)
            || tail >= self.vertices.len()
            || head >= self.vertices.len()
            || !cost.is_finite()
        {
            return Err(invalid("invalid physical arc ID, endpoints, or cost"));
        }
        self.arcs.push(Arc {
            id,
            tail,
            head,
            cost,
            reduced_cost: cost,
            variables: vec![],
            consumption: [0.0; 2],
            bounds: [None; 2],
            elements: vec![],
            memory: vec![],
        });
        Ok(())
    }
    pub fn arc_mut(&mut self, id: i32) -> Result<&mut Arc> {
        self.arcs
            .iter_mut()
            .find(|a| a.id == id)
            .ok_or_else(|| invalid("unknown physical arc ID"))
    }
    pub fn set_bounds(
        &mut self,
        vertex: usize,
        resource: usize,
        lower: f64,
        upper: f64,
    ) -> Result<()> {
        if resource >= self.resources.len() {
            return Err(invalid("resource ID out of range"));
        }
        let window = Window::new(lower, upper)?;
        self.vertices
            .get_mut(vertex)
            .ok_or_else(|| invalid("vertex out of range"))?
            .bounds[resource] = window;
        Ok(())
    }
    pub fn set_ng(&mut self, vertex: usize, elements: &[usize]) -> Result<()> {
        if vertex >= self.vertices.len() || elements.iter().any(|&e| e >= self.element_count) {
            return Err(invalid("NG vertex/element out of range"));
        }
        let mut next = elements.to_vec();
        next.sort_unstable();
        next.dedup();
        self.vertices[vertex].memory = next;
        Ok(())
    }
    pub fn add_elementarity_set(&mut self, members: &[usize]) -> Result<usize> {
        if members.iter().any(|&v| v >= self.vertices.len()) {
            return Err(invalid("elementarity member out of range"));
        }
        if !self.grouped {
            for v in &mut self.vertices {
                v.elements.clear();
                v.memory.clear();
            }
            self.element_count = 0;
            self.grouped = true;
        }
        if self.element_count >= 1024 {
            return Err(invalid("elementarity set capacity"));
        }
        let id = self.element_count;
        self.element_count += 1;
        for &v in members {
            self.vertices[v].elements.push(id);
        }
        Ok(id)
    }
    /// Replaces costs transactionally. Costs are indexed in physical input order.
    pub fn update_reduced_costs(
        &mut self,
        costs: &[f64],
        duals: &[f64],
        rounded: bool,
    ) -> Result<()> {
        if costs.len() != self.arcs.len()
            || duals.len() != self.vertices.len()
            || costs.iter().chain(duals).any(|v| !v.is_finite())
        {
            return Err(invalid(
                "finite pricing costs/duals with matching dimensions required",
            ));
        }
        let reduced: Vec<_> = self
            .arcs
            .iter()
            .zip(costs)
            .map(|(a, &c)| {
                let raw = c - duals[a.head];
                if rounded {
                    (raw * 1e8 + 0.5).floor() / 1e8
                } else {
                    raw
                }
            })
            .collect();
        if reduced.iter().any(|v| !v.is_finite()) {
            return Err(invalid("reduced cost overflow"));
        }
        for ((a, &c), r) in self.arcs.iter_mut().zip(costs).zip(reduced) {
            a.cost = c;
            a.reduced_cost = r;
        }
        Ok(())
    }
    pub(crate) fn validate(&self) -> Result<()> {
        let n = self.vertices.len();
        if n == 0
            || n > 1024
            || self.source >= n
            || self.sink >= n
            || self.source == self.sink
            || self.resources.len() > 2
            || self.element_count > 1024
        {
            return Err(invalid("invalid graph dimensions or distinct terminals"));
        }
        let mut ids = HashSet::new();
        for a in &self.arcs {
            if a.id < 0
                || !ids.insert(a.id)
                || a.tail >= n
                || a.head >= n
                || a.head == self.source
                || a.tail == self.sink
                || !a.cost.is_finite()
                || !a.reduced_cost.is_finite()
                || a.consumption.iter().any(|q| !q.is_finite())
            {
                return Err(invalid("invalid physical arc"));
            }
            for window in a.bounds.iter().flatten() {
                Window::new(window.lower, window.upper)?;
            }
            for (r, k) in self.resources.iter().enumerate() {
                if *k == ResourceKind::Disposable && a.consumption[r] < 0.0 {
                    return Err(invalid("negative disposable consumption is not supported"));
                }
            }
        }
        for v in &self.vertices {
            for r in 0..self.resources.len() {
                Window::new(v.bounds[r].lower, v.bounds[r].upper)?;
            }
        }
        for value in self.initial.iter().flatten() {
            if !value.is_finite() {
                return Err(invalid("finite initial resource required"));
            }
        }
        Ok(())
    }
}
impl Graph {
    pub fn configure_ng(
        &mut self,
        sets: usize,
        mut membership: Vec<Vec<usize>>,
        mut memory: Vec<Vec<usize>>,
    ) -> Result<()> {
        if sets > 1024
            || membership.len() != self.vertices.len()
            || memory.len() != self.vertices.len()
        {
            return Err(invalid("NG membership dimensions"));
        }
        for row in membership.iter_mut().chain(&mut memory) {
            row.sort_unstable();
            row.dedup();
            if row.iter().any(|&e| e >= sets) {
                return Err(invalid("NG set out of range"));
            }
        }
        self.element_count = sets;
        self.grouped = true;
        for ((v, e), m) in self.vertices.iter_mut().zip(membership).zip(memory) {
            v.elements = e;
            v.memory = m;
        }
        for a in &mut self.arcs {
            a.elements.clear();
            a.memory.clear();
        }
        Ok(())
    }
}
