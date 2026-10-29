//! Physical graph input. Physical IDs never index bucket or label storage.
pub mod buckets;
pub mod coupled;
pub mod resources;
pub use resources::{ResourceValues, ScalarValues};
pub mod relaxation;
pub mod symmetry;
use crate::{invalid, Result};
use std::collections::HashSet;
pub mod ng;
mod time_windows;
mod window_arcs;
pub use window_arcs::{WindowArc, WindowArcs};

/// Original route-domain contract, separate from pricing NG relaxations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "clp", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "clp",
    serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)
)]
pub enum RouteDomain {
    #[default]
    PhysicalSimple,
    BoundedWalks {
        max_arcs: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Disposable,
    Equality,
}
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "clp", derive(serde::Deserialize, serde::Serialize))]
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
pub struct Vertex<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub bounds: ScalarValues<Window, D>,
    pub elements: Vec<usize>,
    pub memory: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Arc<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub id: i32,
    pub tail: usize,
    pub head: usize,
    pub cost: f64,
    pub reduced_cost: f64,
    pub variables: Vec<(usize, f64)>,
    pub consumption: ResourceValues<D>,
    pub bounds: ScalarValues<Option<Window>, D>,
    pub elements: Vec<usize>,
    pub memory: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Graph<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    pub vertices: Vec<Vertex<D>>,
    pub arcs: Vec<Arc<D>>,
    pub resources: Vec<ResourceKind>,
    pub source: usize,
    pub sink: usize,
    pub element_count: usize,
    pub elementary: bool,
    route_domain: RouteDomain,
    pub initial: ScalarValues<Option<f64>, D>,
    pub graph_id: i32,
    pub main_resource: Option<usize>,
    grouped: bool,
    coupled: Option<Box<coupled::CoupledKernel>>,
    #[cfg(feature = "multi-time-windows")]
    time_windows: std::collections::BTreeMap<(usize, usize), Vec<Window>>,
}
impl<const D: usize> Graph<D> {
    /// `self == other` with every arc's `reduced_cost` ignored, without
    /// cloning. The destructuring breaks the build if a field is added.
    pub fn same_topology(&self, other: &Graph<D>) -> bool {
        let Graph {
            vertices,
            arcs,
            resources,
            source,
            sink,
            element_count,
            elementary,
            route_domain,
            initial,
            graph_id,
            main_resource,
            grouped,
            coupled,
            #[cfg(feature = "multi-time-windows")]
            time_windows,
        } = self;
        if !(*source == other.source
            && *sink == other.sink
            && *element_count == other.element_count
            && *elementary == other.elementary
            && *route_domain == other.route_domain
            && *initial == other.initial
            && *graph_id == other.graph_id
            && *main_resource == other.main_resource
            && *grouped == other.grouped
            && *coupled == other.coupled
            && *resources == other.resources
            && vertices.len() == other.vertices.len()
            && arcs.len() == other.arcs.len())
        {
            return false;
        }
        #[cfg(feature = "multi-time-windows")]
        if time_windows != &other.time_windows {
            return false;
        }
        let same_arc_scalars = |a: &Arc<D>, b: &Arc<D>| {
            let Arc {
                id,
                tail,
                head,
                cost,
                reduced_cost: _,
                variables,
                consumption,
                bounds,
                elements,
                memory,
            } = a;
            *id == b.id
                && *tail == b.tail
                && *head == b.head
                && *cost == b.cost
                && *consumption == b.consumption
                && *bounds == b.bounds
                && variables.len() == b.variables.len()
                && elements.len() == b.elements.len()
                && memory.len() == b.memory.len()
        };
        // Lengths are equal here. Empty vectors skip the comparison: glibc's
        // AVX-512 memcmp on their dangling pointer takes a fault-suppression
        // assist, which made this check a third of a C201 root.
        fn same<T: PartialEq>(a: &[T], b: &[T]) -> bool {
            a.is_empty() || a == b
        }
        // Contiguous fields first; the per-arc vectors cost a pointer chase.
        arcs.iter()
            .zip(&other.arcs)
            .all(|(a, b)| same_arc_scalars(a, b))
            && *vertices == other.vertices
            && arcs.iter().zip(&other.arcs).all(|(a, b)| {
                same(&a.variables, &b.variables)
                    && same(&a.elements, &b.elements)
                    && same(&a.memory, &b.memory)
            })
    }
    pub fn new(vertices: usize, resources: usize) -> Result<Self> {
        if vertices == 0 || vertices > 1024 || resources > D {
            return Err(invalid(
                "pricing accepts 1..1024 vertices and at most the compiled scalar dimensions",
            ));
        }
        Ok(Self {
            vertices: (0..vertices)
                .map(|v| Vertex {
                    bounds: ScalarValues::filled(resources, Window::default()),
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
            route_domain: RouteDomain::PhysicalSimple,
            initial: ScalarValues::filled(resources, None),
            graph_id: -1,
            main_resource: None,
            grouped: false,
            coupled: None,
            #[cfg(feature = "multi-time-windows")]
            time_windows: Default::default(),
        })
    }
    pub fn route_domain(&self) -> RouteDomain {
        self.route_domain
    }
    /// Change the route domain and refresh installed coupled bound metadata.
    /// Invalid changes restore the previous domain and resource configuration.
    pub fn set_route_domain(&mut self, domain: RouteDomain) -> Result<()> {
        let old = self.route_domain;
        if domain == old {
            return self.validate();
        }
        let previous = self.coupled.take();
        self.route_domain = domain;
        if let Err(error) = self.validate() {
            self.route_domain = old;
            self.coupled = previous;
            return Err(error);
        }
        self.coupled = previous.clone();
        self.refresh_coupled_bounds();
        if let Err(error) = self.validate() {
            self.route_domain = old;
            self.coupled = previous;
            return Err(error);
        }
        Ok(())
    }
    pub fn max_route_arcs(&self) -> Option<usize> {
        match self.route_domain {
            RouteDomain::PhysicalSimple => None,
            RouteDomain::BoundedWalks { max_arcs } => Some(max_arcs),
        }
    }
    pub(crate) fn hard_walk_groups(&self) -> bool {
        self.max_route_arcs().is_some() && self.grouped
    }
    /// Whether every bounded-walk route is physically vertex-simple because
    /// hard groups forbid repeats or the physical graph is acyclic.
    pub(crate) fn bounded_walks_are_vertex_simple(&self) -> bool {
        self.max_route_arcs().is_some()
            && ((self.grouped
                && self.vertices.iter().enumerate().all(|(vertex, data)| {
                    vertex == self.source || vertex == self.sink || !data.elements.is_empty()
                }))
                || self.is_acyclic())
    }
    pub(crate) fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }
    pub(crate) fn topological_order(&self) -> Option<Vec<usize>> {
        let mut outgoing = vec![Vec::new(); self.vertices.len()];
        let mut indegree = vec![0usize; self.vertices.len()];
        for arc in &self.arcs {
            if arc.tail >= self.vertices.len() || arc.head >= self.vertices.len() {
                return None;
            }
            outgoing[arc.tail].push(arc.head);
            indegree[arc.head] += 1;
        }
        let mut queue: Vec<_> = (0..self.vertices.len())
            .filter(|&vertex| indegree[vertex] == 0)
            .collect();
        let mut processed = 0;
        while processed < queue.len() {
            let vertex = queue[processed];
            processed += 1;
            for &head in &outgoing[vertex] {
                indegree[head] -= 1;
                if indegree[head] == 0 {
                    queue.push(head);
                }
            }
        }
        (processed == self.vertices.len()).then_some(queue)
    }
    pub(crate) fn has_explicit_elementarity_groups(&self) -> bool {
        self.grouped
    }
    pub(crate) fn original_max_arcs(&self) -> usize {
        self.max_route_arcs()
            .unwrap_or_else(|| self.vertices.len().saturating_sub(1))
    }
    pub(crate) fn walk_memory_relaxation(&self) -> Option<Self> {
        if self.max_route_arcs().is_none() || self.grouped || self.element_count == 0 {
            return None;
        }
        let mut graph = self.clone();
        graph.element_count = 0;
        for v in &mut graph.vertices {
            v.elements.clear();
            v.memory.clear();
        }
        for a in &mut graph.arcs {
            a.elements.clear();
            a.memory.clear();
        }
        Some(graph)
    }
    /// Append a scalar resource and initialize existing vertices and arcs.
    pub fn add_resource(&mut self, kind: ResourceKind) -> Result<usize> {
        let r = self.resources.len();
        if r >= D || r >= i32::MAX as usize {
            return Err(invalid("resource count exceeds compiled scalar capacity"));
        }
        self.resources.push(kind);
        self.initial[r] = None;
        for vertex in &mut self.vertices {
            vertex.bounds[r] = Window::default();
        }
        for arc in &mut self.arcs {
            arc.consumption[r] = 0.;
            arc.bounds[r] = None;
        }
        Ok(r)
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
            consumption: ResourceValues::filled(self.resources.len(), 0.),
            bounds: ScalarValues::filled(self.resources.len(), None),
            elements: vec![],
            memory: vec![],
        });
        Ok(())
    }
    pub fn arc_mut(&mut self, id: i32) -> Result<&mut Arc<D>> {
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
    /// Add a hard group of physical arcs: a valid route may traverse at most
    /// one arc carrying this elementarity ID. Arc IDs are deduplicated.
    pub fn add_arc_elementarity_set(&mut self, members: &[i32]) -> Result<usize> {
        let mut members = members.to_vec();
        members.sort_unstable();
        members.dedup();
        if members
            .iter()
            .any(|id| !self.arcs.iter().any(|arc| arc.id == *id))
        {
            return Err(invalid("arc elementarity member out of range"));
        }
        if !self.grouped {
            for vertex in &mut self.vertices {
                vertex.elements.clear();
                vertex.memory.clear();
            }
            self.element_count = 0;
            self.grouped = true;
        }
        if self.element_count >= 1024 {
            return Err(invalid("elementarity set capacity"));
        }
        let id = self.element_count;
        self.element_count += 1;
        for arc_id in members {
            self.arcs
                .iter_mut()
                .find(|arc| arc.id == arc_id)
                .expect("arc group members were validated")
                .elements
                .push(id);
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
    /// Check the physical graph and any prepared coupled resources before use.
    pub fn validate(&self) -> Result<()> {
        if let RouteDomain::BoundedWalks { max_arcs } = self.route_domain {
            if max_arcs == 0
                || (self.elementary && !self.grouped)
                || (self.grouped && self.element_count > ng::NG_MEMORY_CAPACITY)
                || (!self.grouped
                    && self
                        .vertices
                        .iter()
                        .enumerate()
                        .any(|(i, v)| !v.elements.is_empty() && v.elements.as_slice() != [i]))
                || self.arcs.iter().any(|a| {
                    !a.elements.is_empty()
                        && (!self.grouped
                            || a.elements
                                .iter()
                                .any(|&element| element >= self.element_count))
                })
            {
                return Err(invalid("bounded walks require a positive arc limit, explicit supported elementarity groups, and no implicit full elementarity"));
            }
        }
        let n = self.vertices.len();
        if n == 0
            || n > 1024
            || self.source >= n
            || self.sink >= n
            || self.source == self.sink
            || self.initial.len() < self.resources.len()
            || self
                .vertices
                .iter()
                .any(|v| v.bounds.len() < self.resources.len())
            || self.element_count > 1024
        {
            return Err(invalid("invalid graph dimensions or distinct terminals"));
        }
        if self.vertices.iter().any(|vertex| {
            vertex
                .elements
                .iter()
                .chain(&vertex.memory)
                .any(|&element| element >= self.element_count)
        }) || self.arcs.iter().any(|arc| {
            arc.elements
                .iter()
                .chain(&arc.memory)
                .any(|&element| element >= self.element_count)
        }) {
            return Err(invalid("graph elementarity or NG element out of range"));
        }
        if !self.grouped && self.arcs.iter().any(|arc| !arc.elements.is_empty()) {
            return Err(invalid(
                "arc elementarity events require explicit elementarity groups",
            ));
        }
        self.validate_time_windows()?;
        if let Some(kernel) = &self.coupled {
            kernel.validate_for(self)?;
        }
        let main_disposable = self
            .main_resource
            .filter(|&resource| self.resources.get(resource) == Some(&ResourceKind::Disposable))
            .or_else(|| {
                self.resources
                    .iter()
                    .position(|kind| *kind == ResourceKind::Disposable)
            });
        let mut ids = HashSet::new();
        for a in &self.arcs {
            if a.consumption.len() < self.resources.len()
                || a.bounds.len() < self.resources.len()
                || a.id < 0
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
                if *k == ResourceKind::Disposable
                    && Some(r) == main_disposable
                    && a.consumption[r] < 0.0
                {
                    return Err(invalid(
                        "designated main disposable resource must have nonnegative consumption",
                    ));
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
impl<const D: usize> Graph<D> {
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
