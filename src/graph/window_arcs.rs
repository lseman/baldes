//! Arc copies that fix the service-start window of a multi-window head vertex.
//!
//! A route that enters vertex `i` through the copy for window `w` starts its
//! service at `i` inside `w`, waiting for it when necessary. The number of
//! times a route selects each window is then an integer combination of arc
//! traversals, so window-selection rows are arc-additive in pricing and every
//! window choice remains reachable under any dual prices.
#[cfg(feature = "multi-time-windows")]
use super::*;

/// One arc into a multi-window vertex, restricted to a single window. The
/// original arc keeps its ID and serves the first feasible window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowArc {
    pub arc: i32,
    pub original: i32,
    pub vertex: usize,
    /// Index into `Graph::time_windows(vertex, resource)`.
    pub window: usize,
}

/// The window copies created by `Graph::split_time_window_arcs`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WindowArcs {
    pub resource: usize,
    /// Every arc into a multi-window vertex, original arcs included.
    pub arcs: Vec<WindowArc>,
}

impl WindowArcs {
    /// The input arc ID of a (possibly copied) arc.
    pub fn original(&self, arc: i32) -> i32 {
        self.arcs
            .iter()
            .find(|a| a.arc == arc)
            .map_or(arc, |a| a.original)
    }
    /// Rewrite a route of the split graph in input arc IDs.
    pub fn restore(&self, route: &mut crate::Route) {
        let map: std::collections::HashMap<_, _> =
            self.arcs.iter().map(|a| (a.arc, a.original)).collect();
        for id in &mut route.arcs {
            *id = map.get(id).copied().unwrap_or(*id);
        }
    }
    /// Copies grouped by (vertex, window), in vertex and window order.
    pub fn selections(&self) -> std::collections::BTreeMap<(usize, usize), Vec<i32>> {
        let mut out = std::collections::BTreeMap::<_, Vec<i32>>::new();
        for a in &self.arcs {
            out.entry((a.vertex, a.window)).or_default().push(a.arc);
        }
        for arcs in out.values_mut() {
            arcs.sort_unstable();
        }
        out
    }
}

#[cfg(feature = "multi-time-windows")]
impl<const D: usize> Graph<D> {
    /// Copy every arc into a vertex with two or more time windows on
    /// `resource`, once per window that the arc can reach; each copy's head
    /// bound is that window. Windows are intersected with the vertex bound and
    /// any existing arc bound, and empty intersections get no copy. Arcs into
    /// single-window vertices are unchanged. Repeating the call adds nothing.
    pub fn split_time_window_arcs(&mut self, resource: usize) -> Result<WindowArcs> {
        if self.resources.get(resource) != Some(&ResourceKind::Disposable) {
            return Err(invalid("window splitting requires a disposable resource"));
        }
        let mut next = self
            .arcs
            .iter()
            .map(|a| a.id)
            .max()
            .map_or(Some(0), |id| id.checked_add(1))
            .ok_or_else(|| invalid("physical arc ID overflow"))?;
        let mut out = WindowArcs {
            resource,
            arcs: vec![],
        };
        for index in 0..self.arcs.len() {
            let head = self.arcs[index].head;
            let windows = self.time_windows(head, resource);
            if windows.len() < 2 {
                continue;
            }
            let bound = self.vertices[head].bounds[resource];
            let arc_bound = self.arcs[index].bounds[resource];
            let feasible: Vec<_> = windows
                .iter()
                .enumerate()
                .filter_map(|(w, window)| {
                    let mut lower = window.lower.max(bound.lower);
                    let mut upper = window.upper.min(bound.upper);
                    if let Some(a) = arc_bound {
                        lower = lower.max(a.lower);
                        upper = upper.min(a.upper);
                    }
                    (lower <= upper).then_some((w, Window { lower, upper }))
                })
                .collect();
            let Some(&(first, window)) = feasible.first() else {
                continue;
            };
            let original = self.arcs[index].id;
            self.arcs[index].bounds[resource] = Some(window);
            out.arcs.push(WindowArc {
                arc: original,
                original,
                vertex: head,
                window: first,
            });
            for &(w, window) in &feasible[1..] {
                let mut copy = self.arcs[index].clone();
                copy.id = next;
                copy.bounds[resource] = Some(window);
                out.arcs.push(WindowArc {
                    arc: next,
                    original,
                    vertex: head,
                    window: w,
                });
                self.arcs.push(copy);
                next = next
                    .checked_add(1)
                    .ok_or_else(|| invalid("physical arc ID overflow"))?;
            }
        }
        Ok(out)
    }

    /// Re-express a route given in input arc IDs on the split graph: replay its
    /// earliest schedule on `resource` and enter each multi-window vertex
    /// through the copy of the window that contains the service start.
    pub fn route_on_window_arcs(&self, split: &WindowArcs, arcs: &[i32]) -> Option<Vec<i32>> {
        let r = split.resource;
        let mut q = self.root_resource(r, true)?;
        let mut out = Vec::with_capacity(arcs.len());
        for &id in arcs {
            let arc = self.arcs.iter().find(|a| a.id == id)?;
            let reach = q + arc.consumption[r];
            let copies: Vec<_> = split.arcs.iter().filter(|a| a.original == id).collect();
            if copies.is_empty() {
                let bounds = arc_window(self.vertices[arc.head].bounds[r], arc.bounds[r]);
                q = self.project_resource(arc.head, r, bounds, reach, true)?;
                out.push(id);
                continue;
            }
            // Earliest start among the copies: the window the schedule reaches first.
            let (start, copy) = copies
                .iter()
                .filter_map(|c| {
                    let a = self.arcs.iter().find(|a| a.id == c.arc)?;
                    let bounds = arc_window(self.vertices[a.head].bounds[r], a.bounds[r]);
                    Some((
                        self.project_resource(a.head, r, bounds, reach, true)?,
                        c.arc,
                    ))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0))?;
            q = start;
            out.push(copy);
        }
        Some(out)
    }
}

#[cfg(feature = "multi-time-windows")]
fn arc_window(vertex: Window, arc: Option<Window>) -> Window {
    arc.map_or(vertex, |a| Window {
        lower: vertex.lower.max(a.lower),
        upper: vertex.upper.min(a.upper),
    })
}

#[cfg(all(test, feature = "multi-time-windows"))]
#[path = "../../tests/unit/graph/window_arcs_tests.rs"]
mod tests;
