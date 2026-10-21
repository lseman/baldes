//! Service-start domains for waiting resources. The ordinary vertex bound is
//! intersected with the union, and arc bounds further restrict the head domain.
use super::*;

impl<const D: usize> Graph<D> {
    /// Configure a nonempty union of closed service-start windows for one
    /// disposable resource. Sorts and merges overlapping/touching intervals,
    /// and replaces the vertex's ordinary bound with the union's hull.
    /// Later `set_bounds` calls intersect that hull with the retained union.
    /// Waiting is allowed between visits, including across gaps in this union.
    #[cfg(feature = "multi-time-windows")]
    pub fn set_time_windows(
        &mut self,
        vertex: usize,
        resource: usize,
        windows: &[Window],
    ) -> Result<()> {
        if vertex >= self.vertices.len()
            || self.resources.get(resource) != Some(&ResourceKind::Disposable)
            || windows.is_empty()
        {
            return Err(invalid(
                "time windows require a valid vertex, disposable resource and nonempty union",
            ));
        }
        let mut sorted = windows.to_vec();
        for window in &sorted {
            Window::new(window.lower, window.upper)?;
        }
        sorted.sort_by(|a, b| a.lower.total_cmp(&b.lower));
        let mut merged: Vec<Window> = Vec::with_capacity(sorted.len());
        for window in sorted {
            if let Some(last) = merged.last_mut() {
                if window.lower <= last.upper {
                    last.upper = last.upper.max(window.upper);
                    continue;
                }
            }
            merged.push(window);
        }
        self.vertices[vertex].bounds[resource] = Window {
            lower: merged[0].lower,
            upper: merged.last().unwrap().upper,
        };
        self.time_windows.insert((vertex, resource), merged);
        Ok(())
    }

    /// The normalized configured union, before intersection with current bounds.
    /// An empty slice means the ordinary single vertex window is used.
    #[cfg(feature = "multi-time-windows")]
    pub fn time_windows(&self, vertex: usize, resource: usize) -> &[Window] {
        self.time_windows
            .get(&(vertex, resource))
            .map_or(&[], Vec::as_slice)
    }

    /// Remove a union, retaining the current ordinary vertex bound.
    #[cfg(feature = "multi-time-windows")]
    pub fn clear_time_windows(&mut self, vertex: usize, resource: usize) {
        self.time_windows.remove(&(vertex, resource));
    }

    pub(crate) fn has_time_windows(&self) -> bool {
        #[cfg(feature = "multi-time-windows")]
        return !self.time_windows.is_empty();
        #[cfg(not(feature = "multi-time-windows"))]
        false
    }

    pub(crate) fn resource_has_time_windows(&self, resource: usize) -> bool {
        #[cfg(feature = "multi-time-windows")]
        return self.time_windows.keys().any(|&(_, r)| r == resource);
        #[cfg(not(feature = "multi-time-windows"))]
        {
            let _ = resource;
            false
        }
    }

    #[cfg(feature = "clp")]
    pub(crate) fn time_window_endpoints(&self, resource: usize) -> impl Iterator<Item = f64> + '_ {
        #[cfg(feature = "multi-time-windows")]
        {
            self.time_windows
                .iter()
                .filter(move |((_, r), _)| *r == resource)
                .flat_map(|(_, windows)| windows.iter().flat_map(|w| [w.lower, w.upper]))
        }
        #[cfg(not(feature = "multi-time-windows"))]
        {
            let _ = resource;
            std::iter::empty()
        }
    }

    pub(super) fn validate_time_windows(&self) -> Result<()> {
        #[cfg(feature = "multi-time-windows")]
        for (&(vertex, resource), windows) in &self.time_windows {
            if vertex >= self.vertices.len()
                || self.resources.get(resource) != Some(&ResourceKind::Disposable)
                || windows.is_empty()
            {
                return Err(invalid("invalid multi-time-window resource or vertex"));
            }
            for w in windows {
                Window::new(w.lower, w.upper)?;
            }
        }
        Ok(())
    }

    /// Earliest allowed service start >= q (forward), or latest <= q (backward).
    /// `bounds` must already intersect the vertex and optional arc domains.
    #[inline]
    pub(crate) fn project_resource(
        &self,
        vertex: usize,
        resource: usize,
        bounds: Window,
        q: f64,
        forward: bool,
    ) -> Option<f64> {
        if !q.is_finite() || bounds.lower > bounds.upper {
            return None;
        }
        #[cfg(feature = "multi-time-windows")]
        if let Some(windows) = self.time_windows.get(&(vertex, resource)) {
            if self.resources[resource] != ResourceKind::Disposable {
                return None;
            }
            if forward {
                for w in windows {
                    let value = q.max(bounds.lower).max(w.lower);
                    if value <= bounds.upper.min(w.upper) {
                        return Some(value);
                    }
                }
            } else {
                for w in windows.iter().rev() {
                    let value = q.min(bounds.upper).min(w.upper);
                    if value >= bounds.lower.max(w.lower) {
                        return Some(value);
                    }
                }
            }
            return None;
        }
        let _ = vertex;
        let equality = self.resources[resource] == ResourceKind::Equality;
        let value = if forward && !equality {
            q.max(bounds.lower)
        } else if !forward && !equality {
            q.min(bounds.upper)
        } else {
            q
        };
        (bounds.lower <= value && value <= bounds.upper).then_some(value)
    }

    /// A fixed initial source value must itself be admissible. Without an
    /// explicit initial value, select the first/last point of the terminal domain.
    pub(crate) fn root_resource(&self, resource: usize, forward: bool) -> Option<f64> {
        let vertex = if forward { self.source } else { self.sink };
        let bounds = self.vertices[vertex].bounds[resource];
        let initial = forward.then(|| self.initial[resource]).flatten();
        let q = initial.unwrap_or(if forward { bounds.lower } else { bounds.upper });
        let projected = self.project_resource(vertex, resource, bounds, q, forward)?;
        if initial.is_some() && projected != q {
            return None;
        }
        Some(projected)
    }
}
