//! Borrowed physical arc-ID lookup. A view cannot outlive a graph mutation.
use super::{Arc, Graph};
use std::collections::HashMap;

pub(crate) struct ArcLookup<'a, const D: usize> {
    graph: &'a Graph<D>,
    positions: HashMap<i32, usize>,
}
impl<const D: usize> Graph<D> {
    /// Dense IDs need no index; sparse IDs preserve first-match semantics.
    pub(crate) fn sparse_arc_lookup(&self) -> Option<ArcLookup<'_, D>> {
        if self
            .arcs
            .iter()
            .enumerate()
            .all(|(i, a)| usize::try_from(a.id) == Ok(i))
        {
            return None;
        }
        let mut positions = HashMap::with_capacity(self.arcs.len());
        for (position, arc) in self.arcs.iter().enumerate() {
            positions.entry(arc.id).or_insert(position);
        }
        Some(ArcLookup {
            graph: self,
            positions,
        })
    }
    pub(crate) fn arc_by_id(&self, id: i32) -> Option<(usize, &Arc<D>)> {
        usize::try_from(id)
            .ok()
            .and_then(|i| self.arcs.get(i).filter(|a| a.id == id).map(|a| (i, a)))
            .or_else(|| self.arcs.iter().enumerate().find(|(_, a)| a.id == id))
    }
}
impl<const D: usize> ArcLookup<'_, D> {
    pub(crate) fn get(&self, id: i32) -> Option<(usize, &Arc<D>)> {
        let position = *self.positions.get(&id)?;
        Some((position, &self.graph.arcs[position]))
    }
}
