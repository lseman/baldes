use super::*;

impl<'p> Individual<'p> {
    // ── Proximity / diversity ─────────────────────────────────────────────

    /// Add a reference to another individual in the proximity structure.
    pub fn add_proximity(&mut self, distance: f64, indiv_id: usize) {
        self.indivs_per_proximity
            .insert((SortableF64(distance), indiv_id));
    }

    /// Remove an individual from the proximity structure by ID.
    pub fn remove_proximity(&mut self, indiv_id: usize) {
        self.indivs_per_proximity.retain(|&(_, id)| id != indiv_id);
    }

    /// Broken pairs distance between this individual and indiv2.
    /// Measures the fraction of arcs that differ (direct or reverse).
    pub fn broken_pairs_distance(&self, indiv2: &Self) -> f64 {
        let nb_clients = self.params.nb_clients as usize;
        let mut differences = 0usize;

        for j in 1..=nb_clients {
            // Check if successor differs.
            if self.successors[j] != indiv2.successors[j]
                && self.successors[j] != indiv2.predecessors[j]
            {
                differences += 1;
            }
            // Check arc entering j from predecessor vs indiv2's arcs.
            if self.predecessors[j] == 0 && indiv2.predecessors[j] != 0 && indiv2.successors[j] != 0
            {
                differences += 1;
            }
        }

        differences as f64 / nb_clients as f64
    }

    /// Average broken pairs distance to the `nb_closest` individuals.
    pub fn average_broken_pairs_distance_closest(&self, nb_closest: usize) -> f64 {
        let size = self.indivs_per_proximity.len().min(nb_closest);
        if size == 0 {
            return 0.0;
        }

        let sum: f64 = self
            .indivs_per_proximity
            .iter()
            .take(size)
            .map(|(d, _)| d.0)
            .sum();

        sum / size as f64
    }
}
