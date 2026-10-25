//! Exact-integer simultaneous-load reversal and a dual-orientation prefix universe.
use super::*;

impl CoupledKernel {
    pub(crate) fn reversal_eligible(&self, reverse: &[usize]) -> bool {
        if reverse.len() != self.arc_count
            || reverse
                .iter()
                .enumerate()
                .any(|(i, &r)| reverse.get(r) != Some(&i))
        {
            return false;
        }
        self.definitions.iter().all(|resource| match resource {
            // Directional vertex masks and forgetting require their own
            // reversal transform, beyond symmetric arc intersections.
            CoupledResource::Selection {
                allowed,
                vertices,
                relaxed,
                ..
            } => {
                !relaxed
                    && vertices.is_empty()
                    && reverse
                        .iter()
                        .enumerate()
                        .all(|(i, &r)| allowed[i] == allowed[r])
            }
            CoupledResource::SimultaneousPickupDelivery {
                capacity,
                deliveries,
                pickups,
                ..
            } => {
                // All retained D/N/M sums and a two-prefix join are exact
                // integers (at most 2^53). No floating reassociation is assumed.
                *capacity <= 4_503_599_627_370_496.
                    && deliveries
                        .iter()
                        .chain(pickups)
                        .all(|x| x.fract() == 0. && *x <= *capacity)
                    && reverse
                        .iter()
                        .enumerate()
                        .all(|(i, &r)| deliveries[i] == deliveries[r] && pickups[i] == pickups[r])
            }
            // Numeric entry constraints and state-dependent costs require a
            // different reversal transform; zero objective alone is not proof.
            _ => false,
        })
    }

    /// Mixed signs need both prefix and suffix peaks. This is prepared once
    /// per sweep/root, never by extending the public standalone replay universe.
    pub(crate) fn reversal_union_required(&self) -> bool {
        self.reversal_union
    }
    pub(crate) fn reversal_root(&self, forward: bool) -> Box<CoupledLabelState> {
        let mut root = self.root(forward);
        root.reversed_use = true;
        if forward && self.reversal_union_required() {
            root.reversal_peaks = Some(Box::new(StateValues::filled(
                self.load_capacities.len(),
                0.,
            )));
        }
        root
    }
    pub(crate) fn extend_reversal_with_workspace(
        &self,
        state: &CoupledLabelState,
        arc: usize,
        forward: bool,
        workspace: &mut CoupledWorkspace,
    ) -> Result<Option<(Box<CoupledLabelState>, f64)>> {
        self.extend_internal(state, arc, forward, true, true, workspace)
    }

    /// The caller establishes reversal eligibility for the physical reverse
    /// map before a sweep. Selection intersection is commutative. Mixed-sign
    /// exact-integer SPD swaps prefix and suffix peaks on reversal. The final
    /// capacity check always uses the actual joined forward orientation.
    pub(crate) fn join_reversed(
        &self,
        f: &CoupledLabelState,
        reversed_prefix: &CoupledLabelState,
    ) -> Result<Option<f64>> {
        if reversed_prefix.suffix.is_some() {
            // Completion APIs may supply an actual backward query even when
            // the retained opposite universe reuses forward prefixes.
            return self.join(f, reversed_prefix);
        }
        self.check_state(f)?;
        self.check_state(reversed_prefix)?;
        if f.suffix.is_some() || !self.initial.is_empty() || !self.budgeted.is_empty() {
            return Err(invalid("unsupported coupled prefix reversal"));
        }
        if self.reversal_union
            && reversed_prefix.reversal_peaks.is_none()
            && reversed_prefix.loads.iter().any(|load| *load != [0.; 3])
        {
            return Err(invalid(
                "mixed-sign reversal requires dual-orientation state",
            ));
        }
        if f.selection
            .iter()
            .zip(&reversed_prefix.selection)
            .any(|(a, b)| a & b == 0)
        {
            return Ok(None);
        }
        for (index, ((&a, &b), &capacity)) in f
            .loads
            .iter()
            .zip(&reversed_prefix.loads)
            .zip(&self.load_capacities)
            .enumerate()
        {
            let mut b = b;
            if let Some(peaks) = &reversed_prefix.reversal_peaks {
                b[2] = peaks[index];
            }
            let load = combine_load(a, b)?;
            if finite(load[0] + load[2])? > capacity {
                return Ok(None);
            }
        }
        Ok(Some(0.))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/graph/coupled/kernel/reversal/tests.rs"]
mod tests;
