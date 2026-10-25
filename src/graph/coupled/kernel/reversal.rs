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
            CoupledResource::Selection { allowed, .. } => reverse
                .iter()
                .enumerate()
                .all(|(i, &r)| allowed[i] == allowed[r]),
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
mod tests {
    use super::*;
    fn graph(pickup: f64, delivery: f64) -> Graph<2> {
        let mut graph = Graph::new(4, 1).unwrap();
        for (u, v) in [(0, 1), (1, 2), (2, 1), (1, 3)] {
            graph.add_arc(u, v, 1.).unwrap();
        }
        graph
            .set_coupled_resources(&[
                CoupledResource::Selection {
                    initial: 3,
                    allowed: vec![3; 4],
                },
                CoupledResource::SimultaneousPickupDelivery {
                    capacity: 20.,
                    deliveries: vec![delivery; 4],
                    pickups: vec![pickup; 4],
                    tolerance: 0.,
                },
            ])
            .unwrap();
        graph
    }
    #[test]
    fn reversed_prefix_matches_real_backward_replay_at_every_internal_split() {
        let reverse = [3, 2, 1, 0];
        for (p, d) in [(2., 1.), (1., 2.), (1., 1.), (0., 1.), (1., 0.)] {
            let graph = graph(p, d);
            let kernel = graph.coupled_kernel().unwrap();
            assert!(kernel.reversal_eligible(&reverse));
            for split in 1..4 {
                let f = kernel
                    .replay_state(&(0..split).collect::<Vec<_>>())
                    .unwrap()
                    .unwrap()
                    .0;
                let prefix: Vec<_> = (split..4).rev().map(|a| reverse[a]).collect();
                let reversed = kernel.replay_state(&prefix).unwrap().unwrap().0;
                let mut b = kernel.root(false);
                for a in (split..4).rev() {
                    b = kernel.extend(&b, a, false).unwrap().unwrap().0;
                }
                assert_eq!(reversed.loads, b.loads);
                assert_eq!(
                    kernel.join_reversed(&f, &reversed).unwrap(),
                    kernel.join(&f, &b).unwrap()
                );
                let mut hull = CoupledStateBounds::default();
                hull.include_reversed(Some(&reversed));
                assert!(kernel.completion_feasible(true, Some(&f), &hull));
            }
        }
    }
    #[test]
    fn mixed_sign_union_retains_reverse_only_prefix_and_joins_strictly() {
        let mut graph = graph(0., 0.);
        graph
            .set_coupled_resources(&[CoupledResource::SimultaneousPickupDelivery {
                capacity: 5.,
                deliveries: vec![0., 4., 4., 0.],
                pickups: vec![4., 0., 0., 4.],
                tolerance: 0.,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        assert!(kernel.reversal_eligible(&[3, 2, 1, 0]));
        assert!(kernel.replay_state(&[0, 1]).unwrap().is_none());
        let mut workspace = CoupledWorkspace::default();
        let root = kernel.reversal_root(true);
        let first = kernel
            .extend_reversal_with_workspace(&root, 0, true, &mut workspace)
            .unwrap()
            .unwrap()
            .0;
        let reverse_only = kernel
            .extend_reversal_with_workspace(&first, 1, true, &mut workspace)
            .unwrap()
            .unwrap()
            .0;
        assert_eq!(reverse_only.load_summaries(), &[[4., 0., 4.]]);
        assert_eq!(reverse_only.reversal_peaks.as_ref().unwrap()[0], 0.);
        // As suffix its reversed orientation is feasible; as forward route it is not.
        assert_eq!(
            kernel.join_reversed(&root, &reverse_only).unwrap(),
            Some(0.)
        );
        assert_eq!(
            kernel.join(&reverse_only, &kernel.root(false)).unwrap(),
            None
        );
        assert_eq!(kernel.join_reversed(&reverse_only, &root).unwrap(), None);
        let opposite = kernel
            .extend_reversal_with_workspace(&root, 1, true, &mut workspace)
            .unwrap()
            .unwrap()
            .0;
        let opposite = kernel
            .extend_reversal_with_workspace(&opposite, 0, true, &mut workspace)
            .unwrap()
            .unwrap()
            .0;
        assert!(!loads_dominate_for_test(&reverse_only, &opposite));
        assert!(!loads_dominate_for_test(&opposite, &reverse_only));
    }
    fn loads_dominate_for_test(a: &CoupledLabelState, b: &CoupledLabelState) -> bool {
        super::super::dominance::loads_dominate(a, b, &[])
    }
    #[test]
    fn mixed_sign_union_matches_independent_integer_walk_oracle() {
        let mut graph = graph(0., 0.);
        graph
            .set_coupled_resources(&[CoupledResource::SimultaneousPickupDelivery {
                capacity: 5.,
                deliveries: vec![0., 2., 2., 0.],
                pickups: vec![3., 0., 0., 3.],
                tolerance: 0.,
            }])
            .unwrap();
        let kernel = graph.coupled_kernel().unwrap();
        fn feasible(path: &[usize]) -> bool {
            let mut deliveries = 0i32;
            let mut net = 0i32;
            let mut peak = 0i32;
            for &arc in path {
                let (d, p) = if arc == 0 { (0, 3) } else { (2, 0) };
                deliveries += d;
                net += p - d;
                peak = peak.max(net);
            }
            deliveries + peak <= 5
        }
        for len in 0..=7 {
            for bits in 0..(1usize << len) {
                let path: Vec<_> = (0..len).map(|i| (bits >> i) & 1).collect();
                let reversed: Vec<_> = path.iter().rev().copied().collect();
                let expected = feasible(&path) || feasible(&reversed);
                let mut state = Some(kernel.reversal_root(true));
                let mut workspace = CoupledWorkspace::default();
                for &arc in &path {
                    state = match state {
                        Some(ref old) => kernel
                            .extend_reversal_with_workspace(old, arc, true, &mut workspace)
                            .unwrap()
                            .map(|x| x.0),
                        None => None,
                    };
                }
                assert_eq!(state.is_some(), expected, "{path:?}");
                if let Some(state) = state {
                    assert_eq!(
                        kernel
                            .join_reversed(&kernel.reversal_root(true), &state)
                            .unwrap()
                            .is_some(),
                        feasible(&reversed),
                        "{path:?}"
                    );
                    assert_eq!(
                        kernel.join(&state, &kernel.root(false)).unwrap().is_some(),
                        feasible(&path),
                        "{path:?}"
                    );
                }
            }
        }
    }
    #[test]
    fn reversal_declines_unproved_floating_numeric_and_asymmetric_cases() {
        let reverse = [3, 2, 1, 0];
        for mode in 0..5 {
            let mut graph = graph(2., 1.);
            let mut definitions = graph.coupled_kernel().unwrap().definitions().to_vec();
            match mode {
                0 => {
                    if let CoupledResource::Selection { allowed, .. } = &mut definitions[0] {
                        allowed[0] = 1;
                    }
                }
                1 => {
                    if let CoupledResource::SimultaneousPickupDelivery { pickups, .. } =
                        &mut definitions[1]
                    {
                        pickups[1] = 0.;
                    }
                }
                2 => {
                    if let CoupledResource::SimultaneousPickupDelivery { deliveries, .. } =
                        &mut definitions[1]
                    {
                        deliveries.fill(0.5);
                    }
                }
                3 => {
                    if let CoupledResource::SimultaneousPickupDelivery { capacity, .. } =
                        &mut definitions[1]
                    {
                        *capacity = 9_007_199_254_740_992.;
                    }
                }
                _ => definitions.push(CoupledResource::CumulativeCost {
                    initial: 0.,
                    increments: vec![0.; 4],
                    weights: vec![0.; 4],
                }),
            }
            graph.set_coupled_resources(&definitions).unwrap();
            assert!(
                !graph.coupled_kernel().unwrap().reversal_eligible(&reverse),
                "mode {mode}"
            );
        }
        assert!(!graph(2., 1.)
            .coupled_kernel()
            .unwrap()
            .reversal_eligible(&[0; 4]));
    }
}
