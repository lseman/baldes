use super::*;

impl<'p> Population<'p> {
    /// Update MDM elite set.
    pub fn update_mdm_elite(&mut self, indiv: &Individual<'p>) {
        const MDM_NB_ELITE: usize = 5;
        self.mdm_elite.push(Box::new(indiv.clone()));
        self.mdm_elite.sort_by(|a, b| {
            a.my_cost_sol
                .penalized_cost
                .partial_cmp(&b.my_cost_sol.penalized_cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if self.mdm_elite.len() > MDM_NB_ELITE {
            self.mdm_elite.truncate(MDM_NB_ELITE);
        }
        self.mdm_elite_non_updating_restarts = 0;
        self.mdm_elite_updated = true;
    }

    /// Mine route fragments shared by the elite set (HGS-MDM mineElite):
    /// the maximal sets of arcs present in at least 80% of the elite
    /// solutions, chained into paths. HGS uses FPMax; with at most five
    /// transactions, intersecting every large enough subset is exact.
    pub fn mine_elite(&mut self) {
        const MDM_NB_PATTERNS: usize = 5;
        const MDM_MIN_SUP: f64 = 0.8;
        if !self.mdm_elite_updated
            || self.mdm_elite_non_updating_restarts < self.mdm_elite_max_non_updating_restarts
            || self.mdm_elite.len() <= 1
        {
            return;
        }
        if self.params.verbose {
            crate::log_err!("  HGS restart | Mining elite patterns");
        }
        let nb_nodes = self.params.nb_clients as usize + 1;
        let transactions: Vec<std::collections::BTreeSet<usize>> = self
            .mdm_elite
            .iter()
            .map(|indiv| {
                indiv
                    .chrom_r
                    .iter()
                    .flat_map(|route| route.windows(2).map(|arc| arc[0] * nb_nodes + arc[1]))
                    .collect()
            })
            .collect();
        let min_support = 2.max((MDM_MIN_SUP * transactions.len() as f64) as usize);

        let mut itemsets: Vec<Vec<usize>> = Vec::new();
        for subset in 1u32..(1 << transactions.len()) {
            if (subset.count_ones() as usize) < min_support {
                continue;
            }
            let mut members = (0..transactions.len()).filter(|&t| subset & (1 << t) != 0);
            let mut common = transactions[members.next().unwrap()].clone();
            for t in members {
                common.retain(|arc| transactions[t].contains(arc));
            }
            if !common.is_empty() {
                itemsets.push(common.into_iter().collect());
            }
        }
        itemsets.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        itemsets.dedup();
        let is_subset = |small: &[usize], large: &[usize]| {
            small.iter().all(|arc| large.binary_search(arc).is_ok())
        };
        let maximal: Vec<&Vec<usize>> = itemsets
            .iter()
            .enumerate()
            .filter(|&(i, set)| {
                !itemsets[..i]
                    .iter()
                    .any(|larger| larger.len() > set.len() && is_subset(set, larger))
            })
            .map(|(_, set)| set)
            .take(MDM_NB_PATTERNS)
            .collect();

        self.mdm_patterns = maximal
            .into_iter()
            .map(|itemset| {
                // Chain the arcs into paths, as HGS-MDM does.
                let mut routes: Vec<Vec<usize>> = Vec::new();
                for &index in itemset {
                    let (n1, n2) = (index / nb_nodes, index % nb_nodes);
                    let lhs = routes
                        .iter()
                        .rposition(|r| n1 != 0 && r.last() == Some(&n1));
                    let rhs = routes
                        .iter()
                        .rposition(|r| n2 != 0 && r.first() == Some(&n2));
                    match (lhs, rhs) {
                        (Some(l), Some(r)) if l != r => {
                            let tail = routes.remove(r);
                            let l = if r < l { l - 1 } else { l };
                            routes[l].extend(tail);
                        }
                        (Some(_), Some(_)) => {}
                        (Some(l), None) => routes[l].push(n2),
                        (None, Some(r)) => routes[r].insert(0, n1),
                        (None, None) => routes.push(vec![n1, n2]),
                    }
                }
                routes
            })
            .collect();
        self.mdm_elite_updated = false;
        self.mdm_next_pattern = 0;
    }

    /// The next mined pattern, cycling through them.
    pub fn next_mdm_pattern(&mut self) -> Option<Vec<Vec<usize>>> {
        if self.mdm_patterns.is_empty() {
            return None;
        }
        let pattern = self.mdm_patterns[self.mdm_next_pattern].clone();
        self.mdm_next_pattern = (self.mdm_next_pattern + 1) % self.mdm_patterns.len();
        Some(pattern)
    }
}
