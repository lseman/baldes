use super::*;

impl<'p> Population<'p> {
    /// Extract feasible routes from the population.
    pub fn extract_feasible_routes(&self) -> Vec<Vec<usize>> {
        let mut routes: Vec<Vec<usize>> = Vec::new();
        for indiv in &self.feasible {
            if !indiv.is_feasible {
                continue;
            }
            for k in 0..indiv.nb_vehicles {
                if indiv.chrom_r[k].is_empty() {
                    break;
                }
                let mut route: Vec<usize> = vec![0];
                route.extend(&indiv.chrom_r[k]);
                route.push(0);
                routes.push(route);
            }
        }
        routes
    }

    /// Extract feasible routes from the best overall solution.
    pub fn extract_best_feasible_routes(&self) -> Vec<Vec<usize>> {
        let mut routes: Vec<Vec<usize>> = Vec::new();
        let target = if self.best_overall.is_feasible {
            &self.best_overall
        } else if let Some(best) = self.get_best_feasible() {
            best
        } else {
            return routes;
        };

        for k in 0..target.nb_vehicles {
            if target.chrom_r[k].is_empty() {
                break;
            }
            let mut route: Vec<usize> = vec![0];
            route.extend(&target.chrom_r[k]);
            route.push(0);
            routes.push(route);
        }
        routes
    }

    /// Extract top N best feasible solutions.
    pub fn extract_top_best_feasible_routes(&self, n: usize) -> Vec<Vec<usize>> {
        let size = (self.params.config.minimum_population_size as usize).min(self.feasible.len());
        let mut routes: Vec<Vec<usize>> = Vec::new();

        for i in 0..size.min(n) {
            if self.feasible[i].is_feasible {
                for k in 0..self.feasible[i].nb_vehicles {
                    if self.feasible[i].chrom_r[k].is_empty() {
                        break;
                    }
                    let mut route: Vec<usize> = vec![0];
                    route.extend(&self.feasible[i].chrom_r[k]);
                    route.push(0);
                    routes.push(route);
                }
            }
        }
        routes
    }
}
