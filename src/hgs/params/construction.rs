use super::*;

impl Params {
    /// Create Params from a BALDES Instance. Coordinates, times and explicit
    /// distances retain full precision in internally scaled units (×10).
    pub fn from_instance(instance: &Instance) -> Self {
        let scale = |value: f64| value * 10.0;
        let mut has_time_windows = false;
        let cli = instance
            .customers
            .iter()
            .enumerate()
            .map(|(i, c)| {
                has_time_windows |= c.due.is_finite();
                Client {
                    cust_num: i as i32,
                    coord_x: c.x * 10.0,
                    coord_y: c.y * 10.0,
                    service_duration: scale(c.service),
                    demand: c.demand as i32,
                    earliest_arrival: scale(c.ready),
                    latest_arrival: if c.due.is_finite() {
                        scale(c.due).min(HORIZON)
                    } else {
                        HORIZON
                    },
                    ..Client::default()
                }
            })
            .collect();
        // Use exactly the selected instance metric; tenths are opt-in.
        let n = instance.customers.len();
        let distances = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| instance.distance(i, j, instance.rounding) * 10.0)
                    .collect()
            })
            .collect::<Vec<Vec<f64>>>();
        let problem_type = if has_time_windows {
            ProblemType::Vrptw
        } else {
            ProblemType::Cvrp
        };
        let mut params = Self::build(
            cli,
            instance.vehicles as i32,
            instance.capacity as i32,
            problem_type,
            Some(&distances),
        );
        params.instance_name = instance.name.clone();
        params
    }

    /// Create Params from raw instance data. Coordinates are scaled by 10;
    /// times are taken as given.
    pub fn from_instance_data(data: &InstanceData) -> Self {
        let cli = (0..data.x_coord.len())
            .map(|i| Client {
                cust_num: i as i32,
                coord_x: data.x_coord[i] * 10.0,
                coord_y: data.y_coord[i] * 10.0,
                service_duration: f64::from(data.service_time[i]),
                demand: data.demand[i],
                earliest_arrival: f64::from(data.window_open[i]),
                latest_arrival: f64::from(data.window_close[i]).min(HORIZON),
                time_windows: if data.n_tw[i] > 0 {
                    data.time_windows[i]
                        .iter()
                        .map(|&(e, l)| TimeWindow::new(e, l))
                        .collect()
                } else {
                    Vec::new()
                },
                n_tw: data.n_tw[i],
                ..Client::default()
            })
            .collect();
        Self::build(
            cli,
            data.n_v,
            data.q,
            data.problem_type,
            data.distances.as_ref(),
        )
    }

    /// Shared part of the constructors (HGS Params::Params). `nb_vehicles` 0
    /// means the default fleet ceil(1.3 * demand / capacity) + 3.
    pub(super) fn build(
        mut cli: Vec<Client>,
        nb_vehicles: i32,
        vehicle_capacity: i32,
        problem_type: ProblemType,
        distances: Option<&Vec<Vec<f64>>>,
    ) -> Self {
        let config = Config::default();
        let n = cli.len();
        let nb_clients = n as i32 - 1;

        let (depot_x, depot_y) = (cli[0].coord_x, cli[0].coord_y);
        let mut total_demand = 0;
        let mut max_demand = 0;
        for client in &mut cli {
            let angle = 32768. * (client.coord_y - depot_y).atan2(client.coord_x - depot_x)
                / std::f64::consts::PI;
            client.polar_angle = CircleSector::positive_mod(angle as i32);
            max_demand = max_demand.max(client.demand);
            total_demand += client.demand;
        }
        let nb_vehicles = if nb_vehicles <= 0 {
            ((1.3 * total_demand as f64 / vehicle_capacity as f64).ceil() + 3.) as i32
        } else {
            nb_vehicles
        };

        let mut time_cost = Matrix::new(n);
        let mut max_dist = 0_f64;
        for i in 0..n {
            for j in 0..n {
                let cost = if i == j {
                    0.
                } else if let Some(distances) = distances {
                    distances[i][j]
                } else {
                    let (dx, dy) = (
                        cli[i].coord_x - cli[j].coord_x,
                        cli[i].coord_y - cli[j].coord_y,
                    );
                    (dx * dx + dy * dy).sqrt()
                };
                time_cost.set(i, j, cost);
                max_dist = max_dist.max(cost);
            }
        }

        // Neighbors by distance plus the cheaper of the two waiting/time-warp
        // orientations (Vidal et al. 2013).
        let (weight_wait, weight_warp) = (0.2, 1.0);
        let mut order_proximities = vec![Vec::new(); n];
        for i in 1..n {
            for j in (1..n).filter(|&j| j != i) {
                let t = time_cost.get(i, j);
                let (a, b) = (&cli[i], &cli[j]);
                let orient = |a: &Client, b: &Client| {
                    let wait =
                        (b.earliest_arrival - t - a.service_duration - a.latest_arrival).max(0.);
                    let warp =
                        (a.earliest_arrival + a.service_duration + t - b.latest_arrival).max(0.);
                    weight_wait * wait as f64 + weight_warp * warp as f64
                };
                order_proximities[i].push((t as f64 + orient(a, b).min(orient(b, a)), j as i32));
            }
            order_proximities[i].sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        }

        let mut savings_list = Vec::with_capacity(n * n.saturating_sub(1) / 2);
        for i in 1..n {
            for j in 1..i {
                let value =
                    (time_cost.get(0, i) + time_cost.get(0, j) - time_cost.get(i, j)) as f64;
                savings_list.push(Savings {
                    c1: i as i32,
                    c2: j as i32,
                    value,
                });
            }
        }
        savings_list.sort_by(|a, b| b.value.total_cmp(&a.value));

        let penalty_capacity = (max_dist as f64 / max_demand.max(1) as f64).clamp(0.1, 1000.);
        let penalty_time_warp = match problem_type {
            ProblemType::Vrptw | ProblemType::Evrp => config.initial_time_warp_penalty,
            ProblemType::Cvrp => 0.,
        };
        let degrees_to_ring = |degrees: i32| (degrees as f64 / 360. * 65536.) as u16;
        let mut params = Params {
            rng: std::cell::RefCell::new(Xoroshiro128Plus::new(config.seed)),
            start_time: Instant::now(),
            penalty_capacity: Cell::new(penalty_capacity),
            penalty_wait_time: 0.0,
            penalty_time_warp: Cell::new(penalty_time_warp),
            proximity_weight_wait_time: weight_wait,
            proximity_weight_time_warp: weight_warp,
            problem_type,
            instance_name: String::new(),
            is_duration_constrained: false,
            is_time_window_constrained: problem_type != ProblemType::Cvrp,
            is_explicit_distance_matrix: distances.is_some(),
            nb_clients,
            nb_vehicles,
            duration_limit: f64::MAX,
            vehicle_capacity,
            total_demand,
            max_demand,
            max_dist,
            cli,
            time_cost,
            order_proximities,
            correlated_vertices: Vec::new(),
            circle_sector_overlap_tolerance: degrees_to_ring(
                config.circle_sector_overlap_tolerance_degrees,
            ),
            min_circle_sector_size: degrees_to_ring(config.min_circle_sector_size_degrees),
            savings_list,
            duals_ptr: None,
            forbidden_edges_set: None,
            verbose: true,
            config,
        };
        params.correlated_vertices = params.correlated_vertices_for(params.config.nb_granular);
        params
    }

    /// The `nb_granular` nearest neighbors of each client by proximity, in
    /// increasing index order (HGS SetCorrelatedVertices).
    pub fn correlated_vertices_for(&self, nb_granular: i32) -> Vec<Vec<i32>> {
        let n = self.nb_clients as usize;
        let count = (nb_granular.max(0) as usize).min(n.saturating_sub(1));
        let mut sets = vec![std::collections::BTreeSet::new(); n + 1];
        for i in 1..=n {
            for &(_, j) in &self.order_proximities[i][..count] {
                sets[i].insert(j);
                if self.config.use_symmetric_correlated_vertices {
                    sets[j as usize].insert(i as i32);
                }
            }
        }
        sets.into_iter()
            .map(|set| set.into_iter().collect())
            .collect()
    }
}
