//! Vehicle-routing input as a one-network set-partitioning model.
use super::*;

impl<const D: usize> Problem<D> {
    /// Cover each vertex other than source and sink exactly once with routes of
    /// one pricing network. The fleet size is the network's multiplicity upper
    /// bound, so it is a quantity row of the master, not a shared row.
    /// Shared row `r` covers `customers()[r]`; each arc charges its head vertex,
    /// so a walk that revisits a vertex charges that row once per visit.
    pub fn set_partitioning(graph: Graph<D>, vehicles: f64) -> Result<Self> {
        if !vehicles.is_finite() || vehicles <= 0. {
            return Err(invalid("positive finite vehicle multiplicity required"));
        }
        let mut row_of = vec![None; graph.vertices.len()];
        let mut rows = vec![];
        for v in 0..graph.vertices.len() {
            if v != graph.source && v != graph.sink {
                row_of[v] = Some(rows.len());
                rows.push(Row {
                    sense: RowSense::Equal,
                    rhs: 1.,
                });
            }
        }
        let arc_entries = graph
            .arcs
            .iter()
            .map(|a| row_of[a.head].map_or_else(Vec::new, |r| vec![(r, 1.)]))
            .collect();
        Ok(Self {
            rows,
            networks: vec![Network {
                id: 0,
                graph,
                multiplicity: Window {
                    lower: 0.,
                    upper: vehicles,
                },
                arc_entries,
                route_entries: vec![],
                activity_entries: vec![],
            }],
        })
    }

    /// Vertices covered by the shared rows of [`Problem::set_partitioning`],
    /// in row order.
    pub fn customers(&self) -> Vec<usize> {
        self.networks.first().map_or_else(Vec::new, |network| {
            let g = &network.graph;
            (0..g.vertices.len())
                .filter(|&v| v != g.source && v != g.sink)
                .collect()
        })
    }

    /// One integer branching variable per arc of every network: the number of
    /// times selected routes traverse it. Parallel arcs stay separate variables.
    pub fn arc_flow_variables(&self, priority: f64) -> Vec<MappedBranchVariable> {
        self.networks
            .iter()
            .flat_map(|network| {
                network
                    .graph
                    .arcs
                    .iter()
                    .map(move |arc| MappedBranchVariable {
                        name: format!("x_{}_{}", network.id, arc.id),
                        priority,
                        networks: vec![MappedNetworkVariable {
                            network: network.id,
                            arcs: vec![(arc.id, 1.)],
                            activities: vec![],
                            per_route: 0.,
                        }],
                        masters: vec![],
                    })
            })
            .collect()
    }

    /// One integer branching variable per undirected edge of every network:
    /// the flow on both arc directions, with the sink identified with the
    /// source (the depot). Edges are named by their smaller vertex first.
    pub fn edge_flow_variables(&self, priority: f64) -> Vec<MappedBranchVariable> {
        let mut variables = vec![];
        for network in &self.networks {
            let g = &network.graph;
            let depot = |v: usize| if v == g.sink { g.source } else { v };
            let mut edges: BTreeMap<(usize, usize), Vec<(i32, f64)>> = BTreeMap::new();
            for arc in &g.arcs {
                let (u, v) = (depot(arc.tail), depot(arc.head));
                if u != v {
                    edges
                        .entry((u.min(v), u.max(v)))
                        .or_default()
                        .push((arc.id, 1.));
                }
            }
            variables.extend(
                edges
                    .into_iter()
                    .map(|((u, v), arcs)| MappedBranchVariable {
                        name: format!("e_{}_{u}_{v}", network.id),
                        priority,
                        networks: vec![MappedNetworkVariable {
                            network: network.id,
                            arcs,
                            activities: vec![],
                            per_route: 0.,
                        }],
                        masters: vec![],
                    }),
            );
        }
        variables
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/set_partitioning_tests.rs"]
mod tests;
