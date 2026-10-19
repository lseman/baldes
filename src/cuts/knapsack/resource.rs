use super::*;

/// An original integer master row and its graph-specific resource/scaling.
#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackRow {
    pub rhs: i32,
    pub graph_resources: BTreeMap<i32, (usize, f64)>,
    pub load_totals: BTreeMap<i32, f64>,
}
impl KnapsackRow {
    pub fn new(graph: i32, resource: usize, scale: f64, rhs: i32) -> Self {
        Self {
            rhs,
            graph_resources: BTreeMap::from([(graph, (resource, scale))]),
            load_totals: BTreeMap::new(),
        }
    }
}
/// Original graph resource metadata used by `prepareSeparation`.
/// IDs are independent of the records' positions and solver coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnapsackResource {
    pub id: usize,
    pub variable_ids: Vec<i32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnapsackGraph {
    pub id: i32,
    pub resources: Vec<KnapsackResource>,
}
/// Preparation associations use graph positions in `graphs`; `None` models
/// a null original graph pointer. Ordered positions preserve native map order.
#[derive(Clone, Debug, PartialEq)]
pub struct KnapsackPreparationRow {
    pub rhs: i32,
    pub load_totals: BTreeMap<i32, f64>,
    pub graph_resources: BTreeMap<Option<usize>, (usize, f64)>,
}
#[derive(Clone, Debug)]
pub struct KnapsackColumn {
    pub graph_id: i32,
    pub resources: Vec<f64>,
    pub value: f64,
}
/// Native RouteLoadKnapsackCut, including its original row association.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourceKnapsackCut {
    pub id: i32,
    pub row: usize,
    pub cut_class: i32,
    pub graph_resources: BTreeMap<i32, (usize, f64)>,
    pub cut: RouteLoadKnapsackCut,
    /// Master dual; pricing uses its negative, rounded to 1e-8.
    pub dual: f64,
}
impl ResourceKnapsackCut {
    pub fn coefficient(&self, graph: i32, resources: &[f64]) -> i32 {
        let Some(&(resource, scale)) = self.graph_resources.get(&graph) else {
            return 0;
        };
        resources.get(resource).map_or(0, |q| {
            self.cut.coefficient((q * scale + 1e-6).floor() as i32)
        })
    }
    /// Evaluate an original-ID cut on a solver-coordinate resource vector.
    pub fn coefficient_mapped(
        &self,
        graph: i32,
        resources: &[f64],
        indices: &BTreeMap<usize, usize>,
    ) -> i32 {
        let Some(&(original, scale)) = self.graph_resources.get(&graph) else {
            return 0;
        };
        indices
            .get(&original)
            .and_then(|&slot| resources.get(slot))
            .map_or(0, |q| {
                self.cut.coefficient((q * scale + 1e-6).floor() as i32)
            })
    }
}
#[derive(Clone, Debug, Default)]
pub struct ResourceKnapsackSeparator {
    pub parameters: KnapsackParameters,
    pub rows: Vec<KnapsackRow>,
    pub next_cut_id: i32,
    pub num_separate_calls: usize,
}
impl ResourceKnapsackSeparator {
    /// failures retain completed rows and the partially prepared current row.
    /// Cut identifiers and separation-call counters survive preparation.
    pub fn prepare(
        &mut self,
        graphs: &[KnapsackGraph],
        rows: &[KnapsackPreparationRow],
    ) -> Result<()> {
        self.rows.clear();
        for row in rows {
            self.rows.push(KnapsackRow {
                rhs: row.rhs,
                load_totals: row.load_totals.clone(),
                graph_resources: BTreeMap::new(),
            });
            for (&graph, &(resource, scale)) in &row.graph_resources {
                let graph = graph
                    .and_then(|index| graphs.get(index))
                    .ok_or_else(|| invalid("knapsack preparation graph"))?;
                let mut exists = false;
                for entry in &graph.resources {
                    if entry.id != resource {
                        continue;
                    }
                    if entry.variable_ids.first().is_none_or(|&id| id < 0) {
                        return Err(invalid("knapsack preparation resource variable"));
                    }
                    exists = true;
                }
                if !exists {
                    return Err(invalid("knapsack preparation resource ID"));
                }
                self.rows
                    .last_mut()
                    .unwrap()
                    .graph_resources
                    .entry(graph.id)
                    .or_insert((resource, scale));
            }
        }
        Ok(())
    }
    pub fn separate(&mut self, columns: &[KnapsackColumn]) -> Result<Vec<ResourceKnapsackCut>> {
        self.separate_impl(columns, None)
    }
    /// Columns use solver coordinates, while rows/cuts retain original IDs.
    pub fn separate_mapped(
        &mut self,
        columns: &[KnapsackColumn],
        indices: &BTreeMap<i32, BTreeMap<usize, usize>>,
    ) -> Result<Vec<ResourceKnapsackCut>> {
        self.separate_impl(columns, Some(indices))
    }
    fn separate_impl(
        &mut self,
        columns: &[KnapsackColumn],
        indices: Option<&BTreeMap<i32, BTreeMap<usize, usize>>>,
    ) -> Result<Vec<ResourceKnapsackCut>> {
        validate_parameters(&self.parameters)?;
        self.num_separate_calls += 1;
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            if row.rhs < 0 || row.rhs > i32::MAX / 8 {
                return Err(invalid("knapsack original rhs out of range"));
            }
            let mut totals = BTreeMap::new();
            for (&load, &value) in &row.load_totals {
                if !value.is_finite() || value < 0. {
                    return Err(invalid("knapsack load totals"));
                }
                totals.insert(load, (value * 1e8 + 0.5).floor() / 1e8);
            }
            for column in columns {
                if !column.value.is_finite() || column.value < 0. {
                    return Err(invalid("knapsack column value"));
                }
                let Some(&(resource, scale)) = row.graph_resources.get(&column.graph_id) else {
                    continue;
                };
                let resource = match indices {
                    Some(indices) => *indices
                        .get(&column.graph_id)
                        .and_then(|map| map.get(&resource))
                        .ok_or_else(|| invalid("knapsack original resource is not mapped"))?,
                    None => resource,
                };
                let q = *column
                    .resources
                    .get(resource)
                    .ok_or_else(|| invalid("knapsack sink resource missing"))?;
                let load = (q * scale + 1e-6).floor();
                if !load.is_finite() || load < 0. || load > (i32::MAX / 8) as f64 {
                    return Err(invalid("knapsack scaled load out of range"));
                }
                *totals.entry(load as i32).or_insert(0.) +=
                    (column.value * 1e8 + 0.5).floor() / 1e8;
            }
            rows.push((totals, row.rhs));
        }
        let separator = KnapsackSeparator::with_parameters(self.parameters.clone())?;
        let mut result = Vec::new();
        for (row, cut) in separator.separate_totals(&rows) {
            let b = &cut.thresholds;
            let rhs = cut.rhs as usize;
            let class = if b.is_empty() {
                0
            } else if (1..rhs).all(|i| i == rhs / 2 || b[i - 1] == b[i]) {
                2
            } else if rhs == 6 && b[0] == b[1] && b[4] == b[5] {
                3
            } else if rhs == 8 && b[0] == b[1] && b[2] == b[3] && b[4] == b[5] && b[6] == b[7] {
                4
            } else if rhs == 10 && b[0] == b[1] && b[2] == b[3] && b[6] == b[7] && b[8] == b[9] {
                5
            } else {
                cut.rhs
            };
            result.push(ResourceKnapsackCut {
                id: self.next_cut_id,
                row,
                cut_class: class,
                graph_resources: self.rows[row].graph_resources.clone(),
                cut,
                dual: 0.,
            });
            self.next_cut_id += 1;
        }
        Ok(result)
    }
}
