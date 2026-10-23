//! Normalize complete finite columns to original-variable equality links.
use super::*;

pub(super) fn model<const D: usize>(
    mut problem: Problem<D>,
    mut variables: Vec<MasterVariable>,
    subproblems: Vec<Subproblem>,
) -> Result<(Problem<D>, Vec<MasterVariable>, Registry)> {
    problem.validate()?;
    super::super::master_variables::validate(&problem, &variables)?;
    let original: Vec<_> = variables.iter().map(|v| v.id).collect();
    let indices: HashMap<_, _> = original
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut used: HashSet<_> = original.iter().copied().collect();
    let mut groups: HashSet<_> = problem.networks.iter().map(|n| n.id).collect();
    // Validate original coordinates before appending any generated linking row.
    for group in &subproblems {
        if group.id < 0 || !groups.insert(group.id) {
            return Err(invalid("duplicate imported pricing subproblem"));
        }
        quantity_bounds(group.multiplicity)?;
        let mut support = HashSet::new();
        for &id in &group.variables {
            if !indices.contains_key(&id) || !support.insert(id) {
                return Err(invalid("imported original-variable support"));
            }
        }
        let mut columns = HashSet::new();
        for column in &group.columns {
            quantity_bounds(column.bounds)?;
            if column.id < 0 || !columns.insert(column.id) || !column.cost.is_finite() {
                return Err(invalid("imported column identity or cost"));
            }
            let mut entries = HashSet::new();
            if column
                .entries
                .iter()
                .any(|&(id, a)| !support.contains(&id) || !a.is_finite() || !entries.insert(id))
            {
                return Err(invalid("coalesced imported column entries required"));
            }
        }
    }
    let mut links = BTreeMap::new();
    for group in &subproblems {
        for &id in &group.variables {
            if links.contains_key(&id) {
                continue;
            }
            let row = problem.rows.len();
            problem.rows.push(Row {
                sense: RowSense::Equal,
                rhs: 0.,
            });
            variables[indices[&id]].entries.push((row, 1.));
            links.insert(id, row);
        }
    }
    let mut registry = Registry {
        original,
        columns: BTreeMap::new(),
        groups: BTreeMap::new(),
        group_rows: BTreeMap::new(),
    };
    let mut next = 0i64;
    for group in subproblems {
        let mut rows = vec![];
        for (sense, rhs) in [
            (RowSense::GreaterEqual, group.multiplicity.lower),
            (RowSense::LessEqual, group.multiplicity.upper),
        ] {
            if !rhs.is_finite() {
                continue;
            }
            let row = problem.rows.len();
            problem.rows.push(Row { sense, rhs });
            rows.push(row);
        }
        let column_ids: HashSet<_> = group.columns.iter().map(|c| c.id).collect();
        registry.group_rows.insert(
            group.id,
            GroupRows {
                links: group.variables.iter().map(|id| (*id, links[id])).collect(),
                multiplicity: rows.clone(),
                next_column: (0..=i32::MAX).find(|id| !column_ids.contains(id)),
            },
        );
        let members = registry.groups.entry(group.id).or_default();
        for column in group.columns {
            while next <= i64::from(i32::MAX) && used.contains(&(next as i32)) {
                next += 1;
            }
            let id = i32::try_from(next)
                .map_err(|_| invalid("imported column variable IDs exhausted"))?;
            used.insert(id);
            next += 1;
            registry.columns.insert((group.id, column.id), id);
            members.push(id);
            let mut entries: Vec<_> = column
                .entries
                .into_iter()
                .filter(|(_, a)| *a != 0.)
                .map(|(id, a)| (links[&id], -a))
                .collect();
            entries.extend(rows.iter().map(|&row| (row, 1.)));
            entries.sort_unstable_by_key(|&(r, _)| r);
            variables.push(MasterVariable {
                id,
                cost: column.cost,
                bounds: column.bounds,
                integer: column.integer,
                entries,
            });
        }
    }
    Ok((problem, variables, registry))
}
