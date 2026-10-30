//! Initial restricted-master columns, pools, and output state.
use super::*;

pub(super) struct InheritedBounds {
    pub active: f64,
    pub numerical: f64,
    pub certified: f64,
}

pub(super) struct InitialState {
    pub loaded: Vec<Option<Column>>,
    pub seen: HashSet<ColumnKey>,
    pub pool: Vec<Column>,
    pub pool_seen: HashSet<ColumnKey>,
    pub out: Output,
}

pub(super) fn initialize<M: LpMaster, C: Controller<M>>(
    master: &mut M,
    controller: &C,
    known: &mut Vec<Column>,
    bound_mode: BoundMode,
    artificial_cost: f64,
    warm: &mut Option<WarmStart>,
    bounds: InheritedBounds,
) -> Result<InitialState> {
    let is_resume = warm.is_some();
    let mut loaded: Vec<Option<Column>> = match warm.as_mut() {
        Some(warm) => std::mem::take(&mut warm.columns),
        None => std::mem::take(known).into_iter().map(Some).collect(),
    };
    loaded.resize(master.costs().len(), None);
    for (index, column) in loaded.iter().enumerate() {
        if let Some(column) = column {
            apply_column_cap(master, controller, index, column.tag)?;
        }
    }
    let seen: HashSet<_> = loaded.iter().flatten().map(Column::signature).collect();
    // Generated columns exist independently of membership in the restricted
    // master. Columns omitted by a pricing insertion test can become
    // eligible under the strict regeneration cutoff.
    let mut pool: Vec<Column> = warm
        .as_mut()
        .map_or_else(Vec::new, |warm| std::mem::take(&mut warm.generated_pool));
    let mut pool_seen: HashSet<_> = pool.iter().map(Column::signature).collect();
    // Include newly supplied initial routes without losing same-batch
    // duplicates when the caller starts with only a populated master.
    let missing: Vec<_> = loaded
        .iter()
        .flatten()
        .filter(|column| !pool_seen.contains(&column.signature()))
        .cloned()
        .collect();
    pool_seen.extend(missing.iter().map(Column::signature));
    pool.extend(missing);
    let mut out = Output {
        status: Status::IterationLimit,
        feasibility: vec![],
        enumeration_requested: false,
        enumeration_failure: None,
        lp: 0.,
        bound: bounds.active,
        bound_mode,
        numerical_bound: bounds.numerical,
        certified_bound: bounds.certified,
        iterations: 0,
        pricing_calls: 0,
        columns: master.costs().len(),
        history: vec![],
        column_tags: loaded
            .iter()
            .map(|c| c.as_ref().map_or(-1, |c| c.tag))
            .collect(),
        inserted_tags: vec![],
        artificials: warm
            .as_mut()
            .map_or_else(Vec::new, |w| std::mem::take(&mut w.artificials)),
        penalty_updates: 0,
        pruned_by_bound: false,
        cut_rounds: 0,
        warm_start: WarmStart {
            bound: f64::NEG_INFINITY,
            bound_mode,
            numerical_bound: f64::NEG_INFINITY,
            certified_bound: f64::NEG_INFINITY,
            stabilization: None,
            proximal: None,
            generated_pool: vec![],
            columns: vec![],
            artificials: vec![],
            cut_artificials: vec![],
            initial_rows: 0,
        },
    };
    out.column_tags.resize(master.costs().len(), -1);
    if artificial_cost > 0. && !is_resume {
        let rows = 0..master.rhs().len();
        for (_, column) in append_artificials(master, rows, artificial_cost)? {
            out.artificials.push(column);
            loaded.push(None);
            out.column_tags.push(-1);
        }
    }
    Ok(InitialState {
        loaded,
        seen,
        pool,
        pool_seen,
        out,
    })
}
