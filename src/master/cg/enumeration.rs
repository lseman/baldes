//! Atomic enumeration-to-master handoff. The live LP and CG metadata are only
//! replaced after every fallible controller and backend operation succeeds.
use super::controller::apply_column_cap;
use super::*;

pub(super) struct Transfer<M> {
    pub master: M,
    pub output: Output,
    pub loaded: Vec<Option<Column>>,
    pub cut_artificials: Vec<(usize, usize)>,
    pub seen: HashSet<ColumnKey>,
    pub pool: Vec<Column>,
    pub pool_seen: HashSet<ColumnKey>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transfer<M: LpMaster, C: Controller<M>>(
    master: &M,
    controller: &mut C,
    duals: &[f64],
    output: &Output,
    loaded: &[Option<Column>],
    cut_artificials: &[(usize, usize)],
    pool: &[Column],
    pool_seen: &HashSet<ColumnKey>,
    insertion_limit: usize,
) -> Result<Option<Transfer<M>>> {
    if !controller.begin_enumeration_transaction(output.bound)? {
        return Ok(None);
    }
    let mut destination = match master.try_clone_for_transaction() {
        Ok(Some(destination)) => destination,
        Ok(None) => {
            controller.finish_enumeration_transaction(false);
            return Ok(None);
        }
        Err(error) => {
            controller.finish_enumeration_transaction(false);
            return Err(error);
        }
    };
    let result = (|| {
        let Some(universe) = controller.enumerated_universe(duals, output.bound)? else {
            return Ok(None);
        };
        // Cached admissible paths need not have a representative's identity.
        let regenerated =
            controller.regenerate_enumerated_columns(pool, duals, output.bound, &universe)?;
        let removed: Vec<_> = (0..loaded.len()).filter(|&i| loaded[i].is_some()).collect();
        let (_, map) = destination.remove(&[], &removed)?;
        let mut next = Transfer {
            master: destination,
            output: output.clone(),
            loaded: loaded
                .iter()
                .zip(&map)
                .filter_map(|(column, retained)| retained.map(|_| column.clone()))
                .collect(),
            cut_artificials: cut_artificials
                .iter()
                .map(|&(row, column)| {
                    map[column]
                        .map(|column| (row, column))
                        .ok_or_else(|| invalid("enumeration removed a cut artificial"))
                })
                .collect::<Result<_>>()?,
            // Removed paths remain eligible for reinsertion.
            seen: HashSet::new(),
            pool: pool.to_vec(),
            pool_seen: pool_seen.clone(),
        };
        next.output.column_tags = output
            .column_tags
            .iter()
            .zip(&map)
            .filter_map(|(&tag, retained)| retained.map(|_| tag))
            .collect();
        next.output.artificials = output.artificials.iter().filter_map(|&i| map[i]).collect();
        let mut loaded_set = HashSet::new();
        for column in regenerated {
            let index = next.master.add_column(column.cost, &column.entries)?;
            apply_column_cap(&mut next.master, controller, index, column.tag)?;
            loaded_set.insert(column.signature());
            next.seen.insert(column.signature());
            next.output.column_tags.push(column.tag);
            next.loaded.push(Some(column));
        }
        let mut missing: Vec<_> = universe
            .iter()
            .filter(|column| !loaded_set.contains(&column.signature()))
            .collect();
        missing.sort_by(|a, b| a.reduced_cost.total_cmp(&b.reduced_cost));
        missing.truncate(insertion_limit);
        for column in missing {
            let identity = column.signature();
            if !loaded_set.insert(identity.clone()) {
                continue;
            }
            let index = next.master.add_column(column.cost, &column.entries)?;
            apply_column_cap(&mut next.master, controller, index, column.tag)?;
            next.output.column_tags.push(column.tag);
            next.output.inserted_tags.push(column.tag);
            next.seen.insert(identity);
            if next.pool_seen.insert(column.signature()) {
                next.pool.push(column.clone());
            }
            next.loaded.push(Some(column.clone()));
        }
        // Controller rollback covers remapping and pool refresh even if either
        // changes some external metadata before returning an error.
        controller.remap_columns(&map)?;
        let current: HashMap<_, _> = next
            .loaded
            .iter()
            .flatten()
            .map(|column| (column.signature(), column))
            .collect();
        for column in &mut next.pool {
            if let Some(refreshed) = current
                .get(&column.signature())
                .filter(|refreshed| refreshed.cost == column.cost)
            {
                column.entries = refreshed.entries.clone();
                column.reduced_cost = refreshed.reduced_cost;
            }
        }
        controller.synchronize_generated_columns(&mut next.pool, &next.loaded)?;
        next.output.columns = next.master.costs().len();
        next.output.enumeration_failure = None;
        Ok(Some(next))
    })();
    controller.finish_enumeration_transaction(result.is_ok());
    result
}
