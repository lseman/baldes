//! Publish inferred bounds that upper caps on signed components cannot express.
use super::*;

/// Canonical component upper caps enforce bounds straddling zero. A positive
/// original lower bound or negative original upper bound additionally needs an
/// original-variable row. Metadata and warm artificials must see the same row.
pub(in crate::master::model) fn bounds<const D: usize>(
    master: &mut ClpMaster,
    state: &mut State<D>,
    previous: &[Bounds],
    known: &mut Vec<cg::Column>,
    mut warm: Option<&mut cg::WarmStart>,
) -> Result<()> {
    let mut inferred = vec![];
    for (index, (variable, old)) in state.master_variables.iter().zip(previous).enumerate() {
        if variable.bounds.lower > 0. && variable.bounds.lower > old.lower {
            inferred.push((index, RowSense::GreaterEqual, variable.bounds.lower));
        }
        if variable.bounds.upper < 0. && variable.bounds.upper < old.upper {
            inferred.push((index, RowSense::LessEqual, variable.bounds.upper));
        }
    }
    for (index, sense, rhs) in inferred {
        let shared = state.shared_rows.len();
        let physical = master.add_row(sense, rhs, &[(2 * index, 1.), (2 * index + 1, -1.)])?;
        state.shared_rows.push(Row { sense, rhs });
        state.master_variables[index].entries.push((shared, 1.));
        let update = |metadata: &mut cg::Column| {
            if let Some((v, sign)) = decode(metadata.tag) {
                if v == index {
                    metadata.entries.push((physical, sign));
                }
            }
        };
        for metadata in known.iter_mut() {
            update(metadata);
        }
        if let Some(warm) = warm.as_deref_mut() {
            for metadata in warm.columns.iter_mut().flatten() {
                update(metadata);
            }
            for metadata in &mut warm.generated_pool {
                update(metadata);
            }
            warm.stabilization = None;
            let penalty = warm
                .artificials
                .iter()
                .map(|&c| master.costs[c])
                .fold(2.76_f64, f64::max);
            let sign = if sense == RowSense::GreaterEqual {
                1.
            } else {
                -1.
            };
            let column = master.add_column(penalty, &[(physical, sign)])?;
            warm.columns.push(None);
            warm.artificials.push(column);
        }
    }
    state.rhs = master.rhs.clone();
    Ok(())
}
