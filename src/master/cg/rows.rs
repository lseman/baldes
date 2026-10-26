//! Shared artificial-column installation for initial, cut and branch rows.
use super::*;

/// Return `(row, column)` identities in row/sign order. All insertion paths
/// use the same sense convention and the backend's batched column operation.
pub(crate) fn append_artificials<M: LpMaster>(
    master: &mut M,
    rows: std::ops::Range<usize>,
    penalty: f64,
) -> Result<Vec<(usize, usize)>> {
    if rows.start > rows.end || rows.end > master.rhs().len() || !penalty.is_finite() {
        return Err(invalid("artificial row range or penalty"));
    }
    if penalty <= 0. {
        return Ok(vec![]);
    }
    let mut entries = vec![];
    for row in rows {
        if master.senses()[row] != RowSense::LessEqual {
            entries.push([(row, 1.)]);
        }
        if master.senses()[row] != RowSense::GreaterEqual {
            entries.push([(row, -1.)]);
        }
    }
    let batch: Vec<_> = entries
        .iter()
        .map(|entry| (penalty, entry.as_slice()))
        .collect();
    let columns = master.add_columns(&batch)?;
    Ok(entries
        .iter()
        .zip(columns)
        .map(|(entry, column)| (entry[0].0, column))
        .collect())
}

#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/cg/rows_tests.rs"]
mod tests;
