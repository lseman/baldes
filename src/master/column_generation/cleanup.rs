use super::*;

/// Alg4EvalByLagrangianDuality::cleanupRestrictedMastColumns,
/// run before pricing. With N generated columns above the threshold, columns
/// in the primal solution stay; the others are stably sorted by reduced cost
/// (Double tolerance 1e-10 relative + 1e-6, creation order on ties) and only
/// the first (int)(N * ratio) minus the solution columns are kept.
pub(super) fn cleanup_columns<M: LpMaster>(
    master: &M,
    tags: &[i32],
    options: &Options,
) -> Vec<usize> {
    let threshold = options.column_cleanup_threshold as i32;
    let generated: Vec<usize> = (0..tags.len()).filter(|&i| tags[i] >= 0).collect();
    let count = generated.len() as i32;
    if threshold >= count {
        return vec![];
    }
    let (solution, mut candidates): (Vec<usize>, Vec<usize>) = generated
        .into_iter()
        .partition(|&i| master.values()[i] != 0.);
    let in_solution = solution.len() as i32;
    if candidates.len() as i32 + in_solution <= threshold {
        return vec![];
    }
    let keep = (count as f32 * options.column_cleanup_ratio) as i32;
    if in_solution >= keep {
        return vec![];
    }
    let rc = master.reduced_costs();
    crate::cuts::stdsort::stable_sort(&mut candidates, |&a, &b| {
        let tolerance = rc[b].abs().max(rc[a].abs()) * 1e-10 + 1e-6;
        if rc[b] - tolerance > rc[a] {
            return true;
        }
        if rc[a] - tolerance > rc[b] {
            return false;
        }
        a < b
    });
    let mut removed = candidates.split_off((keep - in_solution) as usize);
    removed.sort_unstable();
    removed
}
