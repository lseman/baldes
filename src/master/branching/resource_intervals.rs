//! Weighted feasible schedule intervals for accumulated-resource candidates.
use crate::{invalid, Result};

/// Earliest and latest feasible head quantity for one route occurrence, with
/// its current positive LP weight. These are schedule intervals, not independent
/// bounds for different routes or a certificate of packing-set feasibility.
#[derive(Clone, Copy, Debug)]
pub struct ResourceInterval {
    pub lower: f64,
    pub upper: f64,
    pub weight: f64,
}

const EPSILON: f64 = 1e-6;

/// Native resource-generator activity at a stored threshold. This is the mass
/// strictly below the shared boundary, not the lower child's violation row.
/// Keeping that orientation makes U child one (upper) use distance `1-value`
/// and child two (lower) use `value` when recording history.
pub(crate) fn resource_branch_fraction(intervals: &[ResourceInterval], threshold: f64) -> f64 {
    let boundary = threshold - EPSILON;
    intervals
        .iter()
        .filter(|interval| interval.lower < boundary)
        .map(|interval| interval.weight)
        .fold(0., |sum, weight| sum + weight)
}

/// Select a stored threshold using tolerance-merged cumulative interval lists.
///
/// This reproduces the recovered native event walk, including insertion-order
/// effects when endpoints merge within epsilon and first-score tie retention.
/// The caller must supply complete forward/backward feasible intervals for one
/// packing group and resource. This function does not impose child restrictions
/// or establish validity of an original-domain bound.
pub fn select_resource_threshold(
    intervals: &[ResourceInterval],
    resource_upper: f64,
) -> Result<Option<f64>> {
    if !resource_upper.is_finite()
        || intervals.iter().any(|i| {
            !i.lower.is_finite()
                || !i.upper.is_finite()
                || !i.weight.is_finite()
                || i.weight <= 0.
                || i.lower > i.upper
                || i.upper > resource_upper
        })
    {
        return Err(invalid(
            "finite ordered resource intervals and positive weights required",
        ));
    }
    let mut lower = vec![(resource_upper, 0.)];
    let mut upper = vec![(resource_upper, 1.)];
    for interval in intervals {
        update(
            &mut lower,
            interval.lower,
            interval.weight,
            resource_upper,
            false,
        );
        update(
            &mut upper,
            interval.upper,
            interval.weight,
            resource_upper,
            true,
        );
    }
    let (mut i, mut j) = (0, 0);
    let mut threshold;
    if lower[0].0 <= upper[0].0 {
        threshold = lower[0].0;
        i += 1;
    } else {
        threshold = upper[0].0;
        j += 1;
    }
    let (mut left, mut right) = (0_f64, 1_f64);
    let (mut best, mut chosen) = (1. - EPSILON, None);
    while threshold < resource_upper {
        let score = left.max(right);
        if score < best {
            best = score;
            chosen = Some(threshold);
        }
        left = lower[i].1;
        right = upper[j].1;
        if lower[i].0 > upper[j].0 {
            threshold = upper[j].0;
            j += 1;
        } else {
            threshold = lower[i].0;
            i += 1;
        }
    }
    Ok(chosen)
}

fn update(nodes: &mut Vec<(f64, f64)>, quantity: f64, weight: f64, bound: f64, upper: bool) {
    if quantity > bound - EPSILON {
        return;
    }
    // The sentinel at `bound` remains after every accepted endpoint.
    let index = nodes.partition_point(|node| quantity - EPSILON > node.0);
    let inserted = nodes[index].0 > quantity + EPSILON;
    if inserted {
        nodes.insert(index, (quantity, nodes[index].1));
    }
    // A new lower endpoint copies the old cumulative value; its contribution
    // starts at the next node. A merged lower endpoint updates that node too.
    // Upper contributions always start after the matched/inserted endpoint.
    let first = index + usize::from(inserted || upper);
    for node in &mut nodes[first..] {
        if upper {
            node.1 -= weight;
        } else {
            node.1 += weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;



    #[test]
    fn no_split_and_invalid_intervals_are_distinct() {
        assert_eq!(select_resource_threshold(&[], 10.).unwrap(), None);
        let i = ResourceInterval {
            lower: 1.,
            upper: 2.,
            weight: 1.,
        };
        assert_eq!(select_resource_threshold(&[i], 10.).unwrap(), None);
        for bad in [
            ResourceInterval {
                lower: f64::NAN,
                ..i
            },
            ResourceInterval {
                upper: f64::INFINITY,
                ..i
            },
            ResourceInterval { lower: 3., ..i },
            ResourceInterval { weight: 0., ..i },
            ResourceInterval { weight: -1., ..i },
        ] {
            assert!(select_resource_threshold(&[bad], 10.).is_err());
        }
        assert!(select_resource_threshold(&[i], 1.).is_err());
        assert!(select_resource_threshold(&[i], f64::INFINITY).is_err());
    }
}
