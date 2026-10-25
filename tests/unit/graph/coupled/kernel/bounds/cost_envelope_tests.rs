use super::*;
#[test]
fn composed_envelopes_cover_ordered_sums_with_cancellation() {
    for values in [
        vec![1e16, 1., -1e16, 1.],
        vec![-3., 2., 1., -1., -2., 4.],
        vec![f64::from_bits(1); 8],
    ] {
        let mut envelope = CostEnvelope::default();
        // Suffixes compose in reverse construction order, whereas replay
        // sums charges in forward execution order.
        for &value in values.iter().rev() {
            envelope = CostEnvelope::charge(Range::point(value)).combine(&envelope);
        }
        let sum = values.into_iter().fold(0., |sum, x| sum + x);
        let range = envelope.range();
        assert!(
            range.lo <= sum && sum <= range.hi,
            "sum={sum} range={range:?}"
        );
    }
}
