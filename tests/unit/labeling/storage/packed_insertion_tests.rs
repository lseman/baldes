use crate::cuts::state::{DominanceDescriptor, LIMITED_WORDS};
use crate::graph::buckets::BucketId;
use crate::labeling::storage::{InsertionTest, LabelStorage, StoredLabel};
use crate::labeling::Statistics;
use crate::Graph;

fn label(cost: f64, words: [u32; LIMITED_WORDS]) -> StoredLabel<2> {
    StoredLabel {
        id: 0,
        local_id: 0,
        bucket: BucketId {
            vertex: 0,
            bucket: 0,
        },
        vertex: 0,
        cost,
        original_cost: cost,
        resources: [0., 0.].into(),
        ng: 0,
        branch_state: None,
        presence_state: None,
        coupled_state: None,
        parent: None,
        arc: None,
        hops: 1,
        expanded: false,
        cut_words: words,
        additional_cost: 0.,
        half_potential: 0.,
    }
}

fn descriptor(word: i32, subtract_cost: bool, cost: f64) -> DominanceDescriptor {
    DominanceDescriptor {
        mask: 1,
        denominator: 2,
        word,
        shift: 0,
        subtract_cost,
        cost,
    }
}

fn inserted_pair(
    cuts: &[DominanceDescriptor],
    left: [u32; LIMITED_WORDS],
    right: [u32; LIMITED_WORDS],
) -> usize {
    let graph = Graph::<2>::new(2, 2).unwrap();
    let mut storage = LabelStorage::new([1]);
    storage.labels.set_word_limit(LIMITED_WORDS);
    for candidate in [label(0., left), label(0.5, right)] {
        storage.insert(
            &graph,
            candidate,
            true,
            InsertionTest::Full,
            8,
            false,
            cuts,
            &mut Statistics::default(),
        );
    }
    storage.buckets[0][0].active.len()
}

#[test]
fn nonempty_cut_state_can_prevent_cheaper_prefix_dominance() {
    let mut left = [0; LIMITED_WORDS];
    left[0] = 1;
    assert_eq!(inserted_pair(&[], left, [0; LIMITED_WORDS]), 1);
    assert_eq!(
        inserted_pair(&[descriptor(0, false, 1.)], left, [0; LIMITED_WORDS]),
        2
    );
}

#[test]
fn cut_descriptor_veto_is_ordered_and_cannot_be_rescued_later() {
    let mut left = [0; LIMITED_WORDS];
    left[0] = 1;
    left[1] = 1;
    let descriptors = [descriptor(0, true, 1.), descriptor(1, false, 1.25)];
    assert_eq!(inserted_pair(&descriptors, left, [0; LIMITED_WORDS]), 1);
    assert_eq!(
        inserted_pair(&[descriptors[1], descriptors[0]], left, [0; LIMITED_WORDS]),
        2
    );
}

#[test]
fn high_word_signed_extraction_retains_its_dominance_direction() {
    let descriptor = DominanceDescriptor {
        mask: u32::MAX,
        denominator: -2,
        word: 19,
        shift: 31,
        subtract_cost: false,
        cost: 1.,
    };
    let mut signed = [0; LIMITED_WORDS];
    signed[19] = 0x8000_0000;
    assert_eq!(inserted_pair(&[descriptor], signed, [0; LIMITED_WORDS]), 1);
    assert_eq!(inserted_pair(&[descriptor], [0; LIMITED_WORDS], signed), 2);
}
