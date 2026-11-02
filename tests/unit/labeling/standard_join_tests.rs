use super::*;

#[test]
fn standard_join_matches_captured_resource_and_overlap_boundaries() {
    // The reference helper received copied live labels and a prepared
    // one-resource owner; its inputs included values around 1e-6 tolerance.
    let input = include_str!("../../fixtures/enumeration/standard-join-reference-capture.txt");
    let graph = Graph::<2>::new(2, 1).unwrap();
    let mut count = 0;
    for line in input.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        let forward: f64 = fields[0].parse().unwrap();
        let backward: f64 = fields[1].parse().unwrap();
        let left: u64 = fields[2].parse().unwrap();
        let right: u64 = fields[3].parse().unwrap();
        let expected = fields[4] == "1";
        let actual = left & right == 0
            && standard_resources_join(&graph, &[forward, 0.], &[backward, 0.], false);
        assert_eq!(actual, expected, "case {count}: {line}");
        count += 1;
    }
    assert_eq!(count, 180);
}

#[test]
fn symmetric_standard_join_matches_captured_boundaries() {
    // The symmetry flag was deliberately enabled on a prepared native owner.
    let input =
        include_str!("../../fixtures/enumeration/standard-join-symmetric-reference-capture.txt");
    let mut graph = Graph::<2>::new(2, 1).unwrap();
    graph.set_bounds(graph.source, 0, 0., 2.).unwrap();
    graph.set_bounds(graph.sink, 0, 0., 2.).unwrap();
    let mut count = 0;
    for line in input.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        let forward: f64 = fields[0].parse().unwrap();
        let backward: f64 = fields[1].parse().unwrap();
        let left: u64 = fields[2].parse().unwrap();
        let right: u64 = fields[3].parse().unwrap();
        let expected = fields[4] == "1";
        let actual = left & right == 0
            && standard_resources_join(&graph, &[forward, 0.], &[backward, 0.], true);
        assert_eq!(actual, expected, "case {count}: {line}");
        count += 1;
    }
    assert_eq!(count, 180);
}
