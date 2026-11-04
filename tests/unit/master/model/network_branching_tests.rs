use super::*;

#[test]
fn multiplicity_first_matches_the_mapped_native_first_child_and_restores_parent() {
    let problem = format::Model::from_json(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/networks/triangle.json"
    )))
    .unwrap()
    .problem::<2>()
    .unwrap();
    let mut options = Options::default();
    options.cg.stabilize = false;
    options.cg.artificial_cost = 100.;
    let root = solve_with_master_variables(problem, vec![], options.clone()).unwrap();
    assert_eq!(root.master.lp, 1.5);
    // The unified dispatcher with the MultiplicityFirst preset, no strong trials.
    let configured = ModelTreeOptions {
        branching_registry: {
            let mut __branch_registry = crate::master::model::BranchingRegistry::default();
            __branch_registry.strong_branching = None;
            __branch_registry
        },
        ..Default::default()
    };
    let proposals = super::super::proposals::collect(&root, &configured, &[]).unwrap();
    let mut selection = model_strong::choose_proposals(
        &root,
        &configured,
        &mut model_strong::History::default(),
        f64::INFINITY,
        Instant::now() + std::time::Duration::from_secs(10),
        proposals,
    )
    .unwrap()
    .unwrap();
    selection.admit(NetworkBranchDirection::UpperFirst);
    let branches = selection.children;
    assert_eq!(branches.len(), 2);
    let first = branches[0].solve(options.clone()).unwrap();
    // In the public native control, fractional network42 count .5 splits
    // upward first, and requiring one copy raises the exact LP to5/3.
    assert!(
        (first.master.lp - 5. / 3.).abs() < 1e-8,
        "{:?}",
        first.master
    );
    let quantity: f64 = first
        .routes
        .iter()
        .zip(&first.values)
        .filter(|(route, _)| route.network == 42)
        .map(|(_, value)| value)
        .sum();
    assert!((quantity - 1.).abs() < 1e-8);
    assert_eq!(root.snapshot.solve(options).unwrap().master.lp, 1.5);
}

#[test]
fn default_network_branching_uses_the_recovered_native_count_first_policy() {
    assert_eq!(
        ModelTreeOptions::default().branching_registry.branching,
        NetworkBranchingPolicy::MultiplicityFirst
    );
    assert_eq!(
        NetworkBranchingPolicy::parse("resource-first").unwrap(),
        NetworkBranchingPolicy::ResourceFirst
    );
}

#[test]
fn sequential_tree_evaluation_carries_incidence_history_to_siblings() {
    use crate::cuts::separation::{Input, Solution};
    use crate::master::tree::{Model, QueuedNode, TreeFrontier as Queue};

    let (problem, config) = crate::master::model::separation::mixed_arc_problem_and_config();
    let mut options = ModelTreeOptions::default();
    options.node.cg.stabilize = false;
    options.node.preprocessing.enabled = false;
    options.node.separation.rank_one_incidence = Some(config);
    let root = solve_with_master_variables(problem, vec![], options.node.clone()).unwrap();
    let initial = root.snapshot.incidence_separator_history().unwrap();
    let initial_count = initial.generated_cut_count();
    let mut history = initial;
    let capture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/cuts/rank_one_two_graph_mixed_complete_separation_native.json"
    ))
    .unwrap();
    let call = &capture["call"];
    let columns: Vec<_> = call["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|column| Solution {
            graph_id: column["graph"].as_i64().unwrap() as i32,
            enumerated: column["enumerated"].as_i64().unwrap() != 0,
            arc_ids: column["arc_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_i64().unwrap() as i32)
                .collect(),
        })
        .collect();
    let input = Input {
        solutions: columns.iter().collect(),
        values: call["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|column| column["value"].as_f64().unwrap())
            .collect(),
        cut_memory_type: 1,
        solutions_are_set_sequences: false,
        separation_phase: 1,
        active_cuts: vec![],
    };
    let mut registry = crate::cuts::CutRegistry::default();
    assert!(!history
        .separate_and_register(&mut registry, &input)
        .unwrap()
        .is_empty());
    let advanced = history.generated_cut_count();
    assert!(advanced > initial_count);
    let mut result = ModelTreeOutput {
        status: TreeStatus::Incomplete,
        bound_mode: cg::BoundMode::Certified,
        bound: f64::NEG_INFINITY,
        incumbent: f64::INFINITY,
        routes: vec![],
        master_values: vec![],
        nodes: 0,
        processed_nodes: 0,
        parallel_batches: 0,
        strong_branching_trials: 0,
        mapped_history_ranked: 0,
        mapped_branch_evaluations: vec![],
        open_nodes: 0,
        iterations: 0,
        pricing_calls: 0,
        oracle_calls: vec![],
        enumerated_mip_calls: 0,
        enumerated_mip_nodes: 0,
        enumerated_mip_failures: 0,
        enumeration_reports: vec![],
        diving_nodes: 0,
        greedy_nodes: 0,
        restricted_mip_calls: 0,
    };
    let mut subtree_history = None;
    let mut mapped_history = model_strong::History::default();
    mapped_history.incidence_history = Some(history);
    let mut selected_nodes = 0;
    let mut unresolved = None;
    let mut closed_bound = f64::INFINITY;
    let mut search = super::super::adapter::NetworkSearch {
        options: &options,
        result: &mut result,
        deadline: Instant::now() + std::time::Duration::from_secs(10),
        windows: &[],
        base_depth: 0,
        subtree_history: &mut subtree_history,
        mapped_history: &mut mapped_history,
        selected_nodes: &mut selected_nodes,
        unresolved: &mut unresolved,
        closed_bound: &mut closed_bound,
        workers: None,
        local_search_rng: crate::rng::GnuRand::new(1),
    };
    let mut queue = Queue::new(
        options.tree_search,
        options.open_node_limit,
        options.strategy2_bound_rounding,
    );
    for serial in 0..2 {
        let mut entry = QueuedNode {
            node: root.snapshot.fork().unwrap(),
            evaluated: None,
            serial,
            branch_order: 0,
            parent_branch_order: 0,
            strategy2_dual_bound: 0.,
        };
        let output = search
            .evaluate(std::slice::from_mut(&mut entry), &mut queue)
            .unwrap()
            .remove(0)
            .unwrap();
        assert!(
            output
                .snapshot
                .incidence_separator_history()
                .unwrap()
                .generated_cut_count()
                >= advanced
        );
        assert!(
            search
                .mapped_history
                .incidence_history
                .as_ref()
                .unwrap()
                .generated_cut_count()
                >= advanced
        );
        assert_eq!(
            entry
                .node
                .incidence_separator_history()
                .unwrap()
                .generated_cut_count(),
            initial_count
        );
    }
    assert_eq!(
        root.snapshot
            .incidence_separator_history()
            .unwrap()
            .generated_cut_count(),
        initial_count
    );
    let advanced_history = search.mapped_history.incidence_history.clone().unwrap();
    drop(search);
    let mut strong_options = options.clone();
    let mut strong = crate::master::tree::StrongBranchingOptions::default();
    strong.phases = vec![crate::master::tree::StrongPhase {
        candidates: 1,
        subtree_ratio: 0.,
        reduced_cost_fixing: false,
        ..Default::default()
    }];
    strong_options.branching_registry.strong_branching = Some(strong);
    let mut trial_history = model_strong::History::default();
    trial_history.incidence_history = Some(advanced_history);
    let selection = model_strong::choose_proposals(
        &root,
        &strong_options,
        &mut trial_history,
        f64::INFINITY,
        Instant::now() + std::time::Duration::from_secs(10),
        vec![model_strong::Proposal {
            name: "paths:7".into(),
            history_key: "paths:7".into(),
            priority: 1.,
            value: 1.5,
            restriction: model_strong::Restriction::Multiplicity { network: 7 },
            history_eligible: false,
            fresh: true,
            origin: None,
            component_admission: None,
        }],
    )
    .unwrap()
    .unwrap();
    assert!(trial_history.trials > 0);
    let evaluated: Vec<_> = selection.evaluated.iter().flatten().collect();
    assert!(!evaluated.is_empty());
    for result in evaluated {
        assert!(
            result
                .snapshot
                .incidence_separator_history()
                .unwrap()
                .generated_cut_count()
                >= advanced
        );
    }
}

#[test]
fn registered_tree_pairs_network_identities_sides_ranks_and_feasible_node_bounds() {
    use crate::master::tree::{Decision, Limits, Model, QueuedNode, Stop, TreeFrontier as Queue};
    use serde_json::{json, Value};
    struct Observed<M> {
        inner: M,
        evaluations: Vec<Value>,
        children: Vec<Value>,
    }
    impl<M: Model<Node = NodeSnapshot<2>, Output = Output<2>>> Model for Observed<M> {
        type Node = NodeSnapshot<2>;
        type Output = Output<2>;
        fn limits(&self) -> Limits {
            self.inner.limits()
        }
        fn prune(&mut self, queue: &mut Queue<Self::Node, Self::Output>) {
            self.inner.prune(queue)
        }
        fn admit(&mut self, entry: &mut QueuedNode<Self::Node, Self::Output>) -> bool {
            self.inner.admit(entry)
        }
        fn evaluate(
            &mut self,
            entries: &mut [QueuedNode<Self::Node, Self::Output>],
            queue: &mut Queue<Self::Node, Self::Output>,
        ) -> Result<Vec<Result<Self::Output>>> {
            self.inner.evaluate(entries, queue)
        }
        fn consume(
            &mut self,
            entry: Option<QueuedNode<Self::Node, Self::Output>>,
            output: Result<Self::Output>,
            reused: bool,
            queue: &mut Queue<Self::Node, Self::Output>,
        ) -> Result<Decision> {
            let id = entry.as_ref().map_or(0, |entry| entry.serial + 1);
            let rank = entry.as_ref().map_or(1, |entry| entry.branch_order);
            let solved = output.as_ref().unwrap();
            self.evaluations.push(
                json!({"node":id,"rank":rank,"status":format!("{:?}",solved.master.status),
                "lp":solved.master.lp,"numerical_bound":solved.master.numerical_bound,
                "selected_bound":solved.master.bound,"bound_infinite":solved.master.bound==f64::INFINITY}),
            );
            let old: HashSet<_> = queue.iter().map(|entry| entry.serial).collect();
            let decision = self.inner.consume(entry, output, reused, queue)?;
            let mut added: Vec<_> = queue
                .iter()
                .filter(|entry| !old.contains(&entry.serial))
                .collect();
            added.sort_by_key(|entry| entry.serial);
            for entry in added {
                let state = &entry.node.state;
                let row_index = state.shared_rows.len() - 1;
                let row = &state.shared_rows[row_index];
                let networks: Vec<_> = state
                    .oracles
                    .iter()
                    .filter(|oracle| oracle.network.route_entries.contains(&(row_index, 1.)))
                    .collect();
                assert_eq!(networks.len(), 1);
                let operator = match row.sense {
                    RowSense::GreaterEqual => "G",
                    RowSense::LessEqual => "L",
                    RowSense::Equal => "E",
                };
                self.children.push(json!({"node":entry.serial+1,"parent":id,"network":networks[0].network.id,"operator":operator,"threshold":row.rhs}));
            }
            Ok(decision)
        }
    }
    let control: Value = serde_json::from_str(include_str!(
        "../../../fixtures/metasolver_branch_identity.json"
    ))
    .unwrap();
    assert_eq!(
        control["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let mut traces = vec![];
    for control in control["cases"].as_array().unwrap() {
        let problem = format::Model::from_json(&control["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.node.cg.bound_mode = cg::BoundMode::Certified;
        // The reference runner records these controls without strong branching.
        options.branching_registry.strong_branching = None;
        let root = solve_with_master_variables(problem, vec![], options.node.clone()).unwrap();
        let mut result = ModelTreeOutput {
            status: TreeStatus::Incomplete,
            bound_mode: cg::BoundMode::Certified,
            bound: f64::NEG_INFINITY,
            incumbent: f64::INFINITY,
            routes: vec![],
            master_values: vec![],
            nodes: 0,
            processed_nodes: 0,
            parallel_batches: 0,
            strong_branching_trials: 0,
            mapped_history_ranked: 0,
            mapped_branch_evaluations: vec![],
            open_nodes: 0,
            iterations: 0,
            pricing_calls: 0,
            oracle_calls: root
                .snapshot
                .state
                .oracles
                .iter()
                .map(|o| (o.network.id, 0))
                .collect(),
            enumerated_mip_calls: 0,
            enumerated_mip_nodes: 0,
            enumerated_mip_failures: 0,
            enumeration_reports: vec![],
            diving_nodes: 0,
            greedy_nodes: 0,
            restricted_mip_calls: 0,
        };
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        let mut subtree_history = None;
        let mut mapped_history = model_strong::History::default();
        let mut selected_nodes = 0;
        let mut unresolved = None;
        let mut closed_bound = f64::INFINITY;
        let inner = super::super::adapter::NetworkSearch {
            options: &options,
            result: &mut result,
            deadline,
            windows: &[],
            base_depth: 0,
            subtree_history: &mut subtree_history,
            mapped_history: &mut mapped_history,
            selected_nodes: &mut selected_nodes,
            unresolved: &mut unresolved,
            closed_bound: &mut closed_bound,
            workers: None,
            local_search_rng: crate::rng::GnuRand::new(1),
        };
        let mut observer = Observed {
            inner,
            evaluations: vec![],
            children: vec![],
        };
        let mut queue = Queue::new(
            options.tree_search,
            options.open_node_limit,
            options.strategy2_bound_rounding,
        );
        assert_eq!(
            crate::master::tree::run(&mut observer, &mut queue, Some(root)).unwrap(),
            Stop::Exhausted
        );
        let expected_children = control["children"].as_array().unwrap();
        assert_eq!(observer.children.len(), expected_children.len());
        for (actual, expected) in observer.children.iter().zip(expected_children) {
            for key in ["node", "parent", "network", "threshold"] {
                assert_eq!(actual[key], expected[key], "{actual:?} {expected:?}");
            }
            if actual["operator"] == "E" {
                // The sum of nonnegative route quantities is nonnegative. Its
                // zero upper child is exactly equality0 in the generic row API.
                assert_eq!(actual["threshold"], 0.);
                assert_eq!(expected["operator"], "L");
            } else {
                assert_eq!(actual["operator"], expected["operator"]);
            }
        }
        let reference = control["evaluations"].as_array().unwrap();
        assert_eq!(observer.evaluations.len(), reference.len());
        for (rank, (actual, expected)) in observer.evaluations.iter().zip(reference).enumerate() {
            assert_eq!(actual["node"], expected["node"]);
            assert_eq!(actual["rank"], rank + 1);
            if expected["source_infeasible"] == true {
                // Coverage3 cannot fit one remaining route covering at most2.
                // The reference keeps a finite artificial-penalty bound here;
                // BALDES independently proves original-model infeasibility.
                assert_eq!(actual["status"], "Infeasible");
                assert_eq!(actual["bound_infinite"], true);
            } else {
                assert_eq!(actual["status"], "Optimal");
                let bound = actual["selected_bound"].as_f64().unwrap();
                let numerator = expected["objective_exact"][0].as_f64().unwrap();
                let denominator = expected["objective_exact"][1].as_f64().unwrap();
                let product_upper =
                    -crate::core::rounding::multiply_down(-bound, denominator).unwrap();
                assert!(
                    product_upper <= numerator,
                    "certificate exceeds exact prefix LP: {actual:?}"
                );
                assert!(
                    (actual["numerical_bound"].as_f64().unwrap()
                        - expected["reference_bound"].as_f64().unwrap())
                    .abs()
                        < 1e-8
                );
                assert!(
                    (actual["lp"].as_f64().unwrap()
                        - expected["reference_bound"].as_f64().unwrap())
                    .abs()
                        < 1e-8,
                    "{actual:?} {expected:?}"
                );
            }
        }
        traces.push(json!({"case":control["name"],"children":observer.children,"evaluations":observer.evaluations}));
        drop(observer);
        assert_eq!(result.processed_nodes, reference.len());
        assert_eq!(result.incumbent, control["incumbent"].as_f64().unwrap());
    }
    if let Ok(path) = std::env::var("BALDES_REFERENCE_SOURCE_TRACE") {
        std::fs::write(path, serde_json::to_string_pretty(&traces).unwrap()).unwrap();
    }
}

#[test]
fn observed_default_strong_multiplicity_selection_preserves_registered_network_order() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/metasolver_strong_identity.json"
    ))
    .unwrap();
    assert_eq!(
        corpus["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let phases: serde_json::Value = serde_json::from_str(include_str!(
        "../../../fixtures/metasolver_strong_phases.json"
    ))
    .unwrap();
    for case in corpus["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(phases["cases"].as_array().unwrap())
    {
        let problem = format::Model::from_json(&case["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let mut options = strong_tree_options();
        options.node.cg.stabilize = false;
        options.branching_registry.branching = NetworkBranchingPolicy::Phased;
        let root = solve_with_master_variables(problem, vec![], options.node.clone()).unwrap();
        let proposals = super::super::proposals::collect(&root, &options, &[]).unwrap();
        let mut history = model_strong::History::default();
        let selected = model_strong::choose_proposals(
            &root,
            &options,
            &mut history,
            f64::INFINITY,
            Instant::now() + std::time::Duration::from_secs(10),
            proposals,
        )
        .unwrap()
        .unwrap();
        let child = &selected.children[1];
        let row = child.state.shared_rows.len() - 1;
        let network = child
            .state
            .oracles
            .iter()
            .find(|o| o.network.route_entries.contains(&(row, 1.)))
            .unwrap()
            .network
            .id;
        assert_eq!(
            network as i64,
            case["selected_network"].as_i64().unwrap(),
            "{}",
            case["name"]
        );
        for (side, operator) in [(0, "L"), (1, "G")] {
            let child = &selected.children[side];
            assert_eq!(
                child.state.shared_rows.last().unwrap().rhs,
                case["selected_bounds"][operator].as_f64().unwrap()
            );
        }
        let problem = format::Model::from_json(&case["model"].to_string())
            .unwrap()
            .problem::<2>()
            .unwrap();
        let tree = crate::master::tree::solve(problem, options).unwrap();
        assert_eq!(tree.status, TreeStatus::Optimal);
        assert_eq!(
            tree.incumbent,
            case["reference_incumbent"].as_f64().unwrap()
        );
        assert_eq!(
            tree.processed_nodes as u64,
            case["reference_processed_nodes"].as_u64().unwrap()
        );
        assert_eq!(
            tree.strong_branching_trials as u64,
            case["reference_children_count"].as_u64().unwrap()
        );
    }
}

/// Strong trials are opt-in (native `ApplyStrongBranchingEvaluation = false`);
/// these tests exercise them with the default phase schedule.
fn strong_tree_options() -> ModelTreeOptions {
    ModelTreeOptions {
        branching_registry: {
            let mut __branch_registry = crate::master::model::BranchingRegistry::default();
            __branch_registry.strong_branching =
                Some(crate::master::tree::StrongBranchingOptions::default());
            __branch_registry
        },
        ..ModelTreeOptions::default()
    }
}
