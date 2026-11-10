use baldes::inputs::{Instance, Rounding};

const EUC: &str = "NAME : tiny-k2
COMMENT : (test, Optimal value: 0)
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE :\tEUC_2D\t
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 3 4
3\t1\t1
4 6 8
DEMAND_SECTION
1 4
2 0
3 5
4 4
DEPOT_SECTION
 2
 -1
EOF
";

#[test]
fn euc_2d_moves_the_depot_first_and_rounds_to_nearest() {
    let instance = Instance::parse(EUC).unwrap();
    assert_eq!(instance.name, "tiny-k2");
    assert_eq!(instance.rounding, Rounding::Nearest);
    assert!(instance.distances.is_none());
    assert_eq!(
        instance.vehicles, 4,
        "DIMENSION bounds the fleet without VEHICLES"
    );
    let order: Vec<_> = instance
        .customers
        .iter()
        .map(|c| (c.x, c.y, c.demand))
        .collect();
    assert_eq!(
        order,
        [(3., 4., 0.), (0., 0., 4.), (1., 1., 5.), (6., 8., 4.)]
    );
    assert!(!instance.has_time_windows());
    // Depot (3, 4) to (1, 1): sqrt(13) = 3.606.
    assert_eq!(instance.distance(0, 2, Rounding::Nearest), 4.);
    assert_eq!(instance.distance(0, 2, Rounding::Ceil), 4.);
    assert_eq!(instance.distance(0, 2, Rounding::Tenth), 3.6);
    assert!(
        instance.graph(8, Rounding::Nearest).is_err(),
        "no time windows for VRPTW"
    );
    instance.capacity_graph(8, Rounding::Nearest).unwrap();
}

#[test]
fn explicit_weights_in_every_triangle_format() {
    // Symmetric distances of nodes 1..4 (depot 1).
    let full = [
        [0., 2., 3., 4.],
        [2., 0., 5., 6.],
        [3., 5., 0., 7.],
        [4., 6., 7., 0.],
    ];
    let section = |format: &str| -> String {
        let mut weights = vec![];
        for i in 0..4 {
            for j in 0..4 {
                let keep = match format {
                    "FULL_MATRIX" => true,
                    "LOWER_ROW" => j < i,
                    "LOWER_DIAG_ROW" => j <= i,
                    "UPPER_ROW" => j > i,
                    _ => j >= i,
                };
                if keep {
                    weights.push(full[i][j].to_string());
                }
            }
        }
        weights.join(" ")
    };
    for format in [
        "FULL_MATRIX",
        "LOWER_ROW",
        "LOWER_DIAG_ROW",
        "UPPER_ROW",
        "UPPER_DIAG_ROW",
    ] {
        let text = format!(
            "NAME : explicit\nTYPE : CVRP\nDIMENSION : 4\nVEHICLES : 2\nCAPACITY : 10\n\
             EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : {format}\nEDGE_WEIGHT_SECTION\n{}\n\
             DEMAND_SECTION\n1 0\n2 3\n3 3\n4 3\nDEPOT_SECTION\n1\n-1\nEOF\n",
            section(format)
        );
        let instance = Instance::parse(&text).unwrap();
        assert_eq!(instance.vehicles, 2);
        let matrix = instance.distances.as_ref().unwrap();
        assert_eq!(matrix, &full.map(|row| row.to_vec()).to_vec(), "{format}");
        // Explicit weights ignore the rounding and follow truncation.
        assert_eq!(instance.distance(1, 3, Rounding::Tenth), 6.);
        let mut truncated = instance.clone();
        truncated.truncate(2).unwrap();
        assert_eq!(
            truncated.distances.unwrap(),
            vec![vec![0., 2., 3.], vec![2., 0., 5.], vec![3., 5., 0.]]
        );
    }
}

#[test]
fn unsupported_cvrplib_features_are_rejected() {
    let with = |line: &str| EUC.replacen("CAPACITY : 10", &format!("CAPACITY : 10\n{line}"), 1);
    for (text, message) in [
        (with("DISTANCE : 100"), "DISTANCE"),
        (with("SERVICE_TIME : 10"), "SERVICE_TIME"),
        (EUC.replace("TYPE : CVRP", "TYPE : ATSP"), "TYPE"),
        (EUC.replace("EUC_2D", "GEO"), "EDGE_WEIGHT_TYPE"),
        (EUC.replace("2 0\n", "2 1\n"), "depot demand"),
        (
            EUC.replace("4 6 8\n", ""),
            "NODE_COORD_SECTION needs 4 rows",
        ),
    ] {
        let error = Instance::parse(&text).unwrap_err().to_string();
        assert!(error.contains(message), "{message}: {error}");
    }
}

#[test]
fn solomon_input_is_unchanged() {
    let instance = Instance::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/instances/Solomon/R101.txt"
    ))
    .unwrap();
    assert_eq!(instance.rounding, Rounding::None);
    assert!(instance.distances.is_none() && instance.has_time_windows());
}

#[cfg(feature = "clp")]
#[test]
fn cli_solves_a_cvrplib_root() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_baldes"))
        .args([
            "solve",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/instances/cvrplib/E-n22-k4.vrp"
            ),
        ])
        .args([
            "--capacity-cuts",
            "--no-enumeration",
            "--rounding",
            "nearest",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lp: f64 = stdout
        .split_whitespace()
        .find_map(|t| t.strip_prefix("lp="))
        .unwrap()
        .parse()
        .unwrap();
    // E-n22-k4: optimal 375 under TSPLIB nint distances; capacity cuts close the root.
    assert!((lp - 375.).abs() < 1e-6, "lp {lp}");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_baldes"))
        .args([
            "vrptw",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/instances/cvrplib/E-n22-k4.vrp"
            ),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command"));
}

#[test]
fn tenth_rounding_uses_integer_times_and_tenth_costs() {
    let instance = Instance::parse(
        "tiny\nVEHICLE\nNUMBER CAPACITY\n2 10\nCUSTOMER\nCUST NO. X Y DEMAND READY DUE SERVICE\n0 0 0 0 0 100 0\n1 1 1 1 2 50 3\n2 2 0 1 0 60 0\n",
    )
    .unwrap();
    let graph = instance.graph(4, Rounding::Tenth).unwrap();
    for arc in &graph.arcs {
        let time = arc.consumption[0];
        assert_eq!(time, time.round(), "integer travel time {time}");
        // Cost: the HGS distance in tenths; time adds the scaled service.
        let tail = if arc.tail == 3 { 0 } else { arc.tail };
        let service = instance.customers[tail].service * 10.;
        assert!(
            (arc.cost * 10. - (time - service)).abs() < 1e-9,
            "{} vs {}",
            arc.cost,
            time
        );
    }
    // sqrt(2) = 1.414... -> 14 tenths.
    let arc = graph
        .arcs
        .iter()
        .find(|a| a.tail == 0 && a.head == 1)
        .unwrap();
    assert_eq!((arc.consumption[0], arc.cost), (14., 1.4));
    assert_eq!(graph.vertices[1].bounds[0].upper, 500.);
}

fn time_window_input() -> String {
    EUC.replace("NAME : tiny-k2", "NAME : XTW-n4")
        .replace("TYPE : CVRP", "TYPE : VRPTW")
        .replace("DEPOT_SECTION", "TIME_WINDOW_SECTION\n4 0 70\n2 0 100\n1 10 90\n3 20 80\nSERVICE_TIME_SECTION\n3 3\n1 2\n4 4\n2 0\nDEPOT_SECTION")
}

#[test]
fn native_vrptw_preserves_windows_services_and_depot_permutation() {
    let instance = Instance::parse(&time_window_input()).unwrap();
    assert!(instance.is_xtw() && instance.has_time_windows());
    assert_eq!(instance.rounding, Rounding::Nearest);
    assert_eq!(
        instance.vehicles, 3,
        "one available vehicle per customer for a free fleet"
    );
    let resources: Vec<_> = instance
        .customers
        .iter()
        .map(|c| (c.ready, c.due, c.service))
        .collect();
    assert_eq!(
        resources,
        [
            (0., 100., 0.),
            (10., 90., 2.),
            (20., 80., 3.),
            (0., 70., 4.)
        ]
    );
    assert_eq!(
        baldes::extensions::hgs::solution::validate_plan(&instance, &[vec![1, 2], vec![3]])
            .unwrap(),
        20.
    );
    let mut other = instance.clone();
    other.name = "other".into();
    assert!(!other.is_xtw());
    assert!(!Instance::parse(&EUC.replace("tiny-k2", "XTW-n4"))
        .unwrap()
        .is_xtw());
}

#[test]
fn vrptw_rejects_missing_invalid_or_misidentified_resource_rows() {
    let text = time_window_input();
    for invalid in [
        EUC.replace("TYPE : CVRP", "TYPE : VRPTW"),
        text.replace("3 20 80", "3 81 80"),
        text.replace("3 20 80", "3 NaN 80"),
        text.replace("3 20 80", "3 20 inf"),
        text.replace("3 20 80", "3 20"),
        text.replace("3 20 80", "1 20 80"),
        text.replace("3 20 80", "NaN 20 80"),
        text.replace("3 3\n", "3 -1\n"),
        text.replace("3 3\n", "3 inf\n"),
        text.replace("TYPE : VRPTW", "TYPE : CVRP"),
    ] {
        assert!(
            Instance::parse(&invalid).is_err(),
            "accepted invalid VRPTW: {invalid}"
        );
    }
    let without_service = text.replace("SERVICE_TIME_SECTION\n3 3\n1 2\n4 4\n2 0\n", "");
    let instance = Instance::parse(&without_service).unwrap();
    assert!(instance.customers.iter().all(|c| c.service == 0.));
}

#[test]
fn explicit_vrptw_distances_and_fixed_fleet_are_preserved() {
    let text = time_window_input()
        .replace("CAPACITY : 10", "CAPACITY : 10\nVEHICLES : 2")
        .replace(
            "EDGE_WEIGHT_TYPE :\tEUC_2D",
            "EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX",
        )
        .replace(
            "NODE_COORD_SECTION",
            "EDGE_WEIGHT_SECTION\n0 1 2 3\n4 0 5 6\n7 8 0 9\n10 11 12 0\nNODE_COORD_SECTION",
        );
    let instance = Instance::parse(&text).unwrap();
    assert_eq!(instance.vehicles, 2);
    assert_eq!(instance.distance(0, 1, Rounding::Nearest), 4.);
    assert_eq!(instance.distance(1, 0, Rounding::Nearest), 1.);
    assert_eq!(instance.customers[1].service, 2.);
}
