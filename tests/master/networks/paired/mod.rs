//! Registered paired resources, controls, and independent path oracles.
use super::*;

mod cycles;
mod enumeration;
mod objectives;
mod registration;
mod rows;
mod witnesses;

fn check_registered_cases(source: &str) -> usize {
    let corpus: serde_json::Value = serde_json::from_str(source).unwrap();
    assert_eq!(
        corpus["reference_sha256"],
        "103caf4a9c58d6141e935848294bff16677f400323466dc23cd4b9e5c15889d6"
    );
    let mut compared = 0;
    for case in corpus["cases"].as_array().unwrap() {
        if let Some(waiting) = case["registration"]["functions"]["waiting"].as_f64() {
            for arc in case["model"]["networks"][0]["coupled_resources"][0]["arcs"]
                .as_array()
                .unwrap()
            {
                assert_eq!(arc["waiting"].as_i64().unwrap(), waiting.trunc() as i64);
            }
        }
        let expected = case["independent_objective"].as_f64().unwrap();
        if case["reference"]["outcome"] == "reference_reconstruction_defect" {
            assert_eq!(case["prepared"]["mode"], 4);
            assert_eq!(case["reference"]["exit_code"], 1);
            let problem = model::format::Model::from_json(&case["model"].to_string())
                .unwrap()
                .problem::<2>()
                .unwrap();
            let proof = problem.networks[0]
                .graph
                .coupled_kernel()
                .unwrap()
                .certify_front_program(&[1, 2], &mut 1_000_000, None)
                .unwrap_or_else(|| panic!("unknown certificate {}", case["name"]));
            assert!(proof.feasible);
            assert_eq!(proof.objective, [0., 0.]);
        } else {
            assert_eq!(
                expected,
                case["reference"]["bcRecRootLpVal"].as_f64().unwrap()
            );
        }
        for mode in [BoundMode::Numerical, BoundMode::Certified] {
            for direction in [0., 1., 2.] {
                for singular in [false, true] {
                    let mut source = case["model"].clone();
                    if singular {
                        source["networks"] = source["networks"][0].clone();
                    }
                    let problem = model::format::Model::from_json(&source.to_string())
                        .unwrap()
                        .problem::<2>()
                        .unwrap();
                    let mut options = Options::default();
                    options.cg.stabilize = false;
                    options.cg.bound_mode = mode;
                    options.pricing_options =
                        vec![("BALDESuseBidirectionalSearch".into(), direction)];
                    let output = model::solve(problem, options).unwrap();
                    assert_eq!(
                        output.master.status,
                        Status::Optimal,
                        "{} mode={mode:?} direction={direction} {:#?}",
                        case["name"],
                        output.master
                    );
                    assert_eq!(output.master.lp, expected, "{}", case["name"]);
                    assert!(
                        output.master.bound <= expected && expected - output.master.bound < 1e-7
                    );
                    compared += 1;
                }
            }
        }
    }
    compared
}

fn control_point_source_model(mut source: serde_json::Value) -> serde_json::Value {
    fn knots(function: &serde_json::Value) -> serde_json::Value {
        let pairs = function["pairs"].as_array().unwrap();
        let (mut y, mut x) = (pairs[0][0].as_f64().unwrap(), pairs[0][1].as_f64().unwrap());
        let mut points = vec![serde_json::json!([x, y])];
        for pair in &pairs[1..] {
            let end = pair[1].as_f64().unwrap();
            y += pair[0].as_f64().unwrap() * (end - x);
            x = end;
            points.push(serde_json::json!([x, y]));
        }
        serde_json::json!({"points":points})
    }
    for resource in source["coupled_resources"].as_array_mut().unwrap() {
        resource["initial"] = knots(&resource["initial"]);
        for arc in resource["arcs"].as_array_mut().unwrap() {
            for key in ["arrival", "increment", "lower", "upper"] {
                arc[key] = knots(&arc[key]);
            }
        }
        resource["kind"] = "paired_piecewise_linear".into();
        resource.as_object_mut().unwrap().remove("policy");
    }
    source
}
