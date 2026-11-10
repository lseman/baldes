//! Replay every downloaded XTW solution directly against its original input.
use baldes::extensions::hgs::solution::{parse_solution, validate_plan};
use baldes::inputs::{Instance, Rounding};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "instances/XTW".into());
    let mut paths: Vec<_> = std::fs::read_dir(directory)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.retain(|p| p.extension().is_some_and(|e| e == "vrp"));
    paths.sort();
    if paths.is_empty() {
        return Err("no XTW input files found".into());
    }
    for path in &paths {
        let instance = Instance::read(path).map_err(|e| e.to_string())?;
        assert!(instance.is_xtw() && instance.rounding == Rounding::Nearest);
        let text = std::fs::read_to_string(path.with_extension("sol"))?;
        let routes = parse_solution(&text)?;
        let cost = validate_plan(&instance, &routes)?;
        let declared: f64 = text
            .lines()
            .find_map(|line| line.strip_prefix("Cost "))
            .ok_or("solution has no Cost")?
            .trim()
            .parse()?;
        assert_eq!(cost, declared, "{}", instance.name);
        println!(
            "{},customers={},routes={},cost={cost:.0},validated=true",
            instance.name,
            instance.customers.len() - 1,
            routes.len()
        );
    }
    println!(
        "Validated {} native XTW instances and solutions",
        paths.len()
    );
    Ok(())
}
