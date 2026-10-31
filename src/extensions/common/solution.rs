//! CVRPLIB route-file parsing and formatting.
/// Read `Route #1: 1 2 ...` records. Cost metadata is deliberately ignored:
/// objective values must be recomputed using the selected instance metric.
pub fn parse_solution(text: &str) -> Result<Vec<Vec<usize>>, String> {
    let mut routes = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.to_ascii_lowercase().starts_with("route ") {
            continue;
        }
        let (_, stops) = line.split_once(':').ok_or("route record is missing ':'")?;
        let route = stops
            .split_whitespace()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| format!("invalid customer index: {s}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if route.is_empty() {
            return Err("empty route record".into());
        }
        routes.push(route);
    }
    if routes.is_empty() {
        return Err("no Route records found in solution".into());
    }
    Ok(routes)
}

/// Serialize a validated plan in CVRPLib format (internal customer indices).
pub fn format_solution(routes: &[Vec<usize>], cost: f64) -> String {
    use std::fmt::Write;
    let mut text = String::new();
    for (i, route) in routes.iter().filter(|r| !r.is_empty()).enumerate() {
        let _ = write!(text, "Route #{}:", i + 1);
        for customer in route {
            let _ = write!(text, " {customer}");
        }
        text.push('\n');
    }
    let _ = writeln!(text, "Cost {cost:.10}");
    text
}
