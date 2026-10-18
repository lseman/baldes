//! Solomon and CVRPLIB input readers.
use super::*;
impl Instance {
    pub fn parse(text: &str) -> Result<Self> {
        let tsplib = text.lines().map(str::trim).any(|line| {
            let key = line
                .split(':')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_uppercase();
            line.contains(':')
                && matches!(key.as_str(), "DIMENSION" | "EDGE_WEIGHT_TYPE" | "CAPACITY")
        });
        if tsplib {
            Self::parse_cvrplib(text)
        } else {
            Self::parse_solomon(text)
        }
    }
    fn parse_solomon(text: &str) -> Result<Self> {
        let lines: Vec<_> = text
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let name = lines
            .first()
            .ok_or_else(|| invalid("empty Solomon instance"))?
            .to_string();
        let vehicle = lines
            .iter()
            .position(|s| s.to_ascii_uppercase().starts_with("NUMBER"))
            .and_then(|i| lines.get(i + 1))
            .ok_or_else(|| invalid("Solomon vehicle table is missing"))?;
        let numbers = vehicle
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| invalid("invalid Solomon vehicle row"))?;
        if numbers.len() < 2
            || numbers[0] < 1.
            || numbers[0].fract() != 0.
            || numbers[1] <= 0.
            || !numbers[1].is_finite()
        {
            return Err(invalid("invalid Solomon vehicle count/capacity"));
        }
        let start = lines
            .iter()
            .position(|s| s.to_ascii_uppercase().starts_with("CUST NO"))
            .ok_or_else(|| invalid("Solomon customer table is missing"))?
            + 1;
        let mut customers = vec![];
        for row in &lines[start..] {
            let values = row
                .split_whitespace()
                .map(str::parse::<f64>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| invalid("invalid Solomon customer row"))?;
            if values.len() < 7 {
                continue;
            }
            if values.iter().any(|v| !v.is_finite())
                || values[0] != customers.len() as f64
                || values[3] < 0.
                || values[4] > values[5]
                || values[6] < 0.
            {
                return Err(invalid(
                    "invalid Solomon customer identity or resource data",
                ));
            }
            customers.push(Customer {
                x: values[1],
                y: values[2],
                demand: values[3],
                ready: values[4],
                due: values[5],
                service: values[6],
            });
        }
        if customers.len() < 2 {
            return Err(invalid("Solomon instance needs a depot and customers"));
        }
        Ok(Self {
            name,
            vehicles: numbers[0] as usize,
            capacity: numbers[1],
            customers,
            rounding: Rounding::None,
            distances: None,
        })
    }
    /// CVRPLIB (TSPLIB) CVRP: DIMENSION, CAPACITY, EDGE_WEIGHT_TYPE EUC_2D,
    /// CEIL_2D or EXPLICIT (FULL_MATRIX, LOWER_ROW, LOWER_DIAG_ROW, UPPER_ROW,
    /// UPPER_DIAG_ROW), NODE_COORD/DISPLAY_DATA, DEMAND and DEPOT sections.
    /// The depot becomes customer 0, the others keep their order. `vehicles`
    /// is VEHICLES when given, otherwise DIMENSION, as in the converted
    /// Solomon-format files. Route-length limits (DISTANCE) and service times
    /// are not supported.
    fn parse_cvrplib(text: &str) -> Result<Self> {
        let mut headers = std::collections::HashMap::new();
        let mut sections: std::collections::HashMap<String, Vec<f64>> = Default::default();
        let mut current: Option<String> = None;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let first = line
                .split(|c: char| c.is_whitespace() || c == ':')
                .next()
                .unwrap_or("");
            let keyword = first.to_ascii_uppercase();
            if keyword == "EOF" {
                break;
            }
            if keyword.ends_with("_SECTION") {
                current = Some(keyword.clone());
                sections.entry(keyword).or_default();
                continue;
            }
            if first
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            {
                let (key, value) = line
                    .split_once(':')
                    .ok_or_else(|| invalid(format!("CVRPLIB: unexpected line {line:?}")))?;
                headers.insert(key.trim().to_ascii_uppercase(), value.trim().to_string());
                current = None;
                continue;
            }
            let section = current
                .as_ref()
                .ok_or_else(|| invalid(format!("CVRPLIB: data outside a section: {line:?}")))?;
            for token in line.split_whitespace() {
                let value = token
                    .parse::<f64>()
                    .map_err(|_| invalid(format!("CVRPLIB: invalid number {token:?}")))?;
                sections.get_mut(section).unwrap().push(value);
            }
        }
        let header = |key: &str| headers.get(key).map(String::as_str);
        let number = |key: &str| -> Result<f64> {
            header(key)
                .ok_or_else(|| invalid(format!("CVRPLIB: {key} is missing")))?
                .parse::<f64>()
                .map_err(|_| invalid(format!("CVRPLIB: invalid {key}")))
        };
        if let Some(kind) = header("TYPE") {
            if !kind.eq_ignore_ascii_case("CVRP") {
                return Err(invalid(format!(
                    "CVRPLIB: TYPE {kind} is not supported, only CVRP"
                )));
            }
        }
        for key in ["DISTANCE", "SERVICE_TIME"] {
            if headers.contains_key(key) {
                return Err(invalid(format!("CVRPLIB: {key} is not supported")));
            }
        }
        let dimension = number("DIMENSION")?;
        let capacity = number("CAPACITY")?;
        if dimension < 2. || dimension.fract() != 0. || !(capacity > 0. && capacity.is_finite()) {
            return Err(invalid("CVRPLIB: invalid DIMENSION or CAPACITY"));
        }
        let n = dimension as usize;
        let vehicles = match header("VEHICLES") {
            Some(_) => number("VEHICLES")? as usize,
            None => n,
        };
        // (id, values...) rows of a section, by 1-based node id.
        let rows = |name: &str, width: usize| -> Result<Option<Vec<Vec<f64>>>> {
            let Some(values) = sections.get(name) else {
                return Ok(None);
            };
            if values.len() != n * (width + 1) {
                return Err(invalid(format!("CVRPLIB: {name} needs {n} rows")));
            }
            let mut table = vec![vec![]; n];
            for row in values.chunks(width + 1) {
                let id = row[0];
                if id < 1.
                    || id > n as f64
                    || id.fract() != 0.
                    || !table[id as usize - 1].is_empty()
                {
                    return Err(invalid(format!("CVRPLIB: invalid node id in {name}")));
                }
                table[id as usize - 1] = row[1..].to_vec();
            }
            Ok(Some(table))
        };
        let demands = rows("DEMAND_SECTION", 1)?
            .ok_or_else(|| invalid("CVRPLIB: DEMAND_SECTION is missing"))?;
        let coordinates = match rows("NODE_COORD_SECTION", 2)? {
            Some(c) => Some(c),
            None => rows("DISPLAY_DATA_SECTION", 2)?,
        };
        let depot = match sections.get("DEPOT_SECTION").and_then(|d| d.first()) {
            Some(&d) if d >= 1. && d <= n as f64 && d.fract() == 0. => d as usize - 1,
            Some(_) => return Err(invalid("CVRPLIB: invalid depot")),
            None => 0,
        };
        if sections
            .get("DEPOT_SECTION")
            .is_some_and(|d| d.iter().filter(|&&v| v >= 0.).count() > 1)
        {
            return Err(invalid("CVRPLIB: multiple depots are not supported"));
        }
        // Customer order: the depot, then the other nodes by id.
        let order: Vec<usize> = std::iter::once(depot)
            .chain((0..n).filter(|&i| i != depot))
            .collect();
        let weight_type = header("EDGE_WEIGHT_TYPE")
            .unwrap_or("EUC_2D")
            .to_ascii_uppercase();
        let (rounding, distances) = match weight_type.as_str() {
            "EUC_2D" => (Rounding::Nearest, None),
            "CEIL_2D" => (Rounding::Ceil, None),
            "EXPLICIT" => {
                let format = header("EDGE_WEIGHT_FORMAT")
                    .unwrap_or("FULL_MATRIX")
                    .to_ascii_uppercase();
                let weights = sections
                    .get("EDGE_WEIGHT_SECTION")
                    .ok_or_else(|| invalid("CVRPLIB: EDGE_WEIGHT_SECTION is missing"))?;
                let full = explicit_matrix(&format, weights, n)?;
                let matrix = order
                    .iter()
                    .map(|&i| order.iter().map(|&j| full[i][j]).collect())
                    .collect();
                (Rounding::None, Some(matrix))
            }
            other => {
                return Err(invalid(format!(
                    "CVRPLIB: EDGE_WEIGHT_TYPE {other} is not supported"
                )))
            }
        };
        if distances.is_none() && coordinates.is_none() {
            return Err(invalid("CVRPLIB: NODE_COORD_SECTION is missing"));
        }
        let mut customers = vec![];
        for &i in &order {
            let demand = demands[i][0];
            let (x, y) = coordinates
                .as_ref()
                .map_or((0., 0.), |c| (c[i][0], c[i][1]));
            if !(demand >= 0. && demand.is_finite() && x.is_finite() && y.is_finite()) {
                return Err(invalid("CVRPLIB: invalid demand or coordinates"));
            }
            customers.push(Customer {
                x,
                y,
                demand,
                ready: 0.,
                due: f64::INFINITY,
                service: 0.,
            });
        }
        if customers[0].demand != 0. {
            return Err(invalid("CVRPLIB: the depot demand must be zero"));
        }
        let name = header("NAME").unwrap_or("cvrplib").to_string();
        Ok(Self {
            name,
            vehicles,
            capacity,
            customers,
            rounding,
            distances,
        })
    }
    pub fn read(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).map_err(|e| invalid(format!("instance input: {e}")))?;
        Self::parse(&text)
    }
}
/// A TSPLIB EXPLICIT weight section as a full symmetric matrix, by node.
fn explicit_matrix(format: &str, weights: &[f64], n: usize) -> Result<Vec<Vec<f64>>> {
    let pairs: Vec<(usize, usize)> = match format {
        "FULL_MATRIX" => (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).collect(),
        "LOWER_ROW" => (0..n).flat_map(|i| (0..i).map(move |j| (i, j))).collect(),
        "LOWER_DIAG_ROW" => (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect(),
        "UPPER_ROW" => (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect(),
        "UPPER_DIAG_ROW" => (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect(),
        other => {
            return Err(invalid(format!(
                "CVRPLIB: EDGE_WEIGHT_FORMAT {other} is not supported"
            )))
        }
    };
    if weights.len() != pairs.len() {
        return Err(invalid(format!(
            "CVRPLIB: EDGE_WEIGHT_SECTION has {} weights, {format} needs {}",
            weights.len(),
            pairs.len()
        )));
    }
    let mut matrix = vec![vec![0.; n]; n];
    for (&(i, j), &w) in pairs.iter().zip(weights) {
        if !(w >= 0. && w.is_finite()) {
            return Err(invalid("CVRPLIB: invalid edge weight"));
        }
        matrix[i][j] = w;
        if format != "FULL_MATRIX" {
            matrix[j][i] = w;
        }
    }
    Ok(matrix)
}
