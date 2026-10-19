use super::*;

#[derive(Clone, Debug, Default)]
pub struct KnapsackSeparator {
    pub parameters: KnapsackParameters,
}

impl KnapsackSeparator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parameters(parameters: KnapsackParameters) -> Result<Self> {
        validate_parameters(&parameters)?;
        Ok(Self { parameters })
    }

    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<()> {
        if !value.is_finite() {
            return Err(invalid("knapsack parameter must be finite"));
        }
        match name {
            "max_cuts_per_round" => {
                if !value.is_finite() || value < 1.0 || value > i32::MAX as f64 {
                    return Err(invalid("knapsack max_cuts_per_round must be at least 1"));
                }
                self.parameters.max_cuts_per_round = value as usize;
            }
            "use_rounding" => {
                self.parameters.use_rounding = value != 0.0;
            }
            "max_one_k" => {
                if !value.is_finite() || value < 0.0 || value > i32::MAX as f64 {
                    return Err(invalid("knapsack max_one_k must be non-negative"));
                }
                self.parameters.max_one_k = value as i32;
            }
            "violation_tolerance" => {
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid("knapsack violation_tolerance must be non-negative"));
                }
                self.parameters.violation_tolerance = value;
            }
            _ => return Err(invalid(format!("unknown knapsack parameter: {name}"))),
        }
        Ok(())
    }

    pub fn get_parameter(&self, name: &str) -> Result<f64> {
        match name {
            "max_cuts_per_round" => Ok(self.parameters.max_cuts_per_round as f64),
            "use_rounding" => Ok(if self.parameters.use_rounding {
                1.0
            } else {
                0.0
            }),
            "max_one_k" => Ok(self.parameters.max_one_k as f64),
            "violation_tolerance" => Ok(self.parameters.violation_tolerance),
            _ => Err(invalid(format!("unknown knapsack parameter: {name}"))),
        }
    }

    pub fn configure(&mut self, options: &Options) -> Result<()> {
        self.parameters = knapsack_parameters(options, self.parameters.clone())?;
        Ok(())
    }

    pub fn separate(
        &self,
        loads: &[i32],
        values: &[f64],
        rhs: i32,
    ) -> Result<Vec<RouteLoadKnapsackCut>> {
        if loads.len() != values.len() {
            return Err(invalid("loads and values must have the same length"));
        }
        if rhs < 0 || rhs > i32::MAX / 8 {
            return Err(invalid("knapsack rhs is out of range"));
        }
        validate_parameters(&self.parameters)?;
        for &value in values {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid("knapsack values must be finite and non-negative"));
            }
        }
        for &load in loads {
            if load < 0 || load > i32::MAX / 8 {
                return Err(invalid(
                    "knapsack loads must be non-negative and within range",
                ));
            }
        }

        let mut totals: BTreeMap<i32, f64> = BTreeMap::new();
        for (&load, &val) in loads.iter().zip(values.iter()) {
            *totals.entry(load).or_insert(0.0) += (val * 1e8 + 0.5).floor() / 1e8;
        }

        let mut result: Vec<_> = self
            .separate_totals(&[(totals, rhs)])
            .into_iter()
            .map(|(_, cut)| cut)
            .collect();
        // The standalone adapter reports LHS in original column order,
        // independently of the load aggregation used by separation.
        for cut in &mut result {
            cut.lhs = 0.;
            for (&load, &value) in loads.iter().zip(values) {
                cut.lhs += ((value * 1e8 + 0.5).floor() / 1e8) * cut.coefficient(load) as f64;
            }
            cut.violation = cut.lhs - cut.rhs as f64;
        }
        Ok(result)
    }

    // The original bounded candidate tree is shared by all original rows.
    pub(super) fn separate_totals(
        &self,
        rows: &[(BTreeMap<i32, f64>, i32)],
    ) -> Vec<(usize, RouteLoadKnapsackCut)> {
        let mut candidates = CandidateTree::default();
        for (row, (totals, rhs)) in rows.iter().enumerate() {
            if self.parameters.use_rounding {
                self.separation_by_rounding(totals, *rhs, row, &mut candidates);
            }
            if self.parameters.max_one_k > 0 {
                self.separation_of_one_k_inequalities(totals, *rhs, row, &mut candidates);
            }
        }
        let ordered = candidates.into_vec();
        let mut result = Vec::with_capacity(ordered.len());
        for cand in ordered {
            let mut cut = RouteLoadKnapsackCut {
                kind: cand.kind,
                numerator: cand.numerator,
                denominator: cand.denominator,
                rhs: cand.rhs,
                lhs: 0.0,
                violation: 0.0,
                thresholds: cand.thresholds,
                coefficients: cand.coefficients,
            };
            for (&load, &value) in &rows[cand.row].0 {
                cut.lhs += value * cut.coefficient(load) as f64;
            }
            cut.violation = cut.lhs - (cut.rhs as f64);
            result.push((cand.row, cut));
        }
        result
    }

    fn separation_by_rounding(
        &self,
        totals: &BTreeMap<i32, f64>,
        rhs: i32,
        row: usize,
        candidates: &mut CandidateTree,
    ) {
        for &q in totals.keys() {
            if q >= 1 {
                for j in 1..=q {
                    let multiplier = (j as f64) / (q as f64);
                    let target = (multiplier * (rhs as f64) + 1e-6).floor() as i32;
                    let mut lhs = 0.0;
                    for (&load, &val) in totals {
                        lhs += ((load as f64) * multiplier + 1e-6).floor() * val;
                    }
                    if (target as f64) * (lhs - target as f64) > self.parameters.violation_tolerance
                    {
                        let mut coefficients = BTreeMap::new();
                        coefficients.insert(0, 0);
                        for k in 1..=target {
                            let step = ((k as f64) / multiplier - 1e-6).ceil() as i32;
                            coefficients.entry(step).or_insert(k);
                        }
                        let g = gcd(j, q);
                        let score = -(lhs - target as f64) / (target as f64);
                        candidates.insert(Candidate {
                            row,
                            score,
                            kind: KnapsackKind::Rounding,
                            numerator: j / g,
                            denominator: q / g,
                            rhs: target,
                            multiplier,
                            thresholds: Vec::new(),
                            breakpoints: Vec::new(),
                            coefficients,
                        });
                        if candidates.len() > self.parameters.max_cuts_per_round {
                            candidates.pop_last();
                        }
                    }
                }
            }
        }
    }

    fn separation_of_one_k_inequalities(
        &self,
        totals: &BTreeMap<i32, f64>,
        rhs: i32,
        row: usize,
        candidates: &mut CandidateTree,
    ) {
        if totals.is_empty() {
            return;
        }
        let mut unique_vals = BTreeSet::new();
        for &load in totals.keys() {
            let folded = if (rhs + 1) / 2 < load {
                rhs + 1 - load
            } else {
                load
            };
            unique_vals.insert(folded);
        }
        let vals: Vec<i32> = unique_vals.into_iter().collect();

        let mut emit = |low: &[i32]| {
            let cut_rhs = (low.len() * 2) as i32;
            let mut breakpoints = Vec::with_capacity(cut_rhs as usize + 1);
            breakpoints.push(0);
            breakpoints.extend_from_slice(low);
            for &v in low.iter().rev() {
                breakpoints.push(rhs + 1 - v);
            }

            let mut count = 0usize;
            let mut lhs = 0.0;
            for (&load, &val) in totals {
                while count < cut_rhs as usize && breakpoints[count + 1] <= load {
                    count += 1;
                }
                lhs += (count as f64) * val;
            }
            if self.parameters.violation_tolerance + (cut_rhs as f64) < lhs {
                let mut coefficients = BTreeMap::new();
                coefficients.insert(0, 0);
                for i in 1..=(cut_rhs as usize) {
                    if i == cut_rhs as usize || breakpoints[i] < breakpoints[i + 1] {
                        coefficients.entry(breakpoints[i]).or_insert(i as i32);
                    }
                }
                let thresholds = breakpoints[1..].to_vec();
                let score = ((cut_rhs as f64) - lhs) / (cut_rhs as f64);
                candidates.insert(Candidate {
                    row,
                    score,
                    kind: KnapsackKind::Steps,
                    numerator: 1,
                    denominator: 1,
                    rhs: cut_rhs,
                    multiplier: 0.0,
                    thresholds,
                    breakpoints,
                    coefficients,
                });
                if candidates.len() > self.parameters.max_cuts_per_round {
                    candidates.pop_last();
                }
            }
        };

        let size = vals.len();
        for i in 0..size {
            for j in i..size {
                if vals[j] > 2 * vals[i] {
                    break;
                }
                for k in j..size {
                    let (a, b, c) = (vals[i], vals[j], vals[k]);
                    if c <= a + b && rhs < a + 2 * c && rhs < c + 2 * b {
                        emit(&[a, b, c]);
                    }
                }
            }
        }
        if self.parameters.max_one_k > 7 {
            for i in 0..size {
                for j in i..size {
                    if vals[j] > 2 * vals[i] {
                        break;
                    }
                    for k in j..size {
                        let (a, b, c) = (vals[i], vals[j], vals[k]);
                        if c > a + b || !(rhs < 3 * c) {
                            continue;
                        }
                        for l in k..size {
                            let d = vals[l];
                            if d <= a + c && d <= 2 * b && rhs < a + 2 * d && rhs < b + c + d {
                                emit(&[a, b, c, d]);
                            }
                        }
                    }
                }
            }
        }
        if self.parameters.max_one_k > 9 {
            for i in 0..size {
                for j in i..size {
                    if vals[j] > 2 * vals[i] {
                        break;
                    }
                    for k in j..size {
                        let (a, b, c) = (vals[i], vals[j], vals[k]);
                        if c > a + b {
                            continue;
                        }
                        for l in k..size {
                            let d = vals[l];
                            if d > a + c || d > 2 * b || !(rhs < c + 2 * d) {
                                continue;
                            }
                            for m in l..size {
                                let e = vals[m];
                                if e <= a + d
                                    && e <= b + c
                                    && rhs < a + 2 * e
                                    && rhs < b + d + e
                                    && rhs < 2 * c + e
                                {
                                    emit(&[a, b, c, d, e]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
