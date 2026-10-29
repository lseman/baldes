//! Fixed-array/vector composition with the machine helper's ordered scan.
use super::*;

fn append(
    output: &mut Vec<(f64, f64)>,
    coordinate: f64,
    value: f64,
    previous_value: f64,
    overwrite_slope: bool,
    work: &mut Work,
) -> Result<()> {
    // The fixed-array capacity is checked before the candidate is written,
    // including candidates subsequently merged into the preceding segment.
    if output.len() >= 12 {
        return Err(FrontierUnknown::PairLimit);
    }
    work.pairs(output.len() + 1)?;
    let previous = output.last().unwrap().1;
    let slope = checked(
        checked(value - previous_value, "machine composition secant value")?
            / checked(coordinate - previous, "machine composition secant span")?,
        "machine composition secant",
    )?;
    if output.len() > 1 && near(slope, output.last().unwrap().0)? {
        let last = output.last_mut().unwrap();
        if overwrite_slope {
            last.0 = slope;
        }
        last.1 = coordinate;
    } else {
        output.push((slope, coordinate));
    }
    Ok(())
}

pub(super) fn compose(
    outer: &PreparedSlopeFunction,
    inner: &PreparedSlopeFunction,
    lower: f64,
    upper: f64,
    work: &mut Work,
) -> Result<Option<PreparedSlopeFunction>> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(unsupported("composition needs a finite ordered interval"));
    }
    work.pairs(outer.pairs().len())?;
    work.pairs(inner.pairs().len())?;
    work.charge(outer.pairs().len() + inner.pairs().len())?;
    if outer.pairs().len() > 12 {
        return Err(FrontierUnknown::PairLimit);
    }
    let (start, end) = inner.domain();
    if (lower > end && !near(lower, end)?) || (upper < start && !near(upper, start)?) {
        return Ok(None);
    }
    if near(lower, upper)? {
        return short_composition(outer, inner, lower, upper, work).map(Some);
    }
    if inner.pairs().iter().skip(1).any(|pair| pair.0 < 0.) {
        return Err(unsupported("composition inner map decreases"));
    }
    let (mut inner_value, mut inner_index) = if near(lower, start)? {
        (inner.pairs()[0].0, 1)
    } else {
        inner
            .machine_vector_value_and_segment(lower)
            .map_err(FrontierUnknown::Arithmetic)?
            .unwrap_or((1e12, 0))
    };
    // Unlike the short branch, the long scan passes an inner lookup's
    // sentinel into the outer lookup. Only the outer result is rejected.
    let (mut outer_value, mut outer_index) = outer
        .machine_value_and_segment(inner_value)
        .map_err(FrontierUnknown::Arithmetic)?
        .ok_or_else(|| unsupported("machine composition outer lookup sentinel"))?;
    if outer_value == 1e12 {
        return Err(unsupported("machine composition outer sentinel"));
    }
    work.pairs(1)?;
    let mut output = vec![(outer_value, lower)];
    let mut coordinate = lower;
    let mut outer_coordinate = inner_value;
    let mut carried_value = outer_value;
    while inner_index < inner.pairs().len() {
        work.charge(1)?;
        let (slope, segment_end) = inner.pairs()[inner_index];
        let stop = if segment_end <= upper {
            segment_end
        } else {
            upper
        };
        let image = checked(
            checked(
                checked(stop - coordinate, "machine composition inner span")? * slope,
                "machine composition inner multiplication",
            )? + inner_value,
            "machine composition inner value",
        )?;
        let mut previous_value = carried_value;
        let mut exhausted = true;
        while outer_index < outer.pairs().len() {
            work.charge(1)?;
            let (outer_slope, knot) = outer.pairs()[outer_index];
            let crossing = if near(knot, image)? {
                stop
            } else if image < knot {
                exhausted = false;
                break;
            } else {
                checked(
                    checked(
                        checked(knot - inner_value, "machine composition crossing value")? / slope,
                        "machine composition crossing division",
                    )? + coordinate,
                    "machine composition crossing",
                )?
            };
            outer_value = checked(
                outer_value
                    + checked(
                        checked(knot - outer_coordinate, "machine composition outer span")?
                            * outer_slope,
                        "machine composition outer multiplication",
                    )?,
                "machine composition outer value",
            )?;
            append(
                &mut output,
                crossing,
                outer_value,
                previous_value,
                false,
                work,
            )?;
            outer_coordinate = knot;
            previous_value = outer_value;
            outer_index += 1;
        }
        if exhausted && !near(outer.pairs()[outer_index - 1].1, image)? {
            return Err(unsupported("machine composition exhausted outer domain"));
        }
        let last_coordinate = output.last().unwrap().1;
        if last_coordinate < stop && !near(last_coordinate, stop)? {
            let outer_slope = outer
                .pairs()
                .get(outer_index)
                .ok_or_else(|| unsupported("machine composition segment outside active array"))?
                .0;
            carried_value = checked(
                outer_value
                    + checked(
                        checked(
                            image - outer_coordinate,
                            "machine composition final outer span",
                        )? * outer_slope,
                        "machine composition final multiplication",
                    )?,
                "machine composition final value",
            )?;
            append(&mut output, stop, carried_value, previous_value, true, work)?;
        } else {
            carried_value = previous_value;
        }
        inner_index += 1;
        coordinate = segment_end;
        inner_value = image;
        if near(upper, output.last().unwrap().1)? {
            break;
        }
    }
    PreparedSlopeFunction::from_machine_pairs(output)
        .map(Some)
        .map_err(FrontierUnknown::Arithmetic)
}
