//! Resource-generator display names; never an exact disjunction identity.
pub(super) fn generator_name(group: usize, resource: usize, threshold: f64) -> String {
    format!("ES_{group}_rID_{resource}_{}", number(threshold))
}

fn number(value: f64) -> String {
    // Default floating ostream formatting uses six significant digits. Round
    // before choosing fixed/scientific notation, including carry into exponent 6.
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..6).contains(&exponent) {
        return format!(
            "{}e{exponent:+03}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        );
    }
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let digits = mantissa.trim_start_matches('-').replace('.', "");
    let point = exponent + 1;
    let mut fixed = if point <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!(
            "{sign}{digits}{}",
            "0".repeat(point as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    };
    if fixed.contains('.') {
        fixed.truncate(fixed.trim_end_matches('0').trim_end_matches('.').len());
    }
    fixed
}

#[cfg(test)]
#[path = "../../../../tests/unit/master/branching/grouped_resource/name/tests.rs"]
mod tests;
