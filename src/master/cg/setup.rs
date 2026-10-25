//! Preconditions for starting or resuming column generation.
use super::*;

pub(super) fn validate_options(options: &Options) -> Result<()> {
    if let Some(parameters) = &options.proximal {
        parameters.validate()?;
    }
    if !options.multiplicity.is_finite() || options.multiplicity < 0. || options.max_iterations == 0
    {
        return Err(invalid("column generation options"));
    }
    Ok(())
}

pub(super) fn validate_warm_start<M: LpMaster>(master: &M, warm: Option<&WarmStart>) -> Result<()> {
    if let Some(warm) = warm {
        if (!warm.bound.is_finite() && warm.bound != f64::NEG_INFINITY)
            || [warm.numerical_bound, warm.certified_bound]
                .iter()
                .any(|b| !b.is_finite() && *b != f64::NEG_INFINITY)
            || warm.bound
                != match warm.bound_mode {
                    BoundMode::Numerical => warm.numerical_bound,
                    BoundMode::Certified => warm.certified_bound,
                }
            || warm.columns.len() != master.costs().len()
            || warm.initial_rows > master.rhs().len()
            || warm.artificials.iter().any(|&c| c >= master.costs().len())
            || warm
                .cut_artificials
                .iter()
                .any(|&(r, c)| r >= master.rhs().len() || c >= master.costs().len())
            || warm
                .columns
                .iter()
                .flatten()
                .chain(&warm.generated_pool)
                .any(|column| column.entries.iter().any(|&(r, _)| r >= master.rhs().len()))
        {
            return Err(invalid("warm master metadata dimensions"));
        }
    }
    Ok(())
}
