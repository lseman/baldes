//! Finite original-vector column domains, independent of routing graphs.
use super::*;
use crate::master::preprocessing::Bounds;
use std::collections::BTreeSet;
mod compile;
mod envelope;
mod multiple;
mod nonlinear_rows;
pub use envelope::ColumnEnvelope;
pub use multiple::{
    ImportTree, ImportTreeProgress, ImportTreeSolution, ImportTreeStatus, MultiImport,
    MultiImportProgress,
};
mod lifecycle;
use lifecycle::{Context, Definition, Restriction};
mod pricing;
mod snapshot;
mod source;
pub use pricing::{
    ImportCertificate, ImportCosts, ImportOracleReply, ImportPricing, ImportPricingKind,
    ImportPricingMinimum,
};
pub use snapshot::ImportedSnapshot;
pub use source::{ColumnBatch, ColumnImport, ImportOptions, ImportProgress, ImportStatus};

/// One declared column. Cost is additional to original-variable objective
/// costs. Entries reference original master-variable IDs in the declared support.
#[derive(Clone, Debug)]
pub struct Column {
    pub id: i32,
    pub cost: f64,
    pub bounds: Bounds,
    pub integer: bool,
    pub entries: Vec<(i32, f64)>,
}

/// A complete finite domain as declared by the caller, not an assertion about
/// columns omitted from some external pricing problem. IDs share the pricing
/// subproblem namespace with routing networks; column IDs are local to a group.
#[derive(Clone, Debug)]
pub struct Subproblem {
    pub id: i32,
    pub variables: Vec<i32>,
    pub multiplicity: Bounds,
    pub columns: Vec<Column>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ColumnValue {
    pub subproblem: i32,
    pub column: i32,
    pub value: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Values {
    pub original: Vec<(i32, f64)>,
    pub columns: Vec<ColumnValue>,
}

#[derive(Debug)]
struct Registry {
    links: BTreeMap<i32, usize>,
    original: Vec<i32>,
    columns: BTreeMap<(i32, i32), i32>,
    groups: BTreeMap<i32, Vec<i32>>,
}
impl Registry {
    fn values(&self, values: &[(i32, f64)]) -> Result<Values> {
        let map: BTreeMap<_, _> = values.iter().copied().collect();
        if map.len() != values.len()
            || map.len() != self.original.len() + self.columns.len()
            || values.iter().any(|(_, v)| !v.is_finite())
        {
            return Err(invalid("imported master-value registry"));
        }
        let get = |id| {
            map.get(&id)
                .copied()
                .ok_or_else(|| invalid("missing imported master value"))
        };
        Ok(Values {
            original: self
                .original
                .iter()
                .map(|&id| Ok((id, get(id)?)))
                .collect::<Result<_>>()?,
            columns: self
                .columns
                .iter()
                .map(|(&(subproblem, column), &id)| {
                    Ok(ColumnValue {
                        subproblem,
                        column,
                        value: get(id)?,
                    })
                })
                .collect::<Result<_>>()?,
        })
    }
}

#[derive(Debug)]
pub struct ImportedOutput<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    /// Ordinary solver diagnostics and routing results. Its master values also
    /// contain internal column variables; `values` uses the caller's namespaces.
    pub output: Output<D>,
    pub values: Values,
    context: Rc<Context<D>>,
}
impl<const D: usize> ImportedOutput<D> {
    fn new(output: Output<D>, context: Rc<Context<D>>) -> Result<Self> {
        let values = context.registry.values(&output.master_values)?;
        Ok(Self {
            output,
            values,
            context,
        })
    }
    /// A private reusable child retaining the import identity map.
    pub fn snapshot(&self) -> Result<ImportedSnapshot<D>> {
        Ok(ImportedSnapshot {
            node: self.output.snapshot.fork()?,
            context: self.context.clone(),
        })
    }
}

/// Import declared finite column domains alongside optional routing networks.
/// For each declared original variable x, append x=sum(a_j*lambda_j) across
/// all imports declaring x. Undeclared variables remain independent. All
/// lambda variables are loaded immediately; routing pricing stays unchanged.
/// Fixing and multiplicity restrictions retain every fixed column's cost and
/// original-variable contribution. This is not a dynamic pricing callback.
pub fn solve<const D: usize>(
    problem: Problem<D>,
    variables: Vec<MasterVariable>,
    subproblems: Vec<Subproblem>,
    options: Options,
) -> Result<ImportedOutput<D>> {
    lifecycle::rebuild(
        Rc::new(Definition {
            problem,
            variables,
            subproblems,
            envelopes: BTreeMap::new(),
        }),
        None,
        options,
    )
}

fn quantity_bounds(bounds: Bounds) -> Result<()> {
    if !super::master_variables::valid_bounds(bounds)
        || !bounds.lower.is_finite()
        || bounds.lower < 0.
    {
        Err(invalid("nonnegative imported quantity bounds"))
    } else {
        Ok(())
    }
}
