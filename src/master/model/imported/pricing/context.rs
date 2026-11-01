//! Explicit node restrictions visible to external pricing callbacks.
use super::*;

fn intersect(left: Bounds, right: Bounds) -> Bounds {
    Bounds {
        lower: left.lower.max(right.lower),
        upper: left.upper.min(right.upper),
    }
}

impl<const D: usize> ImportPricing<D> {
    /// Explicit bounds on an original aggregate coordinate at this request.
    /// These constrain the sum of contributions across groups, not each new
    /// column's vector. No deductions from the finite restricted master enter.
    pub fn original_bounds(&self, id: i32) -> Result<Bounds> {
        Self::original_bounds_in(&self.definition, &self.context, id)
    }
    fn original_bounds_in(
        definition: &Definition<D>,
        context: &Context<D>,
        id: i32,
    ) -> Result<Bounds> {
        let initial = definition
            .variables
            .iter()
            .find(|variable| variable.id == id)
            .ok_or_else(|| invalid("unknown import original variable"))?
            .bounds;
        Ok(context
            .ordered()
            .into_iter()
            .fold(initial, |current, restriction| match restriction {
                Restriction::Variable(key, bounds) if *key == id => intersect(current, *bounds),
                _ => current,
            }))
    }

    /// Explicit quantity bounds for a known column in this request's group.
    /// Column identities remain group-qualified when retargeting a request.
    pub fn column_bounds(&self, column: i32) -> Result<Bounds> {
        Self::column_bounds_in(&self.context, self.subproblem(), column)
    }
    fn column_bounds_in(context: &Context<D>, group: &Subproblem, column: i32) -> Result<Bounds> {
        let initial = group
            .columns
            .iter()
            .find(|known| known.id == column)
            .ok_or_else(|| invalid("unknown imported column"))?
            .bounds;
        Ok(context
            .ordered()
            .into_iter()
            .fold(initial, |current, restriction| match restriction {
                Restriction::Column(id, key, bounds) if *id == group.id && *key == column => {
                    intersect(current, *bounds)
                }
                _ => current,
            }))
    }

    /// Explicit bounds on total column quantity in this request's group,
    /// including accepted and future columns. These are not per-column bounds.
    pub fn multiplicity_bounds(&self) -> Bounds {
        Self::multiplicity_bounds_in(&self.context, self.subproblem())
    }
    fn multiplicity_bounds_in(context: &Context<D>, group: &Subproblem) -> Bounds {
        context
            .ordered()
            .into_iter()
            .fold(
                group.multiplicity,
                |current, restriction| match restriction {
                    Restriction::Multiplicity(id, bounds) if *id == group.id => {
                        intersect(current, *bounds)
                    }
                    _ => current,
                },
            )
    }

    /// Declared domain restrictions on every individual future column in this
    /// group. Aggregate coordinate bounds do not tighten this feature box.
    pub fn envelope(&self) -> Option<&ColumnEnvelope> {
        self.definition.envelopes.get(&self.subproblem().id)
    }
}

/// Read-only external-oracle child context, available before its first LP solve.
/// Aggregate restrictions and per-pattern envelopes remain distinct.
#[derive(Clone, Debug)]
pub struct ImportOracleNode<const D: usize = { crate::DEFAULT_RESOURCE_DIMENSIONS }> {
    definition: Rc<Definition<D>>,
    context: Rc<Context<D>>,
    position: usize,
    depth: usize,
}
impl<const D: usize> ImportOracleNode<D> {
    pub(in crate::master::model::imported) fn new(
        definition: Rc<Definition<D>>,
        context: Rc<Context<D>>,
        group: i32,
        depth: usize,
    ) -> Result<Self> {
        let position = definition
            .subproblems
            .iter()
            .position(|s| s.id == group)
            .ok_or_else(|| invalid("unknown import oracle group"))?;
        Ok(Self {
            definition,
            context,
            position,
            depth,
        })
    }
    pub fn depth(&self) -> usize {
        self.depth
    }
    pub fn subproblem(&self) -> &Subproblem {
        &self.definition.subproblems[self.position]
    }
    /// Explicit bounds on the aggregate original coordinate, not each pattern.
    pub fn original_bounds(&self, id: i32) -> Result<Bounds> {
        ImportPricing::<D>::original_bounds_in(&self.definition, &self.context, id)
    }
    /// Quantity bounds on a known column qualified by this context's group.
    pub fn column_bounds(&self, column: i32) -> Result<Bounds> {
        ImportPricing::<D>::column_bounds_in(&self.context, self.subproblem(), column)
    }
    pub fn multiplicity_bounds(&self) -> Bounds {
        ImportPricing::<D>::multiplicity_bounds_in(&self.context, self.subproblem())
    }
    pub fn envelope(&self) -> Option<&ColumnEnvelope> {
        self.definition.envelopes.get(&self.subproblem().id)
    }
}
