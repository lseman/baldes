//! Numerical request reuse while source collection changes only known columns.
use super::*;

pub(in crate::master::mp::imported) struct PointCache<const D: usize> {
    point: Option<ImportPricing<D>>,
    // A known feasible point remains feasible when new quantities can be zero.
    feasible_extension: bool,
}

impl<const D: usize> Default for PointCache<D> {
    fn default() -> Self {
        Self {
            point: None,
            feasible_extension: false,
        }
    }
}

impl<const D: usize> PointCache<D> {
    /// Call only for the batch returned to the enclosing collector's atomic
    /// acceptance step. A rejected batch aborts that collection and its cache.
    pub(in crate::master::mp::imported) fn accepted(&mut self, batch: &ColumnBatch) {
        self.feasible_extension = self.point.as_ref().is_some_and(|point| {
            point.kind == ImportPricingKind::Objective
                && batch.columns.iter().all(|column| column.bounds.lower == 0.)
        });
    }

    pub(in crate::master::mp::imported) fn next(
        &mut self,
        session: &ColumnImport<D>,
        options: &Options,
        deadline: Instant,
    ) -> Result<Option<ImportPricing<D>>> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        if let Some(previous) = &self.point {
            if Rc::ptr_eq(&previous.definition, &session.definition)
                && Rc::ptr_eq(&previous.context, &session.context)
            {
                let mut point = previous.clone();
                point.position = session.position;
                if point.open_groups.as_ref() != &session.open_groups {
                    // Completion does not change the numerical RMP. Its proof
                    // coverage does change, so old-group minima cannot be mixed.
                    point.open_groups = Rc::new(session.open_groups.clone());
                    point.epoch = Rc::new(());
                    point.retained_bound = None;
                }
                self.point = Some(point.clone());
                return Ok(Some(point));
            }
        }
        let can_extend = self.feasible_extension
            && self
                .point
                .as_ref()
                .is_some_and(|point| Rc::ptr_eq(&point.context, &session.context));
        let point = if can_extend {
            let mut configured = options.clone();
            configured.cg.time_limit = configured.cg.time_limit.min(remaining);
            // The collector changes no topology, coefficients or restrictions.
            // Retain existing routing columns as the feasible zero-extension
            // witness while the new declared columns receive a fresh registry.
            let seeds = &self.point.as_ref().unwrap().routing;
            match session.objective_point_seeded(configured, false, seeds) {
                Ok(point) => Some(point),
                Err(_) if Instant::now() >= deadline => None,
                Err(error) => return Err(error),
            }
        } else {
            super::next_point(session, options, deadline)?
        };
        self.feasible_extension = point
            .as_ref()
            .is_some_and(|point| point.kind == ImportPricingKind::Objective);
        self.point = point.clone();
        Ok(point)
    }
}
