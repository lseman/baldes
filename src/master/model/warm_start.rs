//! Private retained metadata; public column-generation warm starts remain owned.
use super::*;
use std::ops::{Deref, DerefMut};

/// One immutable descriptor in a shared batch, or a column detached by an edit.
/// Batching avoids one Arc allocation for every column when retaining a solve.
#[derive(Clone)]
pub(super) enum RetainedColumn {
    Batch {
        columns: Arc<Vec<Option<cg::Column>>>,
        index: usize,
    },
    Edited(Arc<cg::Column>),
}
impl std::fmt::Debug for RetainedColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}
impl Deref for RetainedColumn {
    type Target = cg::Column;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Batch { columns, index } => columns[*index].as_ref().unwrap(),
            Self::Edited(column) => column,
        }
    }
}
impl DerefMut for RetainedColumn {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if let Self::Batch { columns, index } = self {
            *self = Self::Edited(Arc::new(columns[*index].as_ref().unwrap().clone()));
        }
        match self {
            Self::Edited(column) => Arc::make_mut(column),
            Self::Batch { .. } => unreachable!(),
        }
    }
}
impl From<cg::Column> for RetainedColumn {
    fn from(column: cg::Column) -> Self {
        Self::Edited(Arc::new(column))
    }
}
impl RetainedColumn {
    pub(super) fn into_owned(self) -> cg::Column {
        match self {
            Self::Batch { columns, index } => match Arc::try_unwrap(columns) {
                Ok(mut columns) => columns[index].take().unwrap(),
                Err(columns) => columns[index].as_ref().unwrap().clone(),
            },
            Self::Edited(column) => Arc::unwrap_or_clone(column),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct RetainedWarmStart {
    pub bound: f64,
    pub bound_mode: cg::BoundMode,
    pub numerical_bound: f64,
    pub certified_bound: f64,
    pub stabilization: Option<crate::master::stabilization::DualPriceSmoothing>,
    pub proximal: Option<cg::ProximalWarmStart>,
    pub generated_pool: Vec<RetainedColumn>,
    pub columns: Vec<Option<RetainedColumn>>,
    pub artificials: Vec<usize>,
    pub cut_artificials: Vec<(usize, usize)>,
    pub initial_rows: usize,
}
impl From<cg::WarmStart> for RetainedWarmStart {
    fn from(mut warm: cg::WarmStart) -> Self {
        let loaded = warm.columns.len();
        let mut batch = std::mem::take(&mut warm.columns);
        batch.extend(
            std::mem::take(&mut warm.generated_pool)
                .into_iter()
                .map(Some),
        );
        let batch = Arc::new(batch);
        let column = |index| RetainedColumn::Batch {
            columns: batch.clone(),
            index,
        };
        Self {
            bound: warm.bound,
            bound_mode: warm.bound_mode,
            numerical_bound: warm.numerical_bound,
            certified_bound: warm.certified_bound,
            stabilization: warm.stabilization,
            proximal: warm.proximal,
            columns: (0..loaded)
                .map(|i| batch[i].as_ref().map(|_| column(i)))
                .collect(),
            generated_pool: (loaded..batch.len()).map(column).collect(),
            artificials: warm.artificials,
            cut_artificials: warm.cut_artificials,
            initial_rows: warm.initial_rows,
        }
    }
}
impl RetainedWarmStart {
    pub(super) fn into_owned(self) -> cg::WarmStart {
        cg::WarmStart {
            bound: self.bound,
            bound_mode: self.bound_mode,
            numerical_bound: self.numerical_bound,
            certified_bound: self.certified_bound,
            stabilization: self.stabilization,
            proximal: self.proximal,
            columns: self
                .columns
                .into_iter()
                .map(|c| c.map(RetainedColumn::into_owned))
                .collect(),
            generated_pool: self
                .generated_pool
                .into_iter()
                .map(RetainedColumn::into_owned)
                .collect(),
            artificials: self.artificials,
            cut_artificials: self.cut_artificials,
            initial_rows: self.initial_rows,
        }
    }
    pub(super) fn clear_proximal(&mut self) {
        if let Some(state) = &mut self.proximal {
            state.child = true;
            state.snapshot.bound = f64::NEG_INFINITY;
        }
    }
}

/// Operations shared by live CG metadata and retained node metadata. Selection
/// precedes mutable access so retained columns detach only when actually edited.
pub(super) trait WarmMetadata {
    fn artificials(&self) -> &[usize];
    fn cut_artificials(&self) -> &[(usize, usize)];
    fn loaded_tags(&self) -> HashSet<i32>;
    fn edit_columns(
        &mut self,
        select: impl FnMut(&cg::Column) -> bool,
        edit: impl FnMut(&mut cg::Column),
    );
    fn clear_stabilization(&mut self);
    fn clear_proximal(&mut self);
    fn push_artificial(&mut self, column: usize);
}

impl WarmMetadata for cg::WarmStart {
    fn artificials(&self) -> &[usize] {
        &self.artificials
    }
    fn cut_artificials(&self) -> &[(usize, usize)] {
        &self.cut_artificials
    }
    fn loaded_tags(&self) -> HashSet<i32> {
        self.columns
            .iter()
            .flatten()
            .map(|column| column.tag)
            .collect()
    }
    fn edit_columns(
        &mut self,
        mut select: impl FnMut(&cg::Column) -> bool,
        mut edit: impl FnMut(&mut cg::Column),
    ) {
        for column in self
            .columns
            .iter_mut()
            .flatten()
            .chain(&mut self.generated_pool)
        {
            if select(&*column) {
                edit(column);
            }
        }
    }
    fn clear_stabilization(&mut self) {
        self.stabilization = None;
    }
    fn clear_proximal(&mut self) {
        cg::WarmStart::clear_proximal(self);
    }
    fn push_artificial(&mut self, column: usize) {
        self.columns.push(None);
        self.artificials.push(column);
    }
}

impl WarmMetadata for RetainedWarmStart {
    fn artificials(&self) -> &[usize] {
        &self.artificials
    }
    fn cut_artificials(&self) -> &[(usize, usize)] {
        &self.cut_artificials
    }
    fn loaded_tags(&self) -> HashSet<i32> {
        self.columns
            .iter()
            .flatten()
            .map(|column| column.tag)
            .collect()
    }
    fn edit_columns(
        &mut self,
        mut select: impl FnMut(&cg::Column) -> bool,
        mut edit: impl FnMut(&mut cg::Column),
    ) {
        for column in self
            .columns
            .iter_mut()
            .flatten()
            .chain(&mut self.generated_pool)
        {
            if select(&*column) {
                edit(column);
            }
        }
    }
    fn clear_stabilization(&mut self) {
        self.stabilization = None;
    }
    fn clear_proximal(&mut self) {
        RetainedWarmStart::clear_proximal(self);
    }
    fn push_artificial(&mut self, column: usize) {
        self.columns.push(None);
        self.artificials.push(column);
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/master/model/retained_warm_start_tests.rs"]
mod tests;
