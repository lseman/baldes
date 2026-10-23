use super::*;

impl<const D: usize> Deref for LabelArena<D> {
    type Target = Vec<StoredLabel<D>>;
    fn deref(&self) -> &Self::Target {
        match &self.records {
            Records::Full(v) => v,
            Records::Compact2(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Compact4(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Compact8(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Auxiliary2(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Auxiliary4(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Auxiliary8(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::AuxiliaryFull(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Branched(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Extended(v) => self
                .inspected
                .get_or_init(|| v.iter().enumerate().map(|(id, l)| l.unpack(id)).collect()),
            Records::Ordinary(v) => self.inspected.get_or_init(|| {
                v.iter()
                    .enumerate()
                    .map(|(id, l)| l.view(id).to_owned())
                    .collect()
            }),
        }
    }
}

impl<const D: usize> DerefMut for LabelArena<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.full_mut()
    }
}

impl<'a, const D: usize> IntoIterator for &'a LabelArena<D> {
    type Item = &'a StoredLabel<D>;
    type IntoIter = std::slice::Iter<'a, StoredLabel<D>>;
    fn into_iter(self) -> Self::IntoIter {
        self.deref().iter()
    }
}

impl<'a, const D: usize> IntoIterator for &'a mut LabelArena<D> {
    type Item = &'a mut StoredLabel<D>;
    type IntoIter = std::slice::IterMut<'a, StoredLabel<D>>;
    fn into_iter(self) -> Self::IntoIter {
        self.full_mut().iter_mut()
    }
}

impl<const D: usize> IntoIterator for LabelArena<D> {
    type Item = StoredLabel<D>;
    type IntoIter = std::vec::IntoIter<StoredLabel<D>>;
    fn into_iter(mut self) -> Self::IntoIter {
        std::mem::take(self.full_mut()).into_iter()
    }
}
