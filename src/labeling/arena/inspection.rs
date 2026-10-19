use super::*;

impl Deref for LabelArena {
    type Target = Vec<StoredLabel>;
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

impl DerefMut for LabelArena {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.full_mut()
    }
}

impl<'a> IntoIterator for &'a LabelArena {
    type Item = &'a StoredLabel;
    type IntoIter = std::slice::Iter<'a, StoredLabel>;
    fn into_iter(self) -> Self::IntoIter {
        self.deref().iter()
    }
}

impl<'a> IntoIterator for &'a mut LabelArena {
    type Item = &'a mut StoredLabel;
    type IntoIter = std::slice::IterMut<'a, StoredLabel>;
    fn into_iter(self) -> Self::IntoIter {
        self.full_mut().iter_mut()
    }
}

impl IntoIterator for LabelArena {
    type Item = StoredLabel;
    type IntoIter = std::vec::IntoIter<StoredLabel>;
    fn into_iter(mut self) -> Self::IntoIter {
        std::mem::take(self.full_mut()).into_iter()
    }
}
