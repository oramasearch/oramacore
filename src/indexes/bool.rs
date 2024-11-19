use std::collections::HashSet;

use dashmap::DashMap;

use crate::types::{DocumentId, FieldId};

#[derive(Debug, Default)]
struct BoolIndexPerField {
    true_docs: HashSet<DocumentId>,
    false_docs: HashSet<DocumentId>,
}

#[derive(Debug, Default)]
pub struct BoolIndex {
    maps: DashMap<FieldId, BoolIndexPerField>,
}

impl BoolIndex {
    pub fn new() -> Self {
        Self {
            maps: Default::default(),
        }
    }

    pub fn add(&self, doc_id: DocumentId, field_id: FieldId, value: bool) {
        let mut btree = self.maps.entry(field_id).or_default();
        if value {
            btree.true_docs.insert(doc_id);
        } else {
            btree.false_docs.insert(doc_id);
        }
    }

    pub fn filter(&self, field_id: FieldId, val: bool) -> HashSet<DocumentId> {
        let btree = match self.maps.get(&field_id) {
            Some(btree) => btree,
            // This should never happen: if the field is not in the index, it means that the field
            // was not indexed, and the filter should not have been created in the first place.
            None => return HashSet::new(),
        };

        if val {
            btree.true_docs.clone()
        } else {
            btree.false_docs.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::types::{DocumentId, FieldId};

    use super::BoolIndex;

    #[test]
    fn test_bool_index_filter() {
        let index = BoolIndex::new();

        index.add(DocumentId(0), FieldId(0), true);
        index.add(DocumentId(1), FieldId(0), false);
        index.add(DocumentId(2), FieldId(0), true);
        index.add(DocumentId(3), FieldId(0), false);
        index.add(DocumentId(4), FieldId(0), true);
        index.add(DocumentId(5), FieldId(0), false);

        let true_docs = index.filter(FieldId(0), true);
        assert_eq!(
            true_docs,
            HashSet::from([DocumentId(0), DocumentId(2), DocumentId(4)])
        );

        let false_docs = index.filter(FieldId(0), false);
        assert_eq!(
            false_docs,
            HashSet::from([DocumentId(1), DocumentId(3), DocumentId(5)])
        );
    }

    #[test]
    fn test_bool_index_filter_unknown_field() {
        let index = BoolIndex::new();

        index.add(DocumentId(0), FieldId(0), true);
        index.add(DocumentId(1), FieldId(0), false);
        index.add(DocumentId(2), FieldId(0), true);
        index.add(DocumentId(3), FieldId(0), false);
        index.add(DocumentId(4), FieldId(0), true);
        index.add(DocumentId(5), FieldId(0), false);

        let true_docs = index.filter(FieldId(1), true);
        assert_eq!(true_docs, HashSet::from([]));

        let false_docs = index.filter(FieldId(1), false);
        assert_eq!(false_docs, HashSet::from([]));
    }
}
