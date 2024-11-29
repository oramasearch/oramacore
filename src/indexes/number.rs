use std::collections::{BTreeMap, HashSet};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

#[derive(Debug, Default)]
pub struct NumberIndex {
    maps: DashMap<FieldId, BTreeMap<Number, HashSet<DocumentId>>>,
}

impl NumberIndex {
    pub fn new() -> Self {
        Self {
            maps: Default::default(),
        }
    }

    pub fn add(&self, doc_id: DocumentId, field_id: FieldId, value: Number) {
        let mut btree = self.maps.entry(field_id).or_default();
        let doc_ids = btree.entry(value).or_default();
        doc_ids.insert(doc_id);
    }

    pub fn filter(&self, field_id: FieldId, filter: NumberFilter) -> HashSet<DocumentId> {
        use std::ops::Bound;

        let btree = match self.maps.get(&field_id) {
            Some(btree) => btree,
            // This should never happen: if the field is not in the index, it means that the field
            // was not indexed, and the filter should not have been created in the first place.
            None => return HashSet::new(),
        };

        match filter {
            NumberFilter::Equal(value) => {
                if let Some(doc_ids) = btree.get(&value) {
                    HashSet::from_iter(doc_ids.iter().cloned())
                } else {
                    HashSet::new()
                }
            }
            NumberFilter::LessThan(value) => btree
                .range((Bound::Unbounded, Bound::Excluded(&value)))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned())
                .collect(),
            NumberFilter::LessThanOrEqual(value) => btree
                .range((Bound::Unbounded, Bound::Included(&value)))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned())
                .collect(),
            NumberFilter::GreaterThan(value) => btree
                .range((Bound::Excluded(&value), Bound::Unbounded))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned())
                .collect(),
            NumberFilter::GreaterThanOrEqual(value) => btree
                .range((Bound::Included(&value), Bound::Unbounded))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned())
                .collect(),
            NumberFilter::Between(min, max) => btree
                .range((Bound::Included(&min), Bound::Included(&max)))
                .flat_map(|(_, doc_ids)| doc_ids.iter().cloned())
                .collect(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum NumberFilter {
    Equal(Number),
    GreaterThan(Number),
    GreaterThanOrEqual(Number),
    LessThan(Number),
    LessThanOrEqual(Number),
    Between(Number, Number),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Number {
    I32(i32),
    F32(f32),
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Number::I32(value) => write!(f, "{}", value),
            Number::F32(value) => write!(f, "{}", value),
        }
    }
}

impl From<i32> for Number {
    fn from(value: i32) -> Self {
        Number::I32(value)
    }
}
impl From<f32> for Number {
    fn from(value: f32) -> Self {
        Number::F32(value)
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::I32(a), Number::I32(b)) => a == b,
            (Number::I32(a), Number::F32(b)) => *a as f32 == *b,
            (Number::F32(a), Number::F32(b)) => {
                // This is against the IEEE 754-2008 standard,
                // But we don't care here.
                if a.is_nan() && b.is_nan() {
                    return true;
                }
                a == b
            }
            (Number::F32(a), Number::I32(b)) => *a == *b as f32,
        }
    }
}
impl Eq for Number {}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // f32 is implemented as "binary32" type defined in IEEE 754-2008
        // So, it means, it can represent also +/- Infinity and NaN
        // Threat NaN as "more" the Infinity
        // See `total_cmp` method in f32
        match (self, other) {
            (Number::I32(a), Number::I32(b)) => a.cmp(b),
            (Number::I32(a), Number::F32(b)) => (*a as f32).total_cmp(b),
            (Number::F32(a), Number::F32(b)) => a.total_cmp(b),
            (Number::F32(a), Number::I32(b)) => a.total_cmp(&(*b as f32)),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32;
    use std::{cmp::Ordering, collections::HashSet};

    use super::*;

    #[test]
    fn test_number_eq() {
        let a = Number::from(1);
        let b = Number::from(1.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(-1);
        let b = Number::from(-1.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(2);
        let b = Number::from(2.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(f32::INFINITY);
        let b = Number::from(f32::NEG_INFINITY);
        assert_ne!(a, b);
        assert!(a != b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(f32::NAN);
        assert_eq!(a, Number::from(f32::NAN));
        assert!(a == Number::from(f32::NAN));
        assert!(a != Number::from(f32::INFINITY));
        assert!(a != Number::from(f32::NEG_INFINITY));
        assert!(a != Number::from(0));
    }

    #[test]
    fn test_number_ord() {
        let a = Number::from(1);

        let b = Number::from(1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(-1);

        let b = Number::from(-1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(1.0);

        let b = Number::from(1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(1);
        assert!(a < Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a < Number::from(f32::NAN));

        let a = Number::from(1.0);
        assert!(a < Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a < Number::from(f32::NAN));

        let a = Number::from(f32::NAN);
        assert!(a > Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a == Number::from(f32::NAN));

        let v = [
            Number::from(1),
            Number::from(1.0),
            Number::from(2),
            Number::from(2.0),
            Number::from(-1),
            Number::from(-1.0),
            Number::from(-2),
            Number::from(-2.0),
            Number::from(f32::INFINITY),
            Number::from(f32::NEG_INFINITY),
            Number::from(f32::NAN),
        ];

        for i in 0..v.len() {
            for j in 0..v.len() {
                let way = v[i].cmp(&v[j]);
                let other_way = v[j].cmp(&v[i]);

                println!("{:?} cmp {:?}", v[i], v[j]);
                assert_eq!(way.reverse(), other_way);
            }
        }
    }

    macro_rules! test_number_filter {
        ($fn_name: ident, $b: expr) => {
            #[test]
            fn $fn_name() {
                let index = NumberIndex::new();

                index.add(DocumentId(0), FieldId(0), 0.into());
                index.add(DocumentId(1), FieldId(0), 1.into());
                index.add(DocumentId(2), FieldId(0), 2.into());
                index.add(DocumentId(3), FieldId(0), 3.into());
                index.add(DocumentId(4), FieldId(0), 4.into());
                index.add(DocumentId(5), FieldId(0), 2.into());

                let a = $b;

                a(index);
            }
        };
    }

    test_number_filter!(test_number_index_filter_eq, |index: NumberIndex| {
        let output = index.filter(FieldId(0), NumberFilter::Equal(2.into()));
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );
    });
    test_number_filter!(test_number_index_filter_lt, |index: NumberIndex| {
        let output = index.filter(FieldId(0), NumberFilter::LessThan(2.into()));
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(0), DocumentId(1)])
        );
    });
    test_number_filter!(test_number_index_filter_lt_equal, |index: NumberIndex| {
        let output = index.filter(FieldId(0), NumberFilter::LessThanOrEqual(2.into()));
        assert_eq!(
            output,
            HashSet::from_iter(vec![
                DocumentId(0),
                DocumentId(1),
                DocumentId(2),
                DocumentId(5)
            ])
        );
    });
    test_number_filter!(test_number_index_filter_gt, |index: NumberIndex| {
        let output = index.filter(FieldId(0), NumberFilter::GreaterThan(2.into()));
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(3), DocumentId(4)])
        );
    });
    test_number_filter!(test_number_index_filter_gt_equal, |index: NumberIndex| {
        let output = index.filter(FieldId(0), NumberFilter::GreaterThanOrEqual(2.into()));
        assert_eq!(
            output,
            HashSet::from_iter(vec![
                DocumentId(3),
                DocumentId(4),
                DocumentId(2),
                DocumentId(5)
            ])
        );
    });
    test_number_filter!(test_number_index_filter_between, |index: NumberIndex| {
        let output = index.filter(FieldId(0), NumberFilter::Between(2.into(), 3.into()));
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(3), DocumentId(2), DocumentId(5)])
        );
    });
}
