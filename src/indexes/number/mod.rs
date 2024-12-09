use std::{
    borrow::Cow,
    collections::{BTreeMap, HashSet},
    mem::swap,
};

use anyhow::Result;
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use dashmap::DashMap;
use linear::LinearNumberIndex;
use merge_iter::{MergeIter, MergeIterState};
use serde::{Deserialize, Serialize};

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

mod linear;
mod merge_iter;
mod serializable_number;
mod stats;

#[derive(Debug)]
pub struct NumberIndex {
    uncommitted: DashMap<FieldId, BTreeMap<Number, HashSet<DocumentId>>>,
    committed: LinearNumberIndex,
}

impl NumberIndex {
    pub fn new() -> Self {
        Self {
            uncommitted: Default::default(),
            committed: LinearNumberIndex::from_iter(std::iter::empty(), 2048),
        }
    }

    pub fn add(&self, doc_id: DocumentId, field_id: FieldId, value: Number) {
        let mut btree = self.uncommitted.entry(field_id).or_default();
        let doc_ids = btree.entry(value).or_default();
        doc_ids.insert(doc_id);
    }

    pub fn filter(&self, field_id: FieldId, filter: NumberFilter) -> Result<HashSet<DocumentId>> {
        use std::ops::Bound;

        let mut doc_ids = self.committed.filter(field_id, &filter, 0)?;

        if let Some(btree) = self.uncommitted.get(&field_id) {
            match filter {
                NumberFilter::Equal(value) => {
                    if let Some(d) = btree.get(&value) {
                        doc_ids.extend(d.iter().cloned());
                    }
                }
                NumberFilter::LessThan(value) => doc_ids.extend(
                    btree
                        .range((Bound::Unbounded, Bound::Excluded(&value)))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::LessThanOrEqual(value) => doc_ids.extend(
                    btree
                        .range((Bound::Unbounded, Bound::Included(&value)))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::GreaterThan(value) => doc_ids.extend(
                    btree
                        .range((Bound::Excluded(&value), Bound::Unbounded))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::GreaterThanOrEqual(value) => doc_ids.extend(
                    btree
                        .range((Bound::Included(&value), Bound::Unbounded))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
                NumberFilter::Between((min, max)) => doc_ids.extend(
                    btree
                        .range((Bound::Included(&min), Bound::Included(&max)))
                        .flat_map(|(_, doc_ids)| doc_ids.iter().cloned()),
                ),
            }
        };

        Ok(doc_ids)
    }

    pub fn commit(&mut self) {
        let committed = self.committed.iter();

        let mut dd: BTreeMap<Number, Cow<Vec<(DocumentId, FieldId)>>> = BTreeMap::new();
        for e in self.uncommitted.iter() {
            let field_id = *e.key();
            let btree = e.value();
            for (number, doc_ids) in btree.iter() {
                let a = dd.entry(*number).or_default();
                for doc_id in doc_ids.iter() {
                    a.to_mut().push((*doc_id, field_id));
                }
            }
        }

        let merge_iter = MergeIter {
            iter1: committed.map(|p| p.unwrap()),
            iter2: dd.into_iter(),
            state: MergeIterState::Unstarted,
        };

        let new_linear = LinearNumberIndex::from_iter(merge_iter, 2048);

        // This is safe because `commit` method keeps **`&mut self`**.
        // So no concurrent access is possible.
        self.committed = new_linear;
        self.uncommitted.clear();
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub enum NumberFilter {
    Equal(#[schema(inline)] Number),
    GreaterThan(#[schema(inline)] Number),
    GreaterThanOrEqual(#[schema(inline)] Number),
    LessThan(#[schema(inline)] Number),
    LessThanOrEqual(#[schema(inline)] Number),
    Between(#[schema(inline)] (Number, Number)),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Number {
    I32(#[schema(inline)] i32),
    F32(#[schema(inline)] f32),
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

impl std::ops::Add for Number {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Number::I32(a), Number::I32(b)) => (a + b).into(),
            (Number::I32(a), Number::F32(b)) => (a as f32 + b).into(),
            (Number::F32(a), Number::F32(b)) => (a + b).into(),
            (Number::F32(a), Number::I32(b)) => (a + b as f32).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use core::f32;
    use std::{cmp::Ordering, collections::HashSet};

    use tempdir::TempDir;

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

                assert_eq!(way.reverse(), other_way);
            }
        }
    }

    macro_rules! test_number_filter {
        ($fn_name: ident, $b: expr) => {
            #[test]
            fn $fn_name() {
                let mut index = NumberIndex::new();

                index.add(DocumentId(0), FieldId(0), 0.into());
                index.add(DocumentId(1), FieldId(0), 1.into());
                index.add(DocumentId(2), FieldId(0), 2.into());
                index.add(DocumentId(3), FieldId(0), 3.into());
                index.add(DocumentId(4), FieldId(0), 4.into());
                index.add(DocumentId(5), FieldId(0), 2.into());

                let a = $b;

                a(&index);

                index.commit();

                a(&index);

                let tmp_dir = TempDir::new("example").unwrap();
                let dump_path = tmp_dir.path().join("index_2");
                std::fs::remove_dir_all(dump_path.clone()).ok();
                std::fs::create_dir_all(dump_path.clone()).unwrap();
                index.committed.save_on_fs_and_unload(dump_path.clone()).unwrap();

                a(&index);
            }
        };
    }

    test_number_filter!(test_number_index_filter_eq, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );
    });
    test_number_filter!(test_number_index_filter_lt, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::LessThan(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(0), DocumentId(1)])
        );
    });
    test_number_filter!(test_number_index_filter_lt_equal, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::LessThanOrEqual(2.into()))
            .unwrap();
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
    test_number_filter!(test_number_index_filter_gt, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::GreaterThan(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(3), DocumentId(4)])
        );
    });
    test_number_filter!(test_number_index_filter_gt_equal, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::GreaterThanOrEqual(2.into()))
            .unwrap();
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
    test_number_filter!(test_number_index_filter_between, |index: &NumberIndex| {
        let output = index
            .filter(FieldId(0), NumberFilter::Between((2.into(), 3.into())))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(3), DocumentId(2), DocumentId(5)])
        );
    });

    #[test]
    fn test_number_commit() {
        let mut index = NumberIndex::new();

        index.add(DocumentId(0), FieldId(0), 0.into());
        index.add(DocumentId(1), FieldId(0), 1.into());
        index.add(DocumentId(2), FieldId(0), 2.into());
        index.add(DocumentId(3), FieldId(0), 3.into());
        index.add(DocumentId(4), FieldId(0), 4.into());
        index.add(DocumentId(5), FieldId(0), 2.into());

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        index.commit();

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );

        let tmp_dir = TempDir::new("example").unwrap();
        let dump_path = tmp_dir.path().join("index_2");
        std::fs::remove_dir_all(dump_path.clone()).ok();
        std::fs::create_dir_all(dump_path.clone()).unwrap();
        index.committed.save_on_fs_and_unload(dump_path.clone()).unwrap();

        let output = index
            .filter(FieldId(0), NumberFilter::Equal(2.into()))
            .unwrap();
        assert_eq!(
            output,
            HashSet::from_iter(vec![DocumentId(2), DocumentId(5)])
        );
    }
}
