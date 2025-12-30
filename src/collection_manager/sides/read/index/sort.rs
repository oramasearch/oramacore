use std::collections::HashSet;
use std::iter::Peekable;

use crate::{
    collection_manager::sides::read::ReadError,
    types::{DocumentId, Number, SortOrder},
};

use super::{
    committed_field::{CommittedBoolField, CommittedDateField, CommittedNumberField},
    path_to_index_id_map::PathToIndexId,
    uncommitted_field::{UncommittedBoolField, UncommittedDateFilterField, UncommittedNumberField},
    CommittedFields, FieldType, UncommittedFields,
};

// =============================================================================
// Type aliases for complex iterator types
// =============================================================================

/// A boxed iterator yielding (sort_key, document_ids) pairs.
/// Used for sorting operations that require forward iteration only.
type SortedDocIdsIter<'s> = Box<dyn Iterator<Item = (Number, &'s HashSet<DocumentId>)> + 's>;

/// A boxed double-ended iterator yielding (sort_key, document_ids) pairs.
/// Used for sorting operations that need both ascending and descending traversal.
type SortedDocIdsBidirectionalIter<'s> =
    Box<dyn DoubleEndedIterator<Item = (Number, &'s HashSet<DocumentId>)> + 's>;

/// Result type for sort execution, returning either a sorted iterator or a read error.
type SortExecuteResult<'s> = Result<SortedDocIdsIter<'s>, ReadError>;

/// A peekable boxed iterator for merge operations.
/// Used internally by SortIterator to peek at the next element without consuming it.
type PeekableSortIter<'s, T> = Peekable<Box<dyn Iterator<Item = (T, &'s HashSet<DocumentId>)> + 's>>;

// =============================================================================
// Sortable trait
// =============================================================================

/// Trait for fields that support sorting operations.
///
/// This trait provides a uniform interface for sorting documents across
/// different field types (Bool, Number, Date). Each implementation
/// converts its native type to `Number` for unified sorting.
///
/// # Design
///
/// The trait returns a boxed `DoubleEndedIterator` to support both ascending
/// and descending sort orders. The iterator yields tuples of (Number, &HashSet<DocumentId>)
/// where Number is the sort key. References to stored HashSets avoid cloning.
trait Sortable {
    /// Returns a sorted iterator over (sort_key, document_ids) pairs.
    ///
    /// The iterator yields values in ascending order by default.
    /// Use `.rev()` on the returned iterator for descending order.
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s>;
}

// =============================================================================
// Sortable implementations for Uncommitted fields
// =============================================================================

impl Sortable for UncommittedNumberField {
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s> {
        Box::new(self.iter_ref())
    }
}

impl Sortable for UncommittedDateFilterField {
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s> {
        Box::new(
            self.iter_ref()
                .map(|(timestamp, doc_ids)| (i64_to_number(timestamp), doc_ids)),
        )
    }
}

impl Sortable for UncommittedBoolField {
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s> {
        let (true_docs, false_docs) = self.inner_ref();
        // Ascending order: false (0) first, then true (1)
        Box::new(
            vec![
                (bool_to_number(false), false_docs),
                (bool_to_number(true), true_docs),
            ]
            .into_iter(),
        )
    }
}

// =============================================================================
// Sortable implementations for Committed fields
// =============================================================================

impl Sortable for CommittedNumberField {
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s> {
        Box::new(
            self.iter_ref()
                .map(|(serializable_number, doc_ids)| (serializable_number.0, doc_ids)),
        )
    }
}

impl Sortable for CommittedDateField {
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s> {
        Box::new(
            self.iter_ref()
                .map(|(timestamp, doc_ids)| (i64_to_number(timestamp), doc_ids)),
        )
    }
}

impl Sortable for CommittedBoolField {
    fn iter_sorted<'s>(&'s self) -> SortedDocIdsBidirectionalIter<'s> {
        let (true_docs, false_docs) = self.inner_ref();
        // Ascending order: false (0) first, then true (1)
        Box::new(
            vec![
                (bool_to_number(false), false_docs),
                (bool_to_number(true), true_docs),
            ]
            .into_iter(),
        )
    }
}

// =============================================================================
// IndexSortContext
// =============================================================================

/// Context required for executing sort operations on an index.
pub struct IndexSortContext<'index> {
    path_to_index_id_map: &'index PathToIndexId,
    uncommitted_fields: &'index UncommittedFields,
    committed_fields: &'index CommittedFields,
    field_name: String,
    order: SortOrder,
}

impl<'index> IndexSortContext<'index> {
    /// Creates a new IndexSortContext with the provided sort dependencies.
    pub(crate) fn new(
        path_to_index_id_map: &'index PathToIndexId,
        uncommitted_fields: &'index UncommittedFields,
        committed_fields: &'index CommittedFields,
        field_name: &str,
        order: SortOrder,
    ) -> Self {
        Self {
            path_to_index_id_map,
            uncommitted_fields,
            committed_fields,
            field_name: field_name.to_string(),
            order,
        }
    }

    /// Executes the sort operation and returns the sorted iterator.
    ///
    /// This is the main entry point for sorting within an IndexSortContext.
    pub fn execute<'s>(&'s self) -> SortExecuteResult<'s>
    where
        'index: 's,
    {
        let (field_id, field_type) = self
            .path_to_index_id_map
            .get_filter_field(&self.field_name)
            .ok_or_else(|| ReadError::SortFieldNotFound(self.field_name.clone()))?;

        match field_type {
            FieldType::Number => {
                let uncommitted = self
                    .uncommitted_fields
                    .number_fields
                    .get(&field_id)
                    .ok_or_else(|| {
                        ReadError::Generic(anyhow::anyhow!(
                            "Field {} is not a number field",
                            self.field_name
                        ))
                    })?;
                let committed = self.committed_fields.number_fields.get(&field_id);
                Ok(calculate_sort_on_field(uncommitted, committed, self.order))
            }
            FieldType::Bool => {
                let uncommitted = self
                    .uncommitted_fields
                    .bool_fields
                    .get(&field_id)
                    .ok_or_else(|| {
                        ReadError::Generic(anyhow::anyhow!(
                            "Field {} is not a bool field",
                            self.field_name
                        ))
                    })?;
                let committed = self.committed_fields.bool_fields.get(&field_id);
                Ok(calculate_sort_on_field(uncommitted, committed, self.order))
            }
            FieldType::Date => {
                let uncommitted = self
                    .uncommitted_fields
                    .date_fields
                    .get(&field_id)
                    .ok_or_else(|| {
                        ReadError::Generic(anyhow::anyhow!(
                            "Field {} is not a date field",
                            self.field_name
                        ))
                    })?;
                let committed = self.committed_fields.date_fields.get(&field_id);
                Ok(calculate_sort_on_field(uncommitted, committed, self.order))
            }
            _ => Err(ReadError::InvalidSortField(
                self.field_name.clone(),
                format!("{field_type:?}"),
            )),
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Converts a boolean value to a Number for sorting.
///
/// false = 0, true = 1
/// This ensures ascending order places false before true.
fn bool_to_number(value: bool) -> Number {
    if value {
        Number::I32(1)
    } else {
        Number::I32(0)
    }
}

/// Converts an i64 timestamp to a Number for sorting.
///
/// Attempts to fit the value into an i32, clamping to i32::MIN/MAX if out of range.
fn i64_to_number(value: i64) -> Number {
    if let Ok(i) = i32::try_from(value) {
        Number::I32(i)
    } else if value > 0 {
        Number::I32(i32::MAX)
    } else {
        Number::I32(i32::MIN)
    }
}

/// Merges sorted iterators from uncommitted and committed fields.
///
/// This generic function handles the combination of uncommitted (in-memory) and
/// committed (persisted) field data.
fn calculate_sort_on_field<'a, UF, CF>(
    uncommitted_field: &'a UF,
    committed_field: Option<&'a CF>,
    order: SortOrder,
) -> SortedDocIdsIter<'a>
where
    UF: Sortable,
    CF: Sortable,
{
    let uncommitted_iter = uncommitted_field.iter_sorted();

    let uncommitted_iter: SortedDocIdsIter<'a> = match order {
        SortOrder::Ascending => Box::new(uncommitted_iter),
        SortOrder::Descending => Box::new(uncommitted_iter.rev()),
    };

    if let Some(committed_field) = committed_field {
        let committed_iter = committed_field.iter_sorted();

        let committed_iter: SortedDocIdsIter<'a> = match order {
            SortOrder::Ascending => Box::new(committed_iter),
            SortOrder::Descending => Box::new(committed_iter.rev()),
        };

        // If there's a committed field, merge the two iterators
        Box::new(SortIterator::new(uncommitted_iter, committed_iter, order))
    } else {
        uncommitted_iter
    }
}

// =============================================================================
// SortIterator - merges two sorted iterators
// =============================================================================

/// Iterator that merges two sorted iterators into a single sorted stream.
///
/// This iterator takes two already-sorted iterators and produces a merged
/// output that maintains the sort order. It uses a peek-compare-advance
/// strategy to efficiently merge without buffering all data.
struct SortIterator<'s1, 's2, T: Ord> {
    iter1: PeekableSortIter<'s1, T>,
    iter2: PeekableSortIter<'s2, T>,
    order: SortOrder,
}

impl<'s1, 's2, T: Ord> SortIterator<'s1, 's2, T> {
    /// Creates a new SortIterator from two iterators.
    fn new(
        iter1: Box<dyn Iterator<Item = (T, &'s1 HashSet<DocumentId>)> + 's1>,
        iter2: Box<dyn Iterator<Item = (T, &'s2 HashSet<DocumentId>)> + 's2>,
        order: SortOrder,
    ) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            order,
        }
    }
}

impl<'s, T: Ord> Iterator for SortIterator<'s, 's, T> {
    type Item = (T, &'s HashSet<DocumentId>);

    fn next(&mut self) -> Option<Self::Item> {
        let el1 = self.iter1.peek();
        let el2 = self.iter2.peek();

        match (el1, el2) {
            (None, None) => None,
            (Some(_), None) => self.iter1.next(),
            (None, Some(_)) => self.iter2.next(),
            (Some((k1, _)), Some((k2, _))) => match self.order {
                SortOrder::Ascending => {
                    if k1 < k2 {
                        self.iter1.next()
                    } else {
                        self.iter2.next()
                    }
                }
                SortOrder::Descending => {
                    if k1 > k2 {
                        self.iter1.next()
                    } else {
                        self.iter2.next()
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_to_number() {
        assert_eq!(bool_to_number(false), Number::I32(0));
        assert_eq!(bool_to_number(true), Number::I32(1));
    }

    #[test]
    fn test_i64_to_number() {
        assert_eq!(i64_to_number(0), Number::I32(0));
        assert_eq!(i64_to_number(100), Number::I32(100));
        assert_eq!(i64_to_number(-100), Number::I32(-100));
        assert_eq!(i64_to_number(i32::MAX as i64), Number::I32(i32::MAX));
        assert_eq!(i64_to_number(i32::MIN as i64), Number::I32(i32::MIN));
        // Values beyond i32 range should clamp
        assert_eq!(i64_to_number(i64::MAX), Number::I32(i32::MAX));
        assert_eq!(i64_to_number(i64::MIN), Number::I32(i32::MIN));
    }

    #[test]
    fn test_sort_iterator_ascending_empty() {
        let iter1: SortedDocIdsIter<'_> = Box::new(std::iter::empty());
        let iter2: SortedDocIdsIter<'_> = Box::new(std::iter::empty());

        let mut merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        assert!(merged.next().is_none());
    }

    #[test]
    fn test_sort_iterator_ascending_one_empty() {
        let data = [
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
        ];
        let iter1: SortedDocIdsIter<'_> = Box::new(data.iter().map(|(k, v)| (*k, v)));
        let iter2: SortedDocIdsIter<'_> = Box::new(std::iter::empty());

        let merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        let collected: Vec<_> = merged.collect();
        let keys: Vec<_> = collected.iter().map(|(k, _)| *k).collect();

        assert_eq!(keys, vec![Number::I32(1), Number::I32(3)]);
    }

    #[test]
    fn test_sort_iterator_ascending() {
        let data1 = [
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
            (Number::I32(5), HashSet::from([DocumentId(50)])),
        ];
        let data2 = [
            (Number::I32(2), HashSet::from([DocumentId(20)])),
            (Number::I32(4), HashSet::from([DocumentId(40)])),
        ];

        let iter1: SortedDocIdsIter<'_> = Box::new(data1.iter().map(|(k, v)| (*k, v)));
        let iter2: SortedDocIdsIter<'_> = Box::new(data2.iter().map(|(k, v)| (*k, v)));

        let merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        let collected: Vec<_> = merged.collect();
        let keys: Vec<_> = collected.iter().map(|(k, _)| *k).collect();

        assert_eq!(
            keys,
            vec![
                Number::I32(1),
                Number::I32(2),
                Number::I32(3),
                Number::I32(4),
                Number::I32(5)
            ]
        );
    }

    #[test]
    fn test_sort_iterator_descending() {
        let data1 = [
            (Number::I32(5), HashSet::from([DocumentId(50)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
            (Number::I32(1), HashSet::from([DocumentId(10)])),
        ];
        let data2 = [
            (Number::I32(4), HashSet::from([DocumentId(40)])),
            (Number::I32(2), HashSet::from([DocumentId(20)])),
        ];

        let iter1: SortedDocIdsIter<'_> = Box::new(data1.iter().map(|(k, v)| (*k, v)));
        let iter2: SortedDocIdsIter<'_> = Box::new(data2.iter().map(|(k, v)| (*k, v)));

        let merged = SortIterator::new(iter1, iter2, SortOrder::Descending);
        let collected: Vec<_> = merged.collect();
        let keys: Vec<_> = collected.iter().map(|(k, _)| *k).collect();

        assert_eq!(
            keys,
            vec![
                Number::I32(5),
                Number::I32(4),
                Number::I32(3),
                Number::I32(2),
                Number::I32(1)
            ]
        );
    }

    #[test]
    fn test_sort_iterator_with_duplicates() {
        let data1 = [
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
        ];
        let data2 = [
            (Number::I32(1), HashSet::from([DocumentId(11)])),
            (Number::I32(3), HashSet::from([DocumentId(31)])),
        ];

        let iter1: SortedDocIdsIter<'_> = Box::new(data1.iter().map(|(k, v)| (*k, v)));
        let iter2: SortedDocIdsIter<'_> = Box::new(data2.iter().map(|(k, v)| (*k, v)));

        let merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        let collected: Vec<_> = merged.collect();
        let keys: Vec<_> = collected.iter().map(|(k, _)| *k).collect();

        // Duplicate keys are preserved (both 1s and both 3s appear)
        assert_eq!(
            keys,
            vec![
                Number::I32(1),
                Number::I32(1),
                Number::I32(3),
                Number::I32(3)
            ]
        );
    }
}
