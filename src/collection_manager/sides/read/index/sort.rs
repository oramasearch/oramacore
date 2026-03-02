use std::collections::{HashMap, HashSet};
use std::iter::Peekable;

use crate::{
    collection_manager::sides::read::ReadError,
    types::{DocumentId, FieldId, Number, SortOrder},
};

use super::{
    bool_field::BoolFieldStorage,
    committed_field::CommittedNumberField,
    date_field::DateFieldStorage,
    path_to_index_id_map::PathToIndexId,
    uncommitted_field::UncommittedNumberField,
    CommittedFields, FieldType, UncommittedFields,
};

// =============================================================================
// DocBatch - abstraction over document ID collections in sort batches
// =============================================================================

/// A batch of document IDs used in sort operations.
///
/// Supports both `HashSet` references (for number/date fields where data is
/// already stored in HashSets) and slice references (for bool fields, avoiding
/// expensive HashSet materialization from mmap'd sorted arrays).
///
/// Using `Slice` for bool fields saves ~6x memory per element compared to
/// `HashSet` (8 bytes vs ~50+ bytes per DocumentId) and avoids hashing overhead.
#[derive(Debug)]
pub enum DocBatch<'a> {
    /// Reference to an existing HashSet (zero-copy for number/date fields).
    HashSet(&'a HashSet<DocumentId>),
    /// Reference to a contiguous slice.
    Slice(&'a [DocumentId]),
    /// Owned Vec of document IDs (used by bool fields which compute on demand).
    Owned(Vec<DocumentId>),
}

impl<'a> DocBatch<'a> {
    /// Returns an iterator over the document IDs in this batch.
    #[inline]
    pub fn iter(&self) -> DocBatchIter<'_> {
        match self {
            DocBatch::HashSet(set) => DocBatchIter::HashSet(set.iter()),
            DocBatch::Slice(slice) => DocBatchIter::Slice(slice.iter()),
            DocBatch::Owned(vec) => DocBatchIter::Slice(vec.iter()),
        }
    }

    /// Returns the number of document IDs in this batch.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            DocBatch::HashSet(set) => set.len(),
            DocBatch::Slice(slice) => slice.len(),
            DocBatch::Owned(vec) => vec.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Iterator over document IDs in a `DocBatch` (borrowing).
///
/// Wraps either a `HashSet::iter()` or a slice `iter()`, dispatching
/// via a single branch per `.next()` call (no dynamic dispatch overhead).
pub enum DocBatchIter<'a> {
    HashSet(std::collections::hash_set::Iter<'a, DocumentId>),
    Slice(std::slice::Iter<'a, DocumentId>),
}

impl<'a> Iterator for DocBatchIter<'a> {
    type Item = &'a DocumentId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DocBatchIter::HashSet(iter) => iter.next(),
            DocBatchIter::Slice(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            DocBatchIter::HashSet(iter) => iter.size_hint(),
            DocBatchIter::Slice(iter) => iter.size_hint(),
        }
    }
}

/// Owning iterator over document IDs in a `DocBatch`.
///
/// For `HashSet`/`Slice` variants, copies each `DocumentId` from the reference.
/// For `Owned`, moves `DocumentId` values out of the underlying `Vec`.
pub enum DocBatchIntoIter<'a> {
    HashSet(std::collections::hash_set::Iter<'a, DocumentId>),
    Slice(std::slice::Iter<'a, DocumentId>),
    Owned(std::vec::IntoIter<DocumentId>),
}

impl<'a> Iterator for DocBatchIntoIter<'a> {
    type Item = DocumentId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DocBatchIntoIter::HashSet(iter) => iter.next().copied(),
            DocBatchIntoIter::Slice(iter) => iter.next().copied(),
            DocBatchIntoIter::Owned(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            DocBatchIntoIter::HashSet(iter) => iter.size_hint(),
            DocBatchIntoIter::Slice(iter) => iter.size_hint(),
            DocBatchIntoIter::Owned(iter) => iter.size_hint(),
        }
    }
}

impl<'a> IntoIterator for DocBatch<'a> {
    type Item = DocumentId;
    type IntoIter = DocBatchIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            DocBatch::HashSet(set) => DocBatchIntoIter::HashSet(set.iter()),
            DocBatch::Slice(slice) => DocBatchIntoIter::Slice(slice.iter()),
            DocBatch::Owned(vec) => DocBatchIntoIter::Owned(vec.into_iter()),
        }
    }
}

// =============================================================================
// Type aliases for complex iterator types
// =============================================================================

/// Public output iterator from `IndexSortContext::execute()`.
/// Yields `(sort_key, DocBatch)` pairs, where DocBatch abstracts
/// over different document collection types (HashSet for number/date,
/// Slice for bool fields).
pub type SortedDocIdsIter<'s> = Box<dyn Iterator<Item = (Number, DocBatch<'s>)> + 's>;

/// Internal iterator type using `&HashSet` directly.
/// Used by `Sortable` trait implementations and `calculate_sort_on_field`
/// for number/date fields where data is already in HashSets.
type InternalSortIter<'s> = Box<dyn Iterator<Item = (Number, &'s HashSet<DocumentId>)> + 's>;

/// Internal bidirectional iterator for number/date fields.
/// Used by `Sortable::iter_sorted()` to support both ascending and descending order.
type InternalBidirectionalSortIter<'s> =
    Box<dyn DoubleEndedIterator<Item = (Number, &'s HashSet<DocumentId>)> + 's>;

/// Result type for sort execution, returning either a sorted iterator or a read error.
pub type SortExecuteResult<'s> = Result<SortedDocIdsIter<'s>, ReadError>;

/// A peekable boxed iterator for internal merge operations.
/// Used by SortIterator to peek at the next element without consuming it.
type PeekableSortIter<'s, T> =
    Peekable<Box<dyn Iterator<Item = (T, &'s HashSet<DocumentId>)> + 's>>;

// =============================================================================
// Sortable trait
// =============================================================================

/// Trait for fields that support sorting operations.
///
/// This trait provides a uniform interface for sorting documents across
/// different field types (Number, Date). Each implementation
/// converts its native type to `Number` for unified sorting.
///
/// # Design
///
/// The trait returns a boxed `DoubleEndedIterator` to support both ascending
/// and descending sort orders. The iterator yields tuples of (Number, &HashSet<DocumentId>)
/// where Number is the sort key. References to stored HashSets avoid cloning.
///
/// Note: Bool fields bypass this trait entirely, using `DocBatch::Slice`
/// to avoid materializing HashSets from mmap'd sorted arrays.
trait Sortable {
    /// Returns a sorted iterator over (sort_key, document_ids) pairs.
    ///
    /// The iterator yields values in ascending order by default.
    /// Use `.rev()` on the returned iterator for descending order.
    fn iter_sorted<'s>(&'s self) -> InternalBidirectionalSortIter<'s>;
}

// =============================================================================
// Sortable implementations for Uncommitted fields
// =============================================================================

impl Sortable for UncommittedNumberField {
    fn iter_sorted<'s>(&'s self) -> InternalBidirectionalSortIter<'s> {
        Box::new(self.iter_ref())
    }
}

// =============================================================================
// Sortable implementations for Committed fields
// =============================================================================

impl Sortable for CommittedNumberField {
    fn iter_sorted<'s>(&'s self) -> InternalBidirectionalSortIter<'s> {
        Box::new(
            self.iter_ref()
                .map(|(serializable_number, doc_ids)| (serializable_number.0, doc_ids)),
        )
    }
}

// =============================================================================
// IndexSortContext
// =============================================================================

/// Context required for executing sort operations on an index.
pub struct IndexSortContext<'index> {
    path_to_index_id_map: &'index PathToIndexId,
    bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
    date_fields: &'index HashMap<FieldId, DateFieldStorage>,
    uncommitted_fields: &'index UncommittedFields,
    committed_fields: &'index CommittedFields,
    field_name: String,
    order: SortOrder,
}

impl<'index> IndexSortContext<'index> {
    /// Creates a new IndexSortContext with the provided sort dependencies.
    pub(crate) fn new(
        path_to_index_id_map: &'index PathToIndexId,
        bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
        date_fields: &'index HashMap<FieldId, DateFieldStorage>,
        uncommitted_fields: &'index UncommittedFields,
        committed_fields: &'index CommittedFields,
        field_name: &str,
        order: SortOrder,
    ) -> Self {
        Self {
            path_to_index_id_map,
            bool_fields,
            date_fields,
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
                let iter = calculate_sort_on_field(uncommitted, committed, self.order);
                // Wrap &HashSet references in DocBatch::HashSet for the public API
                Ok(Box::new(iter.map(|(k, v)| (k, DocBatch::HashSet(v)))))
            }
            FieldType::Bool => {
                let bool_field = self
                    .bool_fields
                    .get(&field_id)
                    .ok_or_else(|| {
                        ReadError::Generic(anyhow::anyhow!(
                            "Field {} is not a bool field",
                            self.field_name
                        ))
                    })?;

                // Compute on demand: collect doc IDs into owned Vecs
                let false_docs: Vec<DocumentId> =
                    bool_field.filter_docs(false).into_iter().map(DocumentId).collect();
                let true_docs: Vec<DocumentId> =
                    bool_field.filter_docs(true).into_iter().map(DocumentId).collect();

                // Ascending: false (0) first, then true (1)
                let data = vec![
                    (bool_to_number(false), DocBatch::Owned(false_docs)),
                    (bool_to_number(true), DocBatch::Owned(true_docs)),
                ];

                let iter: SortedDocIdsIter<'s> = match self.order {
                    SortOrder::Ascending => Box::new(data.into_iter()),
                    SortOrder::Descending => Box::new(data.into_iter().rev()),
                };

                Ok(iter)
            }
            FieldType::Date => {
                let date_field = self
                    .date_fields
                    .get(&field_id)
                    .ok_or_else(|| {
                        ReadError::Generic(anyhow::anyhow!(
                            "Field {} is not a date field",
                            self.field_name
                        ))
                    })?;

                let ascending = matches!(self.order, SortOrder::Ascending);
                let grouped = date_field.sort_grouped(ascending);

                Ok(Box::new(grouped.map(|(timestamp, doc_ids)| {
                    let key = i64_to_number(timestamp);
                    let docs: Vec<DocumentId> =
                        doc_ids.into_iter().map(DocumentId).collect();
                    (key, DocBatch::Owned(docs))
                })))
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

/// Converts an i64 timestamp to a Number for sorting.
///
/// Uses i32::MIN and i32::MAX as NEG_INF/POS_INF
/// for values that exceed i32 range.
fn i64_to_number(value: i64) -> Number {
    if let Ok(i) = i32::try_from(value) {
        Number::I32(i)
    } else if value > 0 {
        Number::I32(i32::MAX)
    } else {
        Number::I32(i32::MIN)
    }
}

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

/// Merges sorted iterators from uncommitted and committed fields.
///
/// This generic function handles the combination of uncommitted (in-memory) and
/// committed (persisted) field data. Returns an internal iterator that yields
/// `&HashSet<DocumentId>` references - the caller wraps these in `DocBatch::HashSet`.
fn calculate_sort_on_field<'a, UF, CF>(
    uncommitted_field: &'a UF,
    committed_field: Option<&'a CF>,
    order: SortOrder,
) -> InternalSortIter<'a>
where
    UF: Sortable,
    CF: Sortable,
{
    let uncommitted_iter = uncommitted_field.iter_sorted();

    let uncommitted_iter: InternalSortIter<'a> = match order {
        SortOrder::Ascending => Box::new(uncommitted_iter),
        SortOrder::Descending => Box::new(uncommitted_iter.rev()),
    };

    if let Some(committed_field) = committed_field {
        let committed_iter = committed_field.iter_sorted();

        let committed_iter: InternalSortIter<'a> = match order {
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
    fn test_sort_iterator_ascending_empty() {
        let iter1: InternalSortIter<'_> = Box::new(std::iter::empty());
        let iter2: InternalSortIter<'_> = Box::new(std::iter::empty());

        let mut merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        assert!(merged.next().is_none());
    }

    #[test]
    fn test_sort_iterator_ascending_one_empty() {
        let data = [
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
        ];
        let iter1: InternalSortIter<'_> = Box::new(data.iter().map(|(k, v)| (*k, v)));
        let iter2: InternalSortIter<'_> = Box::new(std::iter::empty());

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

        let iter1: InternalSortIter<'_> = Box::new(data1.iter().map(|(k, v)| (*k, v)));
        let iter2: InternalSortIter<'_> = Box::new(data2.iter().map(|(k, v)| (*k, v)));

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

        let iter1: InternalSortIter<'_> = Box::new(data1.iter().map(|(k, v)| (*k, v)));
        let iter2: InternalSortIter<'_> = Box::new(data2.iter().map(|(k, v)| (*k, v)));

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

        let iter1: InternalSortIter<'_> = Box::new(data1.iter().map(|(k, v)| (*k, v)));
        let iter2: InternalSortIter<'_> = Box::new(data2.iter().map(|(k, v)| (*k, v)));

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
