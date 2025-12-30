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
/// and descending sort orders. The iterator yields tuples of (Number, HashSet<DocumentId>)
/// where Number is the sort key and HashSet contains all documents with that value.
///
/// Unlike `Filterable` or `Groupable`, this trait doesn't use an associated type
/// for the key since all implementations convert to `Number` for unified sorting.
trait Sortable {
    /// Returns a sorted iterator over (sort_key, document_ids) pairs.
    ///
    /// The iterator yields values in ascending order by default.
    /// Use `.rev()` on the returned iterator for descending order.
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's>;
}

// =============================================================================
// Sortable implementations for Uncommitted fields
// =============================================================================

impl Sortable for UncommittedNumberField {
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's> {
        Box::new(self.iter())
    }
}

impl Sortable for UncommittedDateFilterField {
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's> {
        Box::new(
            self.iter()
                .map(|(timestamp, doc_ids)| (i64_to_number(timestamp), doc_ids)),
        )
    }
}

impl Sortable for UncommittedBoolField {
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's> {
        let (true_docs, false_docs) = self.clone_inner();
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
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's> {
        Box::new(
            self.iter()
                .map(|(serializable_number, doc_ids)| (serializable_number.0, doc_ids)),
        )
    }
}

impl Sortable for CommittedDateField {
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's> {
        Box::new(
            self.iter()
                .map(|(timestamp, doc_ids)| (i64_to_number(timestamp), doc_ids)),
        )
    }
}

impl Sortable for CommittedBoolField {
    fn iter_sorted<'s>(
        &'s self,
    ) -> Box<dyn DoubleEndedIterator<Item = (Number, HashSet<DocumentId>)> + 's> {
        // clone_inner returns Result, but we can safely unwrap here
        // since the bool field is always properly initialized
        let (true_docs, false_docs) = self.clone_inner().expect("BoolField should be valid");
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
///
/// This struct encapsulates all the data needed to sort documents,
/// allowing sort logic to be executed without direct Index coupling.
/// Index is responsible for gathering the data and passing it to the
/// IndexSortContext constructor, maintaining proper encapsulation.
///
/// Validation of field existence and type is deferred to the `execute` method,
/// allowing construction to be infallible.
pub struct IndexSortContext<'index> {
    path_to_index_id_map: &'index PathToIndexId,
    uncommitted_fields: &'index UncommittedFields,
    committed_fields: &'index CommittedFields,
    field_name: String,
    order: SortOrder,
}

impl<'index> IndexSortContext<'index> {
    /// Creates a new IndexSortContext with the provided sort dependencies.
    ///
    /// This constructor stores the parameters without validation. Validation
    /// of field existence and type is deferred to the `execute` method.
    ///
    /// # Arguments
    ///
    /// * `path_to_index_id_map` - Map from field paths to (FieldId, FieldType)
    /// * `uncommitted_fields` - Read guard for in-memory uncommitted fields
    /// * `committed_fields` - Read guard for persisted committed fields
    /// * `field_name` - The name of the field to sort by
    /// * `order` - The sort order (ascending or descending)
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
    /// It validates the field exists and is sortable, then dispatches to the
    /// appropriate field type and returns a merged, sorted iterator combining
    /// uncommitted and committed data.
    ///
    /// Note: Returns a borrowing iterator, so context cannot be consumed.
    ///
    /// # Returns
    ///
    /// A boxed iterator yielding (Number, HashSet<DocumentId>) tuples in sorted order.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The field is not found in the index
    /// - The field type is not sortable (only Bool, Number, Date are supported)
    /// - The field exists in the map but not in the uncommitted fields
    pub fn execute<'s>(
        &'s self,
    ) -> Result<Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + 's>, ReadError>
    where
        'index: 's,
    {
        // Look up the field in the index
        let (field_id, field_type) = self
            .path_to_index_id_map
            .get_filter_field(&self.field_name)
            .ok_or_else(|| ReadError::SortFieldNotFound(self.field_name.clone()))?;

        // Validate field type, get fields, and dispatch to appropriate sort implementation
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
/// committed (persisted) field data. It uses the `Sortable` trait for compile-time
/// type safety.
///
/// # Type Parameters
/// * `UF` - Uncommitted field type implementing Sortable
/// * `CF` - Committed field type implementing Sortable
///
/// # Arguments
/// * `uncommitted_field` - The in-memory uncommitted field
/// * `committed_field` - Optional persisted committed field
/// * `order` - The sort order (ascending or descending)
///
/// # Returns
/// A boxed iterator yielding (Number, HashSet<DocumentId>) tuples in sorted order.
fn calculate_sort_on_field<'a, UF, CF>(
    uncommitted_field: &'a UF,
    committed_field: Option<&'a CF>,
    order: SortOrder,
) -> Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + 'a>
where
    UF: Sortable,
    CF: Sortable,
{
    let uncommitted_iter = uncommitted_field.iter_sorted();

    // Apply sort order to uncommitted iterator
    let uncommitted_iter: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + 'a> = match order
    {
        SortOrder::Ascending => Box::new(uncommitted_iter),
        SortOrder::Descending => Box::new(uncommitted_iter.rev()),
    };

    // If there's a committed field, merge the two iterators
    if let Some(committed_field) = committed_field {
        let committed_iter = committed_field.iter_sorted();

        let committed_iter: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + 'a> =
            match order {
                SortOrder::Ascending => Box::new(committed_iter),
                SortOrder::Descending => Box::new(committed_iter.rev()),
            };

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
///
/// # Type Parameters
/// * `T` - The key type, must implement `Ord + Clone`
struct SortIterator<'s1, 's2, T: Ord + Clone> {
    iter1: Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's1>>,
    iter2: Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's2>>,
    order: SortOrder,
}

impl<'s1, 's2, T: Ord + Clone> SortIterator<'s1, 's2, T> {
    /// Creates a new SortIterator from two iterators.
    ///
    /// Both input iterators must already be sorted according to the specified order.
    fn new(
        iter1: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's1>,
        iter2: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's2>,
        order: SortOrder,
    ) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            order,
        }
    }
}

impl<T: Ord + Clone> Iterator for SortIterator<'_, '_, T> {
    type Item = (T, HashSet<DocumentId>);

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

// =============================================================================
// Tests
// =============================================================================

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
        let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(std::iter::empty());
        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(std::iter::empty());

        let mut merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        assert!(merged.next().is_none());
    }

    #[test]
    fn test_sort_iterator_ascending_one_empty() {
        let data = vec![
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
        ];
        let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data.into_iter());
        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(std::iter::empty());

        let merged = SortIterator::new(iter1, iter2, SortOrder::Ascending);
        let collected: Vec<_> = merged.collect();
        let keys: Vec<_> = collected.iter().map(|(k, _)| *k).collect();

        assert_eq!(keys, vec![Number::I32(1), Number::I32(3)]);
    }

    #[test]
    fn test_sort_iterator_ascending() {
        let data1 = vec![
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
            (Number::I32(5), HashSet::from([DocumentId(50)])),
        ];
        let data2 = vec![
            (Number::I32(2), HashSet::from([DocumentId(20)])),
            (Number::I32(4), HashSet::from([DocumentId(40)])),
        ];

        let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data1.into_iter());
        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data2.into_iter());

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
        let data1 = vec![
            (Number::I32(5), HashSet::from([DocumentId(50)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
            (Number::I32(1), HashSet::from([DocumentId(10)])),
        ];
        let data2 = vec![
            (Number::I32(4), HashSet::from([DocumentId(40)])),
            (Number::I32(2), HashSet::from([DocumentId(20)])),
        ];

        let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data1.into_iter());
        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data2.into_iter());

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
        let data1 = vec![
            (Number::I32(1), HashSet::from([DocumentId(10)])),
            (Number::I32(3), HashSet::from([DocumentId(30)])),
        ];
        let data2 = vec![
            (Number::I32(1), HashSet::from([DocumentId(11)])),
            (Number::I32(3), HashSet::from([DocumentId(31)])),
        ];

        let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data1.into_iter());
        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)>> =
            Box::new(data2.into_iter());

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
