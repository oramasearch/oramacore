use std::collections::{HashMap, HashSet};

use crate::{
    collection_manager::sides::read::ReadError,
    types::{DocumentId, FieldId, Number, SortOrder},
};

use super::{
    bool_field::BoolFieldStorage, date_field::DateFieldStorage, number_field::NumberFieldStorage,
    path_to_index_id_map::PathToIndexId, FieldType,
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

/// Result type for sort execution, returning either a sorted iterator or a read error.
pub type SortExecuteResult<'s> = Result<SortedDocIdsIter<'s>, ReadError>;

// =============================================================================
// IndexSortContext
// =============================================================================

/// Context required for executing sort operations on an index.
pub struct IndexSortContext<'index> {
    path_to_index_id_map: &'index PathToIndexId,
    bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
    number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
    date_fields: &'index HashMap<FieldId, DateFieldStorage>,
    field_name: String,
    order: SortOrder,
}

impl<'index> IndexSortContext<'index> {
    /// Creates a new IndexSortContext with the provided sort dependencies.
    pub(crate) fn new(
        path_to_index_id_map: &'index PathToIndexId,
        bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
        number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
        date_fields: &'index HashMap<FieldId, DateFieldStorage>,
        field_name: &str,
        order: SortOrder,
    ) -> Self {
        Self {
            path_to_index_id_map,
            bool_fields,
            number_fields,
            date_fields,
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
                let number_field = self.number_fields.get(&field_id).ok_or_else(|| {
                    ReadError::Generic(anyhow::anyhow!(
                        "Field {} is not a number field",
                        self.field_name
                    ))
                })?;
                let ascending = matches!(self.order, SortOrder::Ascending);
                let iter = number_field.sort_grouped(ascending);
                // Wrap already-collected Vec<DocumentId> in DocBatch::Owned
                Ok(Box::new(
                    iter.map(|(k, doc_ids)| (k, DocBatch::Owned(doc_ids))),
                ))
            }
            FieldType::Bool => {
                let bool_field = self.bool_fields.get(&field_id).ok_or_else(|| {
                    ReadError::Generic(anyhow::anyhow!(
                        "Field {} is not a bool field",
                        self.field_name
                    ))
                })?;

                // Compute on demand: collect doc IDs into owned Vecs
                let false_docs: Vec<DocumentId> = bool_field
                    .filter_docs(false)
                    .into_iter()
                    .map(DocumentId)
                    .collect();
                let true_docs: Vec<DocumentId> = bool_field
                    .filter_docs(true)
                    .into_iter()
                    .map(DocumentId)
                    .collect();

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
                let date_field = self.date_fields.get(&field_id).ok_or_else(|| {
                    ReadError::Generic(anyhow::anyhow!(
                        "Field {} is not a date field",
                        self.field_name
                    ))
                })?;

                let ascending = matches!(self.order, SortOrder::Ascending);
                let grouped = date_field.sort_grouped(ascending);

                Ok(Box::new(grouped.map(|(timestamp, doc_ids)| {
                    let key = i64_to_number(timestamp);
                    let docs: Vec<DocumentId> = doc_ids.into_iter().map(DocumentId).collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_to_number() {
        assert_eq!(bool_to_number(false), Number::I32(0));
        assert_eq!(bool_to_number(true), Number::I32(1));
    }
}
