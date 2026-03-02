use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use oramacore_fields::number::{
    F64Storage, FilterOp, I64Storage, IndexedValue as NumberIndexedValue, SortGroupedIterator,
    SortOrder as NumberSortOrder, Threshold,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::types::{DocumentId, Number, NumberFacetDefinition, NumberFilter};

use super::committed_field::NumberFieldInfo;
use super::group::GroupValue;

/// Indexed value for the number field on the write side.
/// Separates integer and float values for dual-storage insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumberFieldIndexedValue {
    I64(NumberIndexedValue<i64>),
    F64(NumberIndexedValue<f64>),
}

/// Wrapper around dual `oramacore_fields::number` storages (I64Storage + F64Storage)
/// that provides a unified number field with built-in persistence.
/// Replaces the old UncommittedNumberField + CommittedNumberField pair.
///
/// Number::I32 values are stored as i64 in I64Storage, and Number::F32 values
/// are stored as f64 in F64Storage. This dual-storage approach handles the
/// heterogeneous nature of Number values while keeping each storage homogeneous.
pub struct NumberFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    i64_storage: I64Storage,
    f64_storage: F64Storage,
}

impl std::fmt::Debug for NumberFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NumberFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct NumberFieldStorageStats {
    pub count: usize,
    pub min: Option<Number>,
    pub max: Option<Number>,
}

impl NumberFieldStorage {
    /// Creates a new NumberFieldStorage at the given path with dual I64/F64 storages.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf) -> Result<Self> {
        let i64_path = base_path.join("i64");
        let f64_path = base_path.join("f64");

        let i64_storage = I64Storage::new(i64_path, Threshold::default())
            .context("Failed to create I64Storage for number field")?;
        let f64_storage = F64Storage::new(f64_path, Threshold::default())
            .context("Failed to create F64Storage for number field")?;

        Ok(Self {
            field_path,
            base_path,
            i64_storage,
            f64_storage,
        })
    }

    /// Loads an existing NumberFieldStorage from a NumberFieldInfo,
    /// migrating from old format if necessary.
    pub fn try_load(info: NumberFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();

        // Migration path 1: old bincode format (Vec<(SerializableNumber, HashSet<DocumentId>)>)
        let old_file = base_path.join("number_vec.bin");
        if old_file.exists() {
            use crate::types::SerializableNumber;

            let old_data: Vec<(SerializableNumber, HashSet<DocumentId>)> =
                oramacore_lib::fs::BufferedFile::open(&old_file)
                    .context("Cannot open old number_vec.bin")?
                    .read_bincode_data()
                    .context("Cannot deserialize old number_vec.bin")?;

            let i64_path = base_path.join("i64");
            let f64_path = base_path.join("f64");

            let i64_storage = I64Storage::new(i64_path, Threshold::default())
                .context("Failed to create I64Storage for number migration")?;
            let f64_storage = F64Storage::new(f64_path, Threshold::default())
                .context("Failed to create F64Storage for number migration")?;

            for (serializable_number, doc_ids) in &old_data {
                for doc_id in doc_ids {
                    match serializable_number.0 {
                        Number::I32(n) => {
                            i64_storage
                                .insert(&NumberIndexedValue::Plain(n as i64), doc_id.0)
                                .context("Failed to insert i64 during number migration")?;
                        }
                        Number::F32(f) => {
                            f64_storage
                                .insert(&NumberIndexedValue::Plain(f as f64), doc_id.0)
                                .context("Failed to insert f64 during number migration")?;
                        }
                    }
                }
            }

            i64_storage
                .compact(1)
                .context("Failed to compact i64 during number migration")?;
            f64_storage
                .compact(1)
                .context("Failed to compact f64 during number migration")?;
            i64_storage.cleanup();
            f64_storage.cleanup();

            std::fs::remove_file(&old_file)
                .context("Failed to remove old number_vec.bin after migration")?;

            info!("Migrated number field from number_vec.bin to dual I64/F64 Storage");

            return Ok(Self {
                field_path: info.field_path,
                base_path,
                i64_storage,
                f64_storage,
            });
        }

        // Migration path 2: old OrderedKeyIndex format
        let old_ordered_key = base_path.join("data");
        if old_ordered_key.exists() || base_path.join("index").exists() {
            use crate::types::SerializableNumber;
            use oramacore_lib::data_structures::ordered_key::{BoundedValue, OrderedKeyIndex};

            match OrderedKeyIndex::<SerializableNumber, DocumentId>::load(base_path.clone()) {
                Ok(inner) => {
                    let items = inner
                        .get_items(
                            (true, SerializableNumber::min_value()),
                            (true, SerializableNumber::max_value()),
                        )
                        .context("Cannot get items for number field migration")?;

                    let i64_path = base_path.join("i64");
                    let f64_path = base_path.join("f64");

                    let i64_storage = I64Storage::new(i64_path, Threshold::default())
                        .context("Failed to create I64Storage for OrderedKeyIndex migration")?;
                    let f64_storage = F64Storage::new(f64_path, Threshold::default())
                        .context("Failed to create F64Storage for OrderedKeyIndex migration")?;

                    for item in items {
                        for doc_id in &item.values {
                            match item.key.0 {
                                Number::I32(n) => {
                                    i64_storage
                                        .insert(&NumberIndexedValue::Plain(n as i64), doc_id.0)
                                        .context("Failed to insert i64 during OrderedKeyIndex migration")?;
                                }
                                Number::F32(f) => {
                                    f64_storage
                                        .insert(&NumberIndexedValue::Plain(f as f64), doc_id.0)
                                        .context("Failed to insert f64 during OrderedKeyIndex migration")?;
                                }
                            }
                        }
                    }

                    i64_storage
                        .compact(1)
                        .context("Failed to compact i64 during OrderedKeyIndex migration")?;
                    f64_storage
                        .compact(1)
                        .context("Failed to compact f64 during OrderedKeyIndex migration")?;
                    i64_storage.cleanup();
                    f64_storage.cleanup();

                    info!("Migrated number field from OrderedKeyIndex to dual I64/F64 Storage");

                    return Ok(Self {
                        field_path: info.field_path,
                        base_path,
                        i64_storage,
                        f64_storage,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load OrderedKeyIndex for number migration: {e:?}, trying native format"
                    );
                }
            }
        }

        // New format or already migrated: load both storages natively
        let i64_path = base_path.join("i64");
        let f64_path = base_path.join("f64");

        let i64_storage = I64Storage::new(i64_path, Threshold::default())
            .context("Failed to load I64Storage for number field")?;
        let f64_storage = F64Storage::new(f64_path, Threshold::default())
            .context("Failed to load F64Storage for number field")?;

        Ok(Self {
            field_path: info.field_path,
            base_path,
            i64_storage,
            f64_storage,
        })
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Inserts a document with the given Number value.
    /// Routes I32 → i64_storage, F32 → f64_storage.
    pub fn insert(&self, doc_id: DocumentId, value: Number) -> Result<()> {
        match value {
            Number::I32(n) => self
                .i64_storage
                .insert(&NumberIndexedValue::Plain(n as i64), doc_id.0)
                .context("Failed to insert i64 number value"),
            Number::F32(f) => self
                .f64_storage
                .insert(&NumberIndexedValue::Plain(f as f64), doc_id.0)
                .context("Failed to insert f64 number value"),
        }
    }

    /// Inserts a document using an already-extracted `NumberFieldIndexedValue`.
    /// This accepts both I64 and F64 variants directly from the write side.
    pub fn insert_indexed(
        &self,
        doc_id: DocumentId,
        value: &NumberFieldIndexedValue,
    ) -> Result<()> {
        match value {
            NumberFieldIndexedValue::I64(indexed) => self
                .i64_storage
                .insert(indexed, doc_id.0)
                .context("Failed to insert indexed i64 number value"),
            NumberFieldIndexedValue::F64(indexed) => self
                .f64_storage
                .insert(indexed, doc_id.0)
                .context("Failed to insert indexed f64 number value"),
        }
    }

    /// Deletes a document from both storages.
    pub fn delete(&self, doc_id: DocumentId) {
        self.i64_storage.delete(doc_id.0);
        self.f64_storage.delete(doc_id.0);
    }

    /// Returns true if either storage has pending operations that need compaction.
    pub fn has_pending_ops(&self) -> bool {
        let i64_info = self.i64_storage.info();
        let f64_info = self.f64_storage.info();
        i64_info.pending_inserts > 0
            || i64_info.pending_deletes > 0
            || f64_info.pending_inserts > 0
            || f64_info.pending_deletes > 0
    }

    /// Compacts both storages at the given version number.
    pub fn compact(&self, version: u64) -> Result<()> {
        let i64_info = self.i64_storage.info();
        if i64_info.pending_inserts > 0 || i64_info.pending_deletes > 0 {
            self.i64_storage
                .compact(version)
                .context("Failed to compact I64Storage for number field")?;
        }
        let f64_info = self.f64_storage.info();
        if f64_info.pending_inserts > 0 || f64_info.pending_deletes > 0 {
            self.f64_storage
                .compact(version)
                .context("Failed to compact F64Storage for number field")?;
        }
        Ok(())
    }

    /// Cleans up old version directories in both storages.
    pub fn cleanup(&self) {
        self.i64_storage.cleanup();
        self.f64_storage.cleanup();
    }

    /// Returns the max current offset of both storages (for promotion).
    pub fn current_offset(&self) -> u64 {
        self.i64_storage
            .current_offset()
            .max(self.f64_storage.current_offset())
    }

    // =========================================================================
    // Filter support
    // =========================================================================

    /// Filters documents by number filter, returning collected doc IDs.
    /// Queries both i64 and f64 storages and chains the results.
    pub fn filter(&self, number_filter: &NumberFilter) -> impl Iterator<Item = DocumentId> {
        let i64_op = number_filter_to_i64_filter_op(number_filter);
        let f64_op = number_filter_to_f64_filter_op(number_filter);

        // Collect from i64 storage (may be empty if the filter has no i64 interpretation)
        let i64_docs = match i64_op {
            Some(op) => Box::new(self
                .i64_storage
                .filter(op)
                .into_iter()
                .map(DocumentId)),
            None => Box::new(std::iter::empty()) as Box<dyn Iterator<Item = DocumentId>>,
        };

        // Collect from f64 storage
        let f64_docs = self
            .f64_storage
            .filter(f64_op)
            .into_iter()
            .map(DocumentId);

        i64_docs.into_iter().chain(f64_docs)
    }

    // =========================================================================
    // Sort support
    // =========================================================================

    /// Returns documents grouped by their number value, sorted.
    /// Merges results from both i64 and f64 storages.
    ///
    /// Each yielded `(Number, Vec<DocumentId>)` pair contains a unique number
    /// value and all doc IDs that share that value.
    pub fn sort_grouped(
        &self,
        ascending: bool,
    ) -> DualStorageSortMerge {
        let order = if ascending {
            NumberSortOrder::Ascending
        } else {
            NumberSortOrder::Descending
        };
        DualStorageSortMerge::new(
            self.i64_storage.sort_grouped(order),
            self.f64_storage.sort_grouped(order),
            ascending,
        )
    }

    // =========================================================================
    // Facet support
    // =========================================================================

    /// Calculates facet counts for this number field.
    /// For each range in the facet definition, counts how many documents
    /// from `token_scores` fall in that range.
    pub fn calculate_facet(
        &self,
        facet_param: &NumberFacetDefinition,
        token_scores: &HashMap<DocumentId, f32>,
    ) -> Result<HashMap<String, usize>> {
        let mut result = HashMap::with_capacity(facet_param.ranges.len());

        for range in &facet_param.ranges {
            let filter = NumberFilter::Between((range.from, range.to));
            let count = self
                .filter(&filter)
                .filter(|doc_id| token_scores.contains_key(doc_id))
                .count();

            let label = format!("{}-{}", range.from, range.to);
            result.insert(label, count);
        }

        Ok(result)
    }

    // =========================================================================
    // Group support
    // =========================================================================

    /// Returns all documents grouped by their number value.
    /// Merges results from both i64 and f64 storages.
    pub fn get_grouped_docs(&self) -> HashMap<GroupValue, HashSet<DocumentId>> {
        let mut result = HashMap::new();

        // Iterate i64 storage
        let i64_grouped = self.i64_storage.sort_grouped(NumberSortOrder::Ascending);
        for (value, doc_ids) in i64_grouped {
            let number = i64_to_number(value);
            let group_value = GroupValue::Number(number);
            let entry = result.entry(group_value).or_insert_with(HashSet::new);
            entry.extend(doc_ids.into_iter().map(DocumentId));
        }

        // Iterate f64 storage
        let f64_grouped = self.f64_storage.sort_grouped(NumberSortOrder::Ascending);
        for (value, doc_ids) in f64_grouped {
            let number = f64_to_number(value);
            let group_value = GroupValue::Number(number);
            let entry = result.entry(group_value).or_insert_with(HashSet::new);
            entry.extend(doc_ids.into_iter().map(DocumentId));
        }

        result
    }

    // =========================================================================
    // Stats support
    // =========================================================================

    /// Returns stats about this number field.
    /// Min and max are derived from `sort_grouped` (ascending for min, descending for max).
    pub fn stats(&self) -> NumberFieldStorageStats {
        let i64_info = self.i64_storage.info();
        let f64_info = self.f64_storage.info();

        let i64_count = (i64_info.header_entry_count + i64_info.pending_inserts)
            .saturating_sub(i64_info.deleted_count)
            .saturating_sub(i64_info.pending_deletes);

        let f64_count = (f64_info.header_entry_count + f64_info.pending_inserts)
            .saturating_sub(f64_info.deleted_count)
            .saturating_sub(f64_info.pending_deletes);

        let min = self.sort_grouped(true).next().map(|(n, _)| n);
        let max = self.sort_grouped(false).next().map(|(n, _)| n);

        NumberFieldStorageStats {
            count: i64_count + f64_count,
            min,
            max,
        }
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> NumberFieldInfo {
        NumberFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
        }
    }
}

// =============================================================================
// Sort merge iterator for dual i64/f64 storages
// =============================================================================

/// Iterator that merges sorted grouped results from I64Storage and F64Storage.
///
/// On each `next()`, it peeks both iterators, converts keys to `Number`,
/// compares them, and yields the smaller (ascending) or larger (descending) one.
/// When keys are numerically equal (e.g., i64 value 5 and f64 value 5.0),
/// both are merged into a single batch.
pub struct DualStorageSortMerge {
    i64_iter: std::iter::Peekable<SortGroupedIterator<i64>>,
    f64_iter: std::iter::Peekable<SortGroupedIterator<f64>>,
    ascending: bool,
}

impl DualStorageSortMerge {
    fn new(
        i64_iter: SortGroupedIterator<i64>,
        f64_iter: SortGroupedIterator<f64>,
        ascending: bool,
    ) -> Self {
        Self {
            i64_iter: i64_iter.peekable(),
            f64_iter: f64_iter.peekable(),
            ascending,
        }
    }
}

impl Iterator for DualStorageSortMerge {
    type Item = (Number, Vec<DocumentId>);

    fn next(&mut self) -> Option<Self::Item> {
        let has_i64 = self.i64_iter.peek().is_some();
        let has_f64 = self.f64_iter.peek().is_some();

        match (has_i64, has_f64) {
            (false, false) => None,
            (true, false) => {
                let (val, doc_ids) = self.i64_iter.next()?;
                Some((i64_to_number(val), doc_ids.into_iter().map(DocumentId).collect()))
            }
            (false, true) => {
                let (val, doc_ids) = self.f64_iter.next()?;
                Some((f64_to_number(val), doc_ids.into_iter().map(DocumentId).collect()))
            }
            (true, true) => {
                // Compare i64 and f64 values as f64 for ordering
                let i64_val = self.i64_iter.peek().unwrap().0;
                let f64_val = self.f64_iter.peek().unwrap().0;
                let i64_as_f64 = i64_val as f64;

                // Check if values are equal (integer maps to same float)
                if (i64_as_f64 - f64_val).abs() < f64::EPSILON {
                    // Equal: merge both into one batch
                    let (i64_v, i64_docs) = self.i64_iter.next().unwrap();
                    let (_, f64_docs) = self.f64_iter.next().unwrap();
                    let mut merged: Vec<DocumentId> = i64_docs.into_iter().map(DocumentId).collect();
                    merged.extend(f64_docs.into_iter().map(DocumentId));
                    Some((i64_to_number(i64_v), merged))
                } else {
                    let i64_is_smaller = i64_as_f64 < f64_val;
                    let take_i64 = if self.ascending {
                        i64_is_smaller
                    } else {
                        !i64_is_smaller
                    };

                    if take_i64 {
                        let (val, doc_ids) = self.i64_iter.next().unwrap();
                        Some((i64_to_number(val), doc_ids.into_iter().map(DocumentId).collect()))
                    } else {
                        let (val, doc_ids) = self.f64_iter.next().unwrap();
                        Some((f64_to_number(val), doc_ids.into_iter().map(DocumentId).collect()))
                    }
                }
            }
        }
    }
}

// =============================================================================
// Filter conversion helpers
// =============================================================================

/// Converts a NumberFilter to an i64 FilterOp.
/// Returns None if the filter has no valid i64 interpretation
/// (e.g., Equal(F32(3.14)) has no exact i64 equivalent).
fn number_filter_to_i64_filter_op(filter: &NumberFilter) -> Option<FilterOp<i64>> {
    match filter {
        NumberFilter::Equal(Number::I32(n)) => Some(FilterOp::Eq(*n as i64)),
        NumberFilter::Equal(Number::F32(f)) => {
            // Only match if the float is exactly representable as an integer
            let rounded = *f as i64;
            if (rounded as f64 - *f as f64).abs() < f64::EPSILON {
                Some(FilterOp::Eq(rounded))
            } else {
                None
            }
        }
        NumberFilter::GreaterThan(Number::I32(n)) => Some(FilterOp::Gt(*n as i64)),
        NumberFilter::GreaterThan(Number::F32(f)) => {
            // i64 values > f must be >= ceil(f)
            let ceil = (*f as f64).ceil() as i64;
            Some(FilterOp::Gte(ceil))
        }
        NumberFilter::GreaterThanOrEqual(Number::I32(n)) => Some(FilterOp::Gte(*n as i64)),
        NumberFilter::GreaterThanOrEqual(Number::F32(f)) => {
            let ceil = (*f as f64).ceil() as i64;
            Some(FilterOp::Gte(ceil))
        }
        NumberFilter::LessThan(Number::I32(n)) => Some(FilterOp::Lt(*n as i64)),
        NumberFilter::LessThan(Number::F32(f)) => {
            // i64 values < f must be <= floor(f)
            let floor = (*f as f64).floor() as i64;
            Some(FilterOp::Lte(floor))
        }
        NumberFilter::LessThanOrEqual(Number::I32(n)) => Some(FilterOp::Lte(*n as i64)),
        NumberFilter::LessThanOrEqual(Number::F32(f)) => {
            let floor = (*f as f64).floor() as i64;
            Some(FilterOp::Lte(floor))
        }
        NumberFilter::Between((min, max)) => {
            let min_i64 = match min {
                Number::I32(n) => *n as i64,
                Number::F32(f) => (*f as f64).ceil() as i64,
            };
            let max_i64 = match max {
                Number::I32(n) => *n as i64,
                Number::F32(f) => (*f as f64).floor() as i64,
            };
            if min_i64 > max_i64 {
                None // No integers in this range
            } else {
                Some(FilterOp::BetweenInclusive(min_i64, max_i64))
            }
        }
    }
}

/// Converts a NumberFilter to an f64 FilterOp.
fn number_filter_to_f64_filter_op(filter: &NumberFilter) -> FilterOp<f64> {
    match filter {
        NumberFilter::Equal(n) => FilterOp::Eq(number_to_f64(n)),
        NumberFilter::GreaterThan(n) => FilterOp::Gt(number_to_f64(n)),
        NumberFilter::GreaterThanOrEqual(n) => FilterOp::Gte(number_to_f64(n)),
        NumberFilter::LessThan(n) => FilterOp::Lt(number_to_f64(n)),
        NumberFilter::LessThanOrEqual(n) => FilterOp::Lte(number_to_f64(n)),
        NumberFilter::Between((min, max)) => {
            FilterOp::BetweenInclusive(number_to_f64(min), number_to_f64(max))
        }
    }
}

/// Converts a Number to f64.
fn number_to_f64(n: &Number) -> f64 {
    match n {
        Number::I32(i) => *i as f64,
        Number::F32(f) => *f as f64,
    }
}

/// Converts an i64 value back to a Number.
/// Uses i32 if the value fits, otherwise uses f32 approximation.
fn i64_to_number(value: i64) -> Number {
    if let Ok(i) = i32::try_from(value) {
        Number::I32(i)
    } else {
        Number::F32(value as f32)
    }
}

/// Converts an f64 value back to a Number.
fn f64_to_number(value: f64) -> Number {
    Number::F32(value as f32)
}
