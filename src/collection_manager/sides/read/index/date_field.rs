use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use oramacore_fields::number::{
    FilterOp, I64Storage, IndexedValue as NumberIndexedValue, SortGroupedIterator,
    SortOrder as NumberSortOrder, Threshold,
};
use serde::Serialize;
use tracing::info;

use crate::types::{DateFilter, DocumentId};

use super::committed_field::DateFieldInfo;

/// Wrapper around `oramacore_fields::number::I64Storage` that provides
/// a unified date field with built-in persistence.
/// Replaces the old UncommittedDateFilterField + CommittedDateField pair.
///
/// Dates are stored as i64 millisecond timestamps internally, so I64Storage
/// is a natural fit with no conversion needed at any boundary.
pub struct DateFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    storage: I64Storage,
}

impl std::fmt::Debug for DateFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DateFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct DateFieldStorageStats {
    pub count: usize,
}

impl DateFieldStorage {
    /// Creates a new DateFieldStorage at the given path.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf) -> Result<Self> {
        let storage = I64Storage::new(base_path.clone(), Threshold::default())
            .context("Failed to create I64Storage for date field")?;
        Ok(Self {
            field_path,
            base_path,
            storage,
        })
    }

    /// Loads an existing DateFieldStorage from a DateFieldInfo,
    /// migrating from old format if necessary.
    pub fn try_load(info: DateFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();
        let old_file = base_path.join("date_vec.bin");

        if old_file.exists() {
            // Migration path: read old bincode format (Vec<(i64, HashSet<DocumentId>)>)
            let old_data: Vec<(i64, HashSet<DocumentId>)> =
                oramacore_lib::fs::BufferedFile::open(&old_file)
                    .context("Cannot open old date_vec.bin")?
                    .read_bincode_data()
                    .context("Cannot deserialize old date_vec.bin")?;

            let storage = I64Storage::new(base_path.clone(), Threshold::default())
                .context("Failed to create I64Storage for date migration")?;

            for (timestamp, doc_ids) in &old_data {
                for doc_id in doc_ids {
                    storage
                        .insert(&NumberIndexedValue::Plain(*timestamp), doc_id.0)
                        .context("Failed to insert during date migration")?;
                }
            }
            storage
                .compact(1)
                .context("Failed to compact migrated I64Storage")?;
            storage.cleanup();
            std::fs::remove_file(&old_file)
                .context("Failed to remove old date_vec.bin after migration")?;
            info!("Migrated date field from date_vec.bin to I64Storage");

            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        } else {
            // Check for even older OrderedKeyIndex format
            let old_ordered_key = base_path.join("data");
            if old_ordered_key.exists() || base_path.join("index").exists() {
                // Try to migrate from OrderedKeyIndex format
                use oramacore_lib::data_structures::ordered_key::OrderedKeyIndex;

                match OrderedKeyIndex::<i64, DocumentId>::load(base_path.clone()) {
                    Ok(inner) => {
                        let items = inner
                            .get_items((true, i64::MIN), (true, i64::MAX))
                            .context("Cannot get items for date field migration")?;

                        let storage = I64Storage::new(base_path.clone(), Threshold::default())
                            .context("Failed to create I64Storage for OrderedKeyIndex migration")?;

                        for item in items {
                            for doc_id in &item.values {
                                storage
                                    .insert(&NumberIndexedValue::Plain(item.key), doc_id.0)
                                    .context("Failed to insert during OrderedKeyIndex migration")?;
                            }
                        }
                        storage.compact(1).context(
                            "Failed to compact migrated I64Storage from OrderedKeyIndex",
                        )?;
                        storage.cleanup();

                        info!("Migrated date field from OrderedKeyIndex to I64Storage");

                        return Ok(Self {
                            field_path: info.field_path,
                            base_path,
                            storage,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to load OrderedKeyIndex for date migration: {e:?}, trying native format"
                        );
                    }
                }
            }

            // New format or already migrated: I64Storage loads its own CURRENT/versions/ structure
            let storage = I64Storage::new(base_path.clone(), Threshold::default())
                .context("Failed to load I64Storage for date field")?;
            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Inserts a document with the given i64 timestamp.
    pub fn insert(&self, doc_id: DocumentId, timestamp: i64) -> Result<()> {
        self.storage
            .insert(&NumberIndexedValue::Plain(timestamp), doc_id.0)
            .context("Failed to insert date value")
    }

    /// Inserts a document using an already-extracted `NumberIndexedValue<i64>`.
    /// This accepts both `Plain(i64)` and `Array(Vec<i64>)` variants.
    pub fn insert_indexed(
        &self,
        doc_id: DocumentId,
        value: &NumberIndexedValue<i64>,
    ) -> Result<()> {
        self.storage
            .insert(value, doc_id.0)
            .context("Failed to insert indexed date value")
    }

    /// Deletes a document from the date index.
    pub fn delete(&self, doc_id: DocumentId) {
        self.storage.delete(doc_id.0);
    }

    /// Returns true if there are pending operations that need compaction.
    pub fn has_pending_ops(&self) -> bool {
        let info = self.storage.info();
        info.pending_inserts > 0 || info.pending_deletes > 0
    }

    /// Compacts the storage at the given version number.
    pub fn compact(&self, version: u64) -> Result<()> {
        self.storage
            .compact(version)
            .context("Failed to compact I64Storage for date field")
    }

    /// Cleans up old version directories.
    pub fn cleanup(&self) {
        self.storage.cleanup();
    }

    /// Returns the current version offset (for promotion).
    pub fn current_offset(&self) -> u64 {
        self.storage.current_offset()
    }

    // =========================================================================
    // Filter support
    // =========================================================================

    /// Filters documents by date filter, returning collected doc IDs.
    pub fn filter(&self, date_filter: &DateFilter) -> impl Iterator<Item = DocumentId> {
        let filter_op = date_filter_to_filter_op(date_filter);
        self.storage.filter(filter_op).into_iter().map(DocumentId)
    }

    // =========================================================================
    // Sort support
    // =========================================================================

    /// Returns sorted doc IDs using I64Storage's native sort.
    pub fn sort(&self, ascending: bool) -> impl Iterator<Item = DocumentId> {
        let order = if ascending {
            NumberSortOrder::Ascending
        } else {
            NumberSortOrder::Descending
        };
        self.storage.sort(order).into_iter().map(DocumentId)
    }

    /// Returns documents grouped by their timestamp, sorted.
    ///
    /// Each yielded `(i64, Vec<u64>)` pair contains a unique timestamp
    /// and all doc IDs that share that timestamp. This is needed for
    /// multi-index merge sort to properly interleave results by date.
    pub fn sort_grouped(&self, ascending: bool) -> SortGroupedIterator<i64> {
        let order = if ascending {
            NumberSortOrder::Ascending
        } else {
            NumberSortOrder::Descending
        };
        self.storage.sort_grouped(order)
    }

    // =========================================================================
    // Stats support
    // =========================================================================

    /// Returns stats about this date field.
    pub fn stats(&self) -> DateFieldStorageStats {
        let info = self.storage.info();
        let count = (info.header_entry_count + info.pending_inserts)
            .saturating_sub(info.deleted_count)
            .saturating_sub(info.pending_deletes);
        DateFieldStorageStats { count }
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> DateFieldInfo {
        DateFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
        }
    }
}

/// Converts a DateFilter to an I64Storage FilterOp.
fn date_filter_to_filter_op(filter: &DateFilter) -> FilterOp<i64> {
    match filter {
        DateFilter::Equal(d) => FilterOp::Eq(d.as_i64()),
        DateFilter::GreaterThan(d) => FilterOp::Gt(d.as_i64()),
        DateFilter::GreaterThanOrEqual(d) => FilterOp::Gte(d.as_i64()),
        DateFilter::LessThan(d) => FilterOp::Lt(d.as_i64()),
        DateFilter::LessThanOrEqual(d) => FilterOp::Lte(d.as_i64()),
        DateFilter::Between((d1, d2)) => FilterOp::BetweenInclusive(d1.as_i64(), d2.as_i64()),
    }
}
