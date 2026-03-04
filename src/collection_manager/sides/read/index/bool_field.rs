use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use oramacore_fields::bool::{
    BoolStorage, DeletionThreshold, IndexedValue as BoolIndexedValue, SortData,
    SortOrder as BoolSortOrder,
};
use serde::Serialize;
use tracing::info;

use crate::types::{BoolFacetDefinition, DocumentId};

use super::committed_field::BoolFieldInfo;

/// Wrapper around `oramacore_fields::bool::BoolStorage` that provides
/// a unified bool field with built-in persistence.
/// Replaces the old UncommittedBoolField + CommittedBoolField pair.
pub struct BoolFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    storage: BoolStorage,
}

impl std::fmt::Debug for BoolFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoolFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct BoolFieldStorageStats {
    pub true_count: usize,
    pub false_count: usize,
}

impl BoolFieldStorage {
    /// Creates a new BoolFieldStorage at the given path.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf) -> Result<Self> {
        let storage = BoolStorage::new(base_path.clone(), DeletionThreshold::default())
            .context("Failed to create BoolStorage")?;
        Ok(Self {
            field_path,
            base_path,
            storage,
        })
    }

    /// Loads an existing BoolFieldStorage from a BoolFieldInfo,
    /// migrating from old format if necessary.
    pub fn try_load(info: BoolFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();
        let old_file = base_path.join("bool_map.bin");

        if old_file.exists() {
            // Migration path: read old bincode format (HashMap<bool, HashSet<DocumentId>>)
            let old_map: HashMap<bool, HashSet<DocumentId>> =
                bincode::deserialize_from(std::fs::File::open(&old_file)?)
                    .context("Failed to deserialize old bool_map.bin")?;

            let storage = BoolStorage::new(base_path.clone(), DeletionThreshold::default())
                .context("Failed to create BoolStorage for migration")?;

            for (bool_val, doc_ids) in &old_map {
                for doc_id in doc_ids {
                    storage.insert(&BoolIndexedValue::Plain(*bool_val), doc_id.0);
                }
            }
            storage
                .compact(1)
                .context("Failed to compact migrated BoolStorage")?;
            storage.cleanup();
            std::fs::remove_file(&old_file)
                .context("Failed to remove old bool_map.bin after migration")?;
            info!("Migrated bool field from bool_map.bin to BoolStorage");

            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        } else {
            // New format or already migrated: BoolStorage loads its own CURRENT/versions/ structure
            let storage = BoolStorage::new(base_path.clone(), DeletionThreshold::default())
                .context("Failed to load BoolStorage")?;
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

    /// Inserts a document with the given bool value.
    pub fn insert(&self, doc_id: DocumentId, value: bool) {
        self.storage
            .insert(&BoolIndexedValue::Plain(value), doc_id.0);
    }

    /// Inserts a document using an already-extracted `BoolIndexedValue`.
    /// This accepts both `Plain(bool)` and `Array(Vec<bool>)` variants,
    /// passing them directly to `BoolStorage::insert()`.
    pub fn insert_indexed(&self, doc_id: DocumentId, value: &BoolIndexedValue) {
        self.storage.insert(value, doc_id.0);
    }

    /// Deletes a document from both true and false sets.
    pub fn delete(&self, doc_id: DocumentId) {
        self.storage.delete(doc_id.0);
    }

    /// Returns true if there are pending operations that need compaction.
    pub fn has_pending_ops(&self) -> bool {
        self.storage.info().pending_ops > 0
    }

    /// Compacts the storage at the given version number.
    pub fn compact(&self, version: u64) -> Result<()> {
        self.storage
            .compact(version)
            .context("Failed to compact BoolStorage")
    }

    /// Cleans up old version directories.
    pub fn cleanup(&self) {
        self.storage.cleanup();
    }

    /// Returns stats about this bool field.
    pub fn stats(&self) -> BoolFieldStorageStats {
        let info = self.storage.info();
        BoolFieldStorageStats {
            true_count: info.true_count,
            false_count: info.false_count,
        }
    }

    /// Returns the current version number.
    pub fn current_version_number(&self) -> u64 {
        self.storage.current_version_number()
    }

    // =========================================================================
    // Filter support
    // =========================================================================

    /// Filters documents by bool value, returning collected doc IDs.
    /// We collect to Vec because FilterData borrows from a local,
    /// and the existing Filterable trait requires 'iter lifetime from &'s self.
    pub fn filter_docs(&self, value: bool) -> impl IntoIterator<Item = u64> {
        let filter_data: oramacore_fields::bool::FilterData = self.storage.filter(value);
        filter_data
    }

    // =========================================================================
    // Sort support
    // =========================================================================

    /// Returns sorted doc IDs using BoolStorage's native sort.
    pub fn sort(&self, order: BoolSortOrder) -> SortData {
        self.storage.sort(order)
    }

    // =========================================================================
    // Facet support
    // =========================================================================

    /// Calculates facet counts for this bool field.
    pub fn calculate_facet(
        &self,
        facet_param: &BoolFacetDefinition,
        token_scores: &HashMap<DocumentId, f32>,
    ) -> HashMap<String, usize> {
        let mut result = HashMap::new();

        if facet_param.r#true {
            let filter_data = self.storage.filter(true);
            let count = filter_data
                .iter()
                .filter(|doc_id| token_scores.contains_key(&DocumentId(*doc_id)))
                .count();
            result.insert("true".to_string(), count);
        }

        if facet_param.r#false {
            let filter_data = self.storage.filter(false);
            let count = filter_data
                .iter()
                .filter(|doc_id| token_scores.contains_key(&DocumentId(*doc_id)))
                .count();
            result.insert("false".to_string(), count);
        }

        result
    }

    // =========================================================================
    // Group support
    // =========================================================================

    /// Returns doc IDs grouped by bool value.
    pub fn get_grouped_docs(&self) -> HashMap<bool, HashSet<DocumentId>> {
        let mut result = HashMap::with_capacity(2);

        let true_data = self.storage.filter(true);
        let true_docs: HashSet<DocumentId> = true_data.iter().map(DocumentId).collect();
        result.insert(true, true_docs);

        let false_data = self.storage.filter(false);
        let false_docs: HashSet<DocumentId> = false_data.iter().map(DocumentId).collect();
        result.insert(false, false_docs);

        result
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> BoolFieldInfo {
        BoolFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
        }
    }
}
