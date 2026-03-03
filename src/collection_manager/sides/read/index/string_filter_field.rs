use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use oramacore_fields::string_filter::{
    IndexedValue as StringFilterIndexedValue, StringFilterStorage, Threshold,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::types::{DocumentId, StringFacetDefinition};

use super::committed_field::StringFilterFieldInfo;

/// Wrapper around `oramacore_fields::string_filter::StringFilterStorage` that provides
/// a unified string_filter field with built-in persistence.
/// Replaces the old UncommittedStringFilterField + CommittedStringFilterField pair.
pub struct StringFilterFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    storage: StringFilterStorage,
}

impl std::fmt::Debug for StringFilterFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringFilterFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct StringFilterFieldStorageStats {
    pub key_count: usize,
    pub document_count: usize,
    pub keys: Option<Vec<String>>,
}

// Types for deserializing the old format during migration.
#[derive(Deserialize)]
enum StringFilterFieldDump {
    V1(StringFilterFieldDumpV1),
}

#[derive(Deserialize)]
struct StringFilterFieldDumpV1 {
    data: HashMap<String, HashSet<DocumentId>>,
}

impl StringFilterFieldStorage {
    /// Creates a new StringFilterFieldStorage at the given path.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf) -> Result<Self> {
        let storage = StringFilterStorage::new(base_path.clone(), Threshold::default())
            .context("Failed to create StringFilterStorage")?;
        Ok(Self {
            field_path,
            base_path,
            storage,
        })
    }

    /// Loads an existing StringFilterFieldStorage from a StringFilterFieldInfo,
    /// migrating from old format if necessary.
    pub fn try_load(info: StringFilterFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();
        let old_file = base_path.join("data.bin");

        if old_file.exists() {
            // Migration path: read old bincode format (HashMap<String, HashSet<DocumentId>>)
            let old_data: StringFilterFieldDump = oramacore_lib::fs::BufferedFile::open(&old_file)
                .context("Cannot open old data.bin")?
                .read_bincode_data()
                .context("Cannot deserialize old data.bin")?;
            let StringFilterFieldDump::V1(old_data) = old_data;

            let storage = StringFilterStorage::new(base_path.clone(), Threshold::default())
                .context("Failed to create StringFilterStorage for migration")?;

            for (key, doc_ids) in &old_data.data {
                for doc_id in doc_ids {
                    storage.insert(&StringFilterIndexedValue::Plain(key.clone()), doc_id.0);
                }
            }
            storage
                .compact(1)
                .context("Failed to compact migrated StringFilterStorage")?;
            storage.cleanup();
            std::fs::remove_file(&old_file)
                .context("Failed to remove old data.bin after migration")?;
            info!("Migrated string_filter field from data.bin to StringFilterStorage");

            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        } else {
            // New format or already migrated: StringFilterStorage loads its own CURRENT/versions/ structure
            let storage = StringFilterStorage::new(base_path.clone(), Threshold::default())
                .context("Failed to load StringFilterStorage")?;
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

    /// Inserts a document with the given string value (legacy path).
    pub fn insert(&self, doc_id: DocumentId, value: String) {
        self.storage
            .insert(&StringFilterIndexedValue::Plain(value), doc_id.0);
    }

    /// Inserts a document using an already-extracted `StringFilterIndexedValue`.
    /// This accepts both `Plain(String)` and `Array(Vec<String>)` variants,
    /// passing them directly to `StringFilterStorage::insert()`.
    pub fn insert_indexed(&self, doc_id: DocumentId, value: &StringFilterIndexedValue) {
        self.storage.insert(value, doc_id.0);
    }

    /// Deletes a document from the string filter index.
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
            .context("Failed to compact StringFilterStorage")
    }

    /// Cleans up old version directories.
    pub fn cleanup(&self) {
        self.storage.cleanup();
    }

    /// Returns the current version number.
    pub fn current_version_number(&self) -> u64 {
        self.storage.current_version_number()
    }

    // =========================================================================
    // Filter support
    // =========================================================================

    /// Filters documents by string value, returning collected doc IDs.
    pub fn filter_docs(&self, value: &str) -> Vec<DocumentId> {
        self.storage.filter(value).iter().map(DocumentId).collect()
    }

    // =========================================================================
    // Facet support
    // =========================================================================

    /// Calculates facet counts for this string_filter field.
    /// Discovers all unique string keys and counts matching documents.
    pub fn calculate_facet(
        &self,
        _facet_param: &StringFacetDefinition,
        token_scores: &HashMap<DocumentId, f32>,
    ) -> HashMap<String, usize> {
        let keys = self.storage.keys();
        let mut result = HashMap::with_capacity(keys.len());

        for key in keys {
            let filter_data = self.storage.filter(&key);
            let count = filter_data
                .iter()
                .filter(|doc_id| token_scores.contains_key(&DocumentId(*doc_id)))
                .count();
            result.insert(key, count);
        }

        result
    }

    // =========================================================================
    // Group support
    // =========================================================================

    /// Returns doc IDs grouped by string value.
    pub fn get_grouped_docs(&self) -> HashMap<String, HashSet<DocumentId>> {
        let keys = self.storage.keys();
        let mut result = HashMap::with_capacity(keys.len());

        for key in keys {
            let filter_data = self.storage.filter(&key);
            let doc_ids: HashSet<DocumentId> = filter_data.iter().map(DocumentId).collect();
            result.insert(key, doc_ids);
        }

        result
    }

    // =========================================================================
    // Stats support
    // =========================================================================

    /// Returns stats about this string_filter field.
    pub fn stats(&self) -> StringFilterFieldStorageStats {
        let info = self.storage.info();
        let keys = self.storage.keys();
        let doc_count =
            info.total_postings_count.saturating_sub(info.deleted_count) + info.pending_ops;
        StringFilterFieldStorageStats {
            key_count: keys.len(),
            document_count: doc_count,
            keys: Some(keys),
        }
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> StringFilterFieldInfo {
        StringFilterFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
        }
    }
}
