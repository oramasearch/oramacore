use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use oramacore_fields::string::{
    ContributionsResult, DocumentFilter, IndexedValue as StringIndexedValue, SearchParams,
    SegmentConfig, StringStorage, TermData, Threshold,
};
use oramacore_lib::filters::FilterResult;
use serde::Serialize;
use tracing::info;

use crate::{
    collection_manager::sides::{InsertStringTerms, TermStringField},
    types::DocumentId,
};

use super::committed_field::StringFieldInfo;

/// Returns the SegmentConfig tuned for OramaCore's workload.
/// We raise `max_postings_per_segment` from the library default (5M) to 50M
/// because all segments are scanned on every search — fewer, larger segments
/// means fewer FST lookups per query token.
fn oramacore_segment_config() -> SegmentConfig {
    SegmentConfig {
        max_postings_per_segment: 50_000_000,
        ..Default::default()
    }
}

/// Wrapper around `oramacore_fields::string::StringStorage` that provides
/// a unified string (fulltext/BM25) field with built-in persistence.
/// Replaces the old UncommittedStringField + CommittedStringField pair.
pub struct StringFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    storage: StringStorage,
}

impl std::fmt::Debug for StringFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StringFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct StringFieldStorageStats {
    pub unique_terms_count: usize,
    pub total_documents: u64,
    pub avg_field_length: f64,
}

/// Adapter that bridges `FilterResult<DocumentId>` to `oramacore_fields::string::DocumentFilter`.
pub struct DocIdSetFilter<'a> {
    filter: &'a FilterResult<DocumentId>,
}

impl<'a> DocIdSetFilter<'a> {
    pub fn new(filter: &'a FilterResult<DocumentId>) -> Self {
        Self { filter }
    }
}

impl DocumentFilter for DocIdSetFilter<'_> {
    fn contains(&self, doc_id: u64) -> bool {
        self.filter.contains(&DocumentId(doc_id))
    }
}

impl StringFieldStorage {
    /// Creates a new StringFieldStorage at the given path.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf) -> Result<Self> {
        let storage = StringStorage::new(base_path.clone(), oramacore_segment_config())
            .context("Failed to create StringStorage")?;
        Ok(Self {
            field_path,
            base_path,
            storage,
        })
    }

    /// Loads an existing StringFieldStorage from a StringFieldInfo,
    /// migrating from old FST format if necessary.
    pub fn try_load(info: StringFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();
        let old_fst_file = base_path.join("fst.map");

        if old_fst_file.exists() {
            // Migration path: read old FST format and re-index into StringStorage.
            // We use the old CommittedStringField reader to iterate all terms+postings,
            // then insert them into the new StringStorage.
            use super::committed_field::string::load_old_fst_data;

            let storage = StringStorage::new(base_path.clone(), oramacore_segment_config())
                .context("Failed to create StringStorage for migration")?;

            let entries = load_old_fst_data(&base_path)
                .context("Failed to read old FST data for migration")?;

            for (doc_id, indexed_value) in entries {
                storage.insert(doc_id, indexed_value);
            }

            storage
                .compact(1)
                .context("Failed to compact migrated StringStorage")?;
            storage.cleanup();

            // Remove old files
            let old_posting_file = base_path.join("posting_id_storage.map");
            let old_lengths_file = base_path.join("length_per_documents.map");
            if old_fst_file.exists() {
                std::fs::remove_file(&old_fst_file)
                    .context("Failed to remove old fst.map after migration")?;
            }
            if old_posting_file.exists() {
                std::fs::remove_file(&old_posting_file)
                    .context("Failed to remove old posting_id_storage.map after migration")?;
            }
            if old_lengths_file.exists() {
                std::fs::remove_file(&old_lengths_file)
                    .context("Failed to remove old length_per_documents.map after migration")?;
            }

            info!("Migrated string field from FST format to StringStorage");

            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        } else {
            // New format or already migrated: StringStorage loads its own CURRENT/versions/ structure
            let storage = StringStorage::new(base_path.clone(), oramacore_segment_config())
                .context("Failed to load StringStorage")?;
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

    /// Inserts a document using a native `IndexedValue` from oramacore_fields.
    pub fn insert(&self, doc_id: DocumentId, value: StringIndexedValue) {
        self.storage.insert(doc_id.0, value);
    }

    /// Inserts a document using the legacy `InsertStringTerms` format.
    /// Converts old format positions (usize) to new format (u32) and builds
    /// an `IndexedValue` that `StringStorage` understands.
    pub fn insert_legacy(&self, doc_id: DocumentId, field_length: u16, terms: InsertStringTerms) {
        let mut new_terms = std::collections::HashMap::with_capacity(terms.len());
        for (term, term_field) in terms {
            let TermStringField {
                exact_positions,
                positions,
            } = term_field;
            let term_data = TermData::new(
                exact_positions.into_iter().map(|p| p as u32).collect(),
                positions.into_iter().map(|p| p as u32).collect(),
            );
            new_terms.insert(term.0, term_data);
        }
        let indexed_value = StringIndexedValue::new(field_length, new_terms);
        self.storage.insert(doc_id.0, indexed_value);
    }

    /// Deletes a document from this string field.
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
            .context("Failed to compact StringStorage")
    }

    /// Cleans up old version directories.
    pub fn cleanup(&self) {
        self.storage.cleanup();
    }

    /// Returns the current version number.
    pub fn current_version_number(&self) -> u64 {
        self.storage.current_version_number()
    }

    /// Collects raw per-token normalized TF contributions for cross-field BM25F scoring.
    /// Does not compute IDF -- that is done at the corpus level in token_score.rs.
    pub fn collect_contributions(&self, params: &SearchParams<'_>) -> Result<ContributionsResult> {
        self.storage
            .collect_contributions(params)
            .map_err(|e| anyhow::anyhow!("StringStorage collect_contributions failed: {e}"))
    }

    /// Same as `collect_contributions` but also applies a document filter.
    pub fn collect_contributions_with_filter(
        &self,
        params: &SearchParams<'_>,
        filter: &impl DocumentFilter,
    ) -> Result<ContributionsResult> {
        self.storage
            .collect_contributions_with_filter(params, filter)
            .map_err(|e| {
                anyhow::anyhow!("StringStorage collect_contributions_with_filter failed: {e}")
            })
    }

    /// Returns stats about this string field.
    pub fn stats(&self) -> StringFieldStorageStats {
        let info = self.storage.info();
        StringFieldStorageStats {
            unique_terms_count: info.unique_terms_count,
            total_documents: info.total_documents,
            avg_field_length: info.avg_field_length,
        }
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> StringFieldInfo {
        StringFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
        }
    }
}
