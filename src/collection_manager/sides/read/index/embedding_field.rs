use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use oramacore_fields::embedding::{
    DistanceMetric, EmbeddingConfig, EmbeddingIndexer, EmbeddingStorage, SegmentConfig,
};
use oramacore_lib::{
    data_structures::{hnsw2::HNSW2Index, vector_bruteforce::VectorBruteForce},
    filters::FilterResult,
    fs::BufferedFile,
};
use serde::Serialize;
use tracing::info;

use crate::{python::embeddings::Model, types::DocumentId};

use super::committed_field::{VectorFieldInfo, VectorSearchParams};

const BRUTE_FORCE_INDEX_FILE_NAME: &str = "index.vec";
const HNSW_INDEX_FILE_NAME: &str = "index.hnsw";
const HNSW_INDEX_2_FILE_NAME: &str = "index.hnsw2";

/// Wrapper around `oramacore_fields::embedding::EmbeddingStorage` that provides
/// a unified embedding field with built-in persistence and migration support.
/// Replaces the old UncommittedVectorField + CommittedVectorField pair.
pub struct EmbeddingFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    model: Model,
    storage: EmbeddingStorage,
}

impl std::fmt::Debug for EmbeddingFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .field("model", &self.model)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct EmbeddingFieldStorageStats {
    pub dimensions: usize,
    pub vector_count: usize,
    pub model: Model,
}

/// Adapter to bridge `FilterResult<DocumentId>` with `oramacore_fields::embedding::DocumentFilter`.
struct FilterResultFilter<'a> {
    filter: &'a FilterResult<DocumentId>,
}
impl oramacore_fields::embedding::DocumentFilter for FilterResultFilter<'_> {
    fn contains(&self, doc_id: u64) -> bool {
        self.filter.contains(&DocumentId(doc_id))
    }
}

impl EmbeddingFieldStorage {
    /// Creates a new EmbeddingFieldStorage at the given path.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf, model: Model) -> Result<Self> {
        let config = EmbeddingConfig::new(model.dimensions(), DistanceMetric::Cosine)
            .context("Failed to create EmbeddingConfig")?;
        let storage = EmbeddingStorage::new(base_path.clone(), config, SegmentConfig::default())
            .context("Failed to create EmbeddingStorage")?;
        Ok(Self {
            field_path,
            base_path,
            model,
            storage,
        })
    }

    /// Loads an existing EmbeddingFieldStorage from a VectorFieldInfo,
    /// migrating from old formats (HNSW2, HNSW, BruteForce) if necessary.
    pub fn try_load(info: VectorFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();
        let model = info.model;

        let hnsw2_path = base_path.join(HNSW_INDEX_2_FILE_NAME);
        let hnsw_path = base_path.join(HNSW_INDEX_FILE_NAME);
        let brute_force_path = base_path.join(BRUTE_FORCE_INDEX_FILE_NAME);

        let config = EmbeddingConfig::new(model.dimensions(), DistanceMetric::Cosine)
            .context("Failed to create EmbeddingConfig")?;

        // Check if already migrated (CURRENT file exists and no old files remain).
        // If both new and old formats exist (crash recovery scenario), skip migration
        // since the new format data is already valid.
        let current_file = base_path.join("CURRENT");
        let has_new_format = current_file.exists();
        let has_old_files = hnsw2_path.exists() || hnsw_path.exists() || brute_force_path.exists();

        if has_new_format && has_old_files {
            // Crash recovery: new format already persisted, just clean up old files
            info!("Found both new and old embedding formats, cleaning up old files");
            Self::remove_old_files(&hnsw2_path, &hnsw_path, &brute_force_path);

            let storage =
                EmbeddingStorage::new(base_path.clone(), config, SegmentConfig::default())
                    .context("Failed to load EmbeddingStorage")?;
            return Ok(Self {
                field_path: info.field_path,
                base_path,
                model,
                storage,
            });
        }

        if has_old_files {
            // Migration path: read old format, insert into EmbeddingStorage, compact
            let storage =
                EmbeddingStorage::new(base_path.clone(), config, SegmentConfig::default())
                    .context("Failed to create EmbeddingStorage for migration")?;

            let indexer = EmbeddingIndexer::new(model.dimensions());

            if hnsw2_path.exists() {
                info!("Migrating embedding field from HNSW2 format");
                let index: HNSW2Index<DocumentId> = BufferedFile::open(&hnsw2_path)
                    .context("Cannot open HNSW2 file")?
                    .read_bincode_data()
                    .context("Cannot deserialize HNSW2 file")?;

                info!("Migrating {} embeddings from HNSW2 format", index.len());

                for (doc_id, vector) in index.into_data() {
                    if let Some(indexed_value) = indexer.index_vec(&vector) {
                        storage.insert(doc_id.0, indexed_value);
                    }
                }
            } else if hnsw_path.exists() {
                info!("Migrating embedding field from legacy HNSW format");
                let data = BufferedFile::open(&hnsw_path)
                    .context("Cannot open HNSW file")?
                    .read_as_vec()
                    .context("Cannot read HNSW file")?;
                let index: HNSW2Index<DocumentId> =
                    HNSW2Index::deserialize_bincode_compat(&data)
                        .context("Cannot deserialize legacy HNSW file")?;

                info!(
                    "Migrating {} embeddings from legacy HNSW format",
                    index.len()
                );

                for (doc_id, vector) in index.into_data() {
                    if let Some(indexed_value) = indexer.index_vec(&vector) {
                        storage.insert(doc_id.0, indexed_value);
                    }
                }
            } else if brute_force_path.exists() {
                info!("Migrating embedding field from BruteForce format");
                let index: VectorBruteForce<DocumentId> = BufferedFile::open(&brute_force_path)
                    .context("Cannot open BruteForce file")?
                    .read_bincode_data()
                    .context("Cannot deserialize BruteForce file")?;

                info!(
                    "Migrating {} embeddings from legacy BruteForce format",
                    index.len()
                );

                for (doc_id, vector) in index.into_data() {
                    if let Some(indexed_value) = indexer.index_vec(&vector) {
                        storage.insert(doc_id.0, indexed_value);
                    }
                }
            }

            // Persist migrated data
            if storage.info().pending_ops > 0 {
                storage
                    .compact(1)
                    .context("Failed to compact migrated EmbeddingStorage")?;
                storage.cleanup();
            }

            // Remove old files after successful migration
            Self::remove_old_files(&hnsw2_path, &hnsw_path, &brute_force_path);

            info!("Embedding field migration complete");

            Ok(Self {
                field_path: info.field_path,
                base_path,
                model,
                storage,
            })
        } else {
            // New format or already migrated: EmbeddingStorage loads its own CURRENT/versions/ structure
            let storage =
                EmbeddingStorage::new(base_path.clone(), config, SegmentConfig::default())
                    .context("Failed to load EmbeddingStorage")?;
            Ok(Self {
                field_path: info.field_path,
                base_path,
                model,
                storage,
            })
        }
    }

    /// Removes old legacy index files after migration.
    fn remove_old_files(hnsw2_path: &Path, hnsw_path: &Path, brute_force_path: &Path) {
        for path in [hnsw2_path, hnsw_path, brute_force_path] {
            if path.exists() {
                if let Err(e) = std::fs::remove_file(path) {
                    tracing::warn!("Failed to remove old embedding file {:?}: {}", path, e);
                }
            }
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub fn model(&self) -> Model {
        self.model
    }

    /// Inserts embeddings for a document. Multiple vectors per document are supported.
    pub fn insert(&self, doc_id: DocumentId, vectors: Vec<Vec<f32>>) {
        let indexer = EmbeddingIndexer::new(self.model.dimensions());
        if let Some(indexed_value) = indexer.index_vec_vec(&vectors) {
            self.storage.insert(doc_id.0, indexed_value);
        }
    }

    /// Deletes a document from the embedding index.
    pub fn delete(&self, doc_id: DocumentId) {
        self.storage.delete(doc_id.0);
    }

    /// Searches for nearest neighbors, converting cosine distances to similarity scores.
    ///
    /// The conversion `similarity = 1.0 - distance` is correct because:
    /// - `oramacore_fields` cosine distance = `1.0 - cosine_similarity`
    /// - So `1.0 - distance = cosine_similarity` (same value as old code)
    /// - `model.rescale_score()` is then applied identically
    pub fn search(
        &self,
        params: &VectorSearchParams<'_>,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        let results = match params.filtered_doc_ids {
            Some(filter) => {
                let doc_filter = FilterResultFilter { filter };
                self.storage
                    .search_with_filter(params.target, params.limit, None, &doc_filter)
                    .context("Cannot perform filtered embedding search")?
            }
            None => self
                .storage
                .search(params.target, params.limit, None)
                .context("Cannot perform embedding search")?,
        };

        for (doc_id, distance) in results {
            // Convert cosine distance to similarity
            let similarity = 1.0 - distance;
            let score = self.model.rescale_score(similarity);
            if score >= params.similarity {
                let v = output.entry(DocumentId(doc_id)).or_insert(0.0);
                *v += score;
            }
        }
        Ok(())
    }

    /// Returns true if there are pending operations that need compaction.
    pub fn has_pending_ops(&self) -> bool {
        self.storage.info().pending_ops > 0
    }

    /// Compacts the storage at the given version number.
    pub fn compact(&self, version: u64) -> Result<()> {
        self.storage
            .compact(version)
            .context("Failed to compact EmbeddingStorage")
    }

    /// Cleans up old version directories and unreferenced segments.
    pub fn cleanup(&self) {
        self.storage.cleanup();
    }

    /// Returns the current version number.
    pub fn current_version_number(&self) -> u64 {
        self.storage.current_version_number()
    }

    /// Returns stats about this embedding field.
    pub fn stats(&self) -> EmbeddingFieldStorageStats {
        let info = self.storage.info();
        EmbeddingFieldStorageStats {
            dimensions: info.dimensions,
            vector_count: info.num_embeddings,
            model: self.model,
        }
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
            model: self.model,
        }
    }
}
