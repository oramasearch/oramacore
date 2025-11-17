use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use oramacore_lib::data_structures::hnsw2::HNSW2Index;
use oramacore_lib::filters::FilterResult;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    collection_manager::sides::read::{
        index::committed_field::offload_utils::{InnerCommittedField, LoadedField},
        OffloadFieldConfig,
    },
    lock::OramaSyncLock,
    python::embeddings::Model,
    types::DocumentId,
};
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};

#[derive(Debug)]
pub struct CommittedVectorField {
    inner: OramaSyncLock<
        InnerCommittedField<LoadedCommittedVectorField, CommittedVectorFieldStats, VectorFieldInfo>,
    >,
}

pub struct LoadedCommittedVectorField {
    field_path: Box<[String]>,
    inner: HNSW2Index<DocumentId>,
    data_dir: PathBuf,
    model: Model,
}

impl Debug for LoadedCommittedVectorField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = serde_json::to_string(&self.inner).unwrap_or_default();
        f.debug_struct("LoadedCommittedVectorField")
            .field("field_path", &self.field_path)
            .field("inner", &inner)
            .field("data_dir", &self.data_dir)
            .finish()
    }
}

impl CommittedVectorField {
    pub fn from_iter<I>(
        field_path: Box<[String]>,
        iter: I,
        model: Model,
        data_dir: PathBuf,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let loaded = LoadedCommittedVectorField::from_iter(field_path, iter, model, data_dir)?;

        let inner = InnerCommittedField::new_loaded(loaded, offload_config);
        let inner = OramaSyncLock::new("vector_inner", inner);
        Ok(Self { inner })
    }

    pub fn from_dump_and_iter(
        field_path: Box<[String]>,
        data_dir: PathBuf,
        iter: impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        model: Model,
        new_data_dir: PathBuf,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let loaded = LoadedCommittedVectorField::from_dump_and_iter(
            field_path,
            data_dir,
            iter,
            uncommitted_document_deletions,
            model,
            new_data_dir,
        )?;

        let inner = InnerCommittedField::new_loaded(loaded, offload_config);
        let inner = OramaSyncLock::new("vector_inner", inner);
        Ok(Self { inner })
    }

    pub fn try_load(info: VectorFieldInfo, offload_config: OffloadFieldConfig) -> Result<Self> {
        let loaded = LoadedCommittedVectorField::try_load(info)?;
        let inner = InnerCommittedField::new_loaded(loaded, offload_config);
        let inner = OramaSyncLock::new("vector_inner", inner);
        Ok(Self { inner })
    }

    fn loaded(&self) -> bool {
        self.inner.read("loaded").unwrap().loaded()
    }

    fn load(&self) {
        if self.loaded() {
            return;
        }

        let mut inner = self.inner.write("load").unwrap();
        if let InnerCommittedField::Unloaded {
            offload_config,
            field_info,
            ..
        } = &**inner
        {
            let loaded = LoadedCommittedVectorField::try_load(field_info.clone())
                .expect("Cannot load committed vector field");
            **inner = InnerCommittedField::new_loaded(loaded, *offload_config)
        }
    }

    pub fn unload_if_not_used(&self) {
        let lock = self.inner.read("unload_if_not_used").unwrap();
        if lock.should_unload() {
            drop(lock); // Release the read lock before unloading
            self.unload();
        }
    }

    fn unload(&self) {
        let mut inner = self.inner.write("unload").unwrap();
        if let InnerCommittedField::Loaded {
            field,
            offload_config,
            ..
        } = &**inner
        {
            let field_path = field.field_path();
            debug!("Unloading committed vector field {:?}", field_path,);
            let mut stats = field.stats();
            stats.loaded = false; // Mark field as unloaded
            **inner = InnerCommittedField::unloaded(*offload_config, stats, field.info());

            info!("Committed vector field {:?} unloaded", field_path,);
        }
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        self.inner.read("get_field_info").unwrap().info()
    }

    pub fn field_path(&self) -> Box<[String]> {
        self.inner
            .read("field_path")
            .unwrap()
            .info()
            .field_path
            .clone()
    }

    pub fn search(
        &self,
        target: &[f32],
        similarity: f32,
        limit: usize,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        self.load();

        let inner = self.inner.read("search").unwrap();

        if let Some(field) = inner.get_load_unchecked() {
            field.search(
                target,
                similarity,
                limit,
                filtered_doc_ids,
                output,
                uncommitted_deleted_documents,
            )
        } else {
            Ok(())
        }
    }

    pub fn stats(&self) -> CommittedVectorFieldStats {
        let inner = self.inner.read("stats").unwrap();
        inner.stats()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
    pub model: Model,
}

impl LoadedCommittedVectorField {
    pub fn from_iter<I>(
        field_path: Box<[String]>,
        iter: I,
        model: Model,
        data_dir: PathBuf,
    ) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let mut inner = HNSW2Index::new(model.dimensions());
        for (doc_id, vectors) in iter {
            for vector in vectors {
                inner
                    .add_owned(vector, doc_id)
                    .context("Cannot add vector")?;
            }
        }
        inner.build().context("Cannot build hnsw index")?;

        create_if_not_exists(&data_dir).context("Cannot create data directory")?;
        BufferedFile::create_or_overwrite(data_dir.join("index.hnsw"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&inner)
            .context("Cannot write hnsw file")?;

        Ok(Self {
            field_path,
            inner,
            data_dir,
            model,
        })
    }

    pub fn from_dump_and_iter(
        field_path: Box<[String]>,
        data_dir: PathBuf,
        iter: impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        model: Model,
        new_data_dir: PathBuf,
    ) -> Result<Self> {
        let dump_file_path = data_dir.join("index.hnsw");

        let inner: HNSW2Index<DocumentId> = BufferedFile::open(dump_file_path)
            .context("Cannot open hnsw file")?
            .read_bincode_data()
            .context("Cannot read hnsw file")?;
        let dim = inner.dim();

        let iter = iter
            .flat_map(|(doc_id, vectors)| vectors.into_iter().map(move |vector| (doc_id, vector)))
            .chain(inner.into_data())
            .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id));

        let mut new_inner = HNSW2Index::new(dim);

        for (doc_id, vector) in iter {
            new_inner
                .add(&vector, doc_id)
                .context("Cannot add vector")?;
        }
        new_inner.build().context("Cannot build hnsw index")?;

        create_if_not_exists(&new_data_dir).context("Cannot create data directory")?;
        BufferedFile::create_or_overwrite(new_data_dir.join("index.hnsw"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&new_inner)
            .context("Cannot write hnsw file")?;

        Ok(Self {
            field_path,
            inner: new_inner,
            model,
            data_dir: new_data_dir,
        })
    }

    pub fn try_load(info: VectorFieldInfo) -> Result<Self> {
        let dump_file_path = info.data_dir.join("index.hnsw");

        let inner: HNSW2Index<DocumentId> = BufferedFile::open(dump_file_path)
            .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;

        Ok(Self {
            field_path: info.field_path,
            inner,
            data_dir: info.data_dir,
            model: info.model,
        })
    }

    pub fn field_path(&self) -> Box<[String]> {
        self.field_path.clone()
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
            model: self.model,
        }
    }

    fn is_e5_model(&self) -> bool {
        matches!(
            self.model,
            Model::MultilingualE5Small | Model::MultilingualE5Base | Model::MultilingualE5Large
        )
    }

    pub fn search(
        &self,
        target: &[f32],
        similarity: f32,
        limit: usize,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        // We filtered matches by:
        // - `filtered_doc_ids`: removed by `search` method
        // - `uncommitted_deleted_documents`: removed by `document deletion` method
        // - `similarity` threshold: removed by `search` method
        // If there're not uncomitted deletions or the user doesn't filter, the limit is ok:
        // HNSW returns the most probable matches first, so we can stop when we reach the limit.
        // Otherwise, we should continue the search till reach the limit.
        // Anyway, the implementation below returns a Vec, so we should redo the search till reach the limit.
        // For now, we just double the limit.
        // TODO: implement a better way to handle this.
        let limit = if filtered_doc_ids.is_none() && uncommitted_deleted_documents.is_empty() {
            limit
        } else {
            limit * 2
        };
        let data = self.inner.search(target.to_vec(), limit * 2);
        let is_e5_model = self.is_e5_model();

        for (doc_id, mut score) in data {
            if filtered_doc_ids.is_some_and(|ids| !ids.contains(&doc_id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(&doc_id) {
                continue;
            }

            // Rescale E5 model scores from [0.7, 1.0] to [0.0, 1.0]
            if is_e5_model {
                score = rescale_e5_similarity_score(score);
            }

            if score >= similarity {
                let v = output.entry(doc_id).or_insert(0.0);
                *v += score;
            }
        }

        Ok(())
    }
}

impl LoadedField for LoadedCommittedVectorField {
    type Stats = CommittedVectorFieldStats;
    type Info = VectorFieldInfo;

    fn stats(&self) -> Self::Stats {
        CommittedVectorFieldStats {
            dimensions: self.inner.dim(),
            vector_count: self.inner.len(),
            loaded: true,
        }
    }

    fn info(&self) -> Self::Info {
        self.get_field_info()
    }
}

/// Rescales E5 embedding model cosine similarity scores from their typical range [0.7, 1.0] to [0.0, 1.0].
/// E5 models produce similarity scores that rarely go below 0.7, making the effective range much narrower.
/// This rescaling helps normalize the scores to use the full [0.0, 1.0] range for better search ranking.
fn rescale_e5_similarity_score(score: f32) -> f32 {
    const E5_MIN_SCORE: f32 = 0.7;
    const E5_MAX_SCORE: f32 = 1.0;

    // Clamp the score to the expected E5 range to handle edge cases
    let clamped_score = score.clamp(E5_MIN_SCORE, E5_MAX_SCORE);

    // Rescale from [0.7, 1.0] to [0.0, 1.0]
    (clamped_score - E5_MIN_SCORE) / (E5_MAX_SCORE - E5_MIN_SCORE)
}

#[derive(Serialize, Debug, Clone)]
pub struct CommittedVectorFieldStats {
    pub dimensions: usize,
    pub vector_count: usize,
    pub loaded: bool,
}
