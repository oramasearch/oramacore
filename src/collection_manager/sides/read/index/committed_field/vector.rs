use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use oramacore_lib::data_structures::{hnsw2::HNSW2Index, vector_bruteforce::VectorBruteForce};
use oramacore_lib::filters::FilterResult;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    collection_manager::sides::read::{
        index::{
            committed_field::offload_utils::{InnerCommittedField, LoadedField},
            uncommitted_field::UncommittedVectorField,
        },
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

const MIN_HNSW_DOCS: usize = 10_000;

pub enum VectorIndex {
    HNSW(HNSW2Index<DocumentId>),
    Plain(VectorBruteForce<DocumentId>),
}
impl VectorIndex {
    fn add(&mut self, vector: &[f32], doc_id: DocumentId) -> Result<()> {
        match self {
            VectorIndex::HNSW(index) => index.add(vector, doc_id),
            VectorIndex::Plain(index) => {
                index.add_owned(vector.to_vec(), doc_id);
                Ok(())
            }
        }
    }

    fn add_owned(&mut self, vector: Vec<f32>, doc_id: DocumentId) -> Result<()> {
        match self {
            VectorIndex::HNSW(index) => index.add_owned(vector, doc_id),
            VectorIndex::Plain(index) => {
                index.add_owned(vector, doc_id);
                Ok(())
            }
        }
    }

    fn build(&mut self) -> Result<()> {
        match self {
            VectorIndex::HNSW(index) => index.build(),
            VectorIndex::Plain(_) => Ok(()),
        }
    }
}

pub struct LoadedCommittedVectorField {
    field_path: Box<[String]>,
    inner: VectorIndex,
    data_dir: PathBuf,
    model: Model,
}

impl Debug for LoadedCommittedVectorField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner_type = match &self.inner {
            VectorIndex::HNSW(_) => "hsnw",
            VectorIndex::Plain(_) => "plain",
        };
        f.debug_struct("LoadedCommittedVectorField")
            .field("field_path", &self.field_path)
            .field("inner_type", &inner_type)
            .field("data_dir", &self.data_dir)
            .finish()
    }
}

impl CommittedVectorField {
    pub fn from_uncommitted(
        uncommitted: &UncommittedVectorField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let iter = uncommitted.iter();

        let inner = if uncommitted.len() < MIN_HNSW_DOCS {
            let mut new_index = VectorBruteForce::new(uncommitted.dimension);
            for (id, vectors) in iter {
                if uncommitted_document_deletions.contains(&id) {
                    continue;
                }

                for vector in vectors {
                    new_index.add_owned(vector, id);
                }
            }
            create_if_not_exists(&data_dir).context("Cannot create data directory")?;
            BufferedFile::create_or_overwrite(data_dir.join("index.vec"))
                .context("Cannot create vector file")?
                .write_bincode_data(&new_index)
                .context("Cannot write vector file")?;

            VectorIndex::Plain(new_index)
        } else {
            let mut new_index = HNSW2Index::new(uncommitted.dimension);
            for (id, vectors) in iter {
                if uncommitted_document_deletions.contains(&id) {
                    continue;
                }

                for vector in vectors {
                    new_index
                        .add_owned(vector, id)
                        .context("Cannot add vector")?;
                }
            }
            new_index.build().context("Cannot build hnsw index")?;

            create_if_not_exists(&data_dir).context("Cannot create data directory")?;
            BufferedFile::create_or_overwrite(data_dir.join("index.hnsw"))
                .context("Cannot create hnsw file")?
                .write_bincode_data(&new_index)
                .context("Cannot write hnsw file")?;

            VectorIndex::HNSW(new_index)
        };

        let loaded = LoadedCommittedVectorField {
            field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
            inner,
            data_dir,
            model: uncommitted.get_model(),
        };

        let inner = InnerCommittedField::new_loaded(loaded, offload_config);
        let inner = OramaSyncLock::new("vector_inner", inner);
        Ok(Self { inner })
    }

    pub fn add_uncommitted(
        &self,
        uncommitted: &UncommittedVectorField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let dim = self.get_field_info().model.dimensions();

        let total_docs = uncommitted.len() + self.stats().vector_count;
        let mut new_index = if total_docs < MIN_HNSW_DOCS {
            let mut new_inner = VectorBruteForce::new(dim);
            new_inner.set_capacity(total_docs);
            VectorIndex::Plain(new_inner)
        } else {
            let new_inner = HNSW2Index::new(dim);
            VectorIndex::HNSW(new_inner)
        };

        self.load();
        let lock = self.inner.read("commit").unwrap();
        let old = lock.get_load_unchecked().expect("already loaded");
        match &old.inner {
            VectorIndex::HNSW(index) => {
                let old_data = index.get_data();
                for (doc_id, vector) in old_data {
                    if uncommitted_document_deletions.contains(&doc_id) {
                        continue;
                    }
                    new_index.add(vector, doc_id).context("Cannot add vector")?;
                }
            }
            VectorIndex::Plain(index) => {
                let old_data = index.get_data();
                for (doc_id, vector) in old_data {
                    if uncommitted_document_deletions.contains(&doc_id) {
                        continue;
                    }
                    new_index.add(vector, doc_id).context("Cannot add vector")?;
                }
            }
        };

        for (doc_id, vectors) in uncommitted.iter() {
            if uncommitted_document_deletions.contains(&doc_id) {
                continue;
            }
            for vector in vectors {
                new_index
                    .add_owned(vector, doc_id)
                    .context("Cannot add vector")?;
            }
        }
        new_index.build().context("Cannot build hnsw index")?;

        create_if_not_exists(&data_dir).context("Cannot create data directory")?;
        match &new_index {
            VectorIndex::HNSW(index) => {
                let dump_file_path = data_dir.join("index.hnsw");
                BufferedFile::create_or_overwrite(dump_file_path)
                    .context("Cannot create hnsw file")?
                    .write_bincode_data(&index)
                    .context("Cannot write hnsw file")?;
            }
            VectorIndex::Plain(index) => {
                let dump_file_path = data_dir.join("index.vec");
                BufferedFile::create_or_overwrite(dump_file_path)
                    .context("Cannot create vector file")?
                    .write_bincode_data(&index)
                    .context("Cannot write vector file")?;
            }
        }

        Ok(Self {
            inner: OramaSyncLock::new(
                "vector_inner",
                InnerCommittedField::new_loaded(
                    LoadedCommittedVectorField {
                        field_path: self.field_path(),
                        inner: new_index,
                        data_dir,
                        model: self.get_field_info().model,
                    },
                    offload_config,
                ),
            ),
        })
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
    pub fn try_load(info: VectorFieldInfo) -> Result<Self> {
        let is_hnsw = std::fs::exists(info.data_dir.join("index.hnsw"))?;
        let inner = if is_hnsw {
            let dump_file_path = info.data_dir.join("index.hnsw");
            let inner: HNSW2Index<DocumentId> = BufferedFile::open(dump_file_path)
                .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
                .read_bincode_data()
                .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;
            VectorIndex::HNSW(inner)
        } else {
            let dump_file_path = info.data_dir.join("index.vec");
            let inner: VectorBruteForce<DocumentId> = BufferedFile::open(dump_file_path)
                .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
                .read_bincode_data()
                .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;
            VectorIndex::Plain(inner)
        };

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
        let data = match &self.inner {
            VectorIndex::HNSW(index) => index.search(target, limit),
            VectorIndex::Plain(index) => index.search(target, limit),
        };
        let scale = self.model.get_scale();

        for (doc_id, score) in data {
            if filtered_doc_ids.is_some_and(|ids| !ids.contains(&doc_id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(&doc_id) {
                continue;
            }

            // Some models need a rescale because they produce "similar" embeddings.
            // For instance, E5 models produce similarity scores that rarely go below
            // 0.7, making the effective range much narrower.
            // So, cosine similarity scores are usually in a narrow range like [0.7, 1.0],
            // instead of the full [0.0, 1.0] range.
            // This rescaling helps normalize the scores to use the full [0.0, 1.0]
            // range for better search ranking.
            let score = if let Some((min, max)) = scale {
                // Clamp the score to the expected E5 range to handle edge cases
                let clamped_score = score.clamp(min, max);
                // Rescale from [0.7, 1.0] to [0.0, 1.0]
                (clamped_score - min) / (max - min)
            } else {
                score
            };

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
        let (index_type, dim, len) = match &self.inner {
            VectorIndex::HNSW(index) => ("hnsw", index.dim(), index.len()),
            VectorIndex::Plain(index) => ("plain", index.dim(), index.len()),
        };
        CommittedVectorFieldStats {
            dimensions: dim,
            vector_count: len,
            loaded: true,
            index_type,
        }
    }

    fn info(&self) -> Self::Info {
        self.get_field_info()
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct CommittedVectorFieldStats {
    pub dimensions: usize,
    pub vector_count: usize,
    pub loaded: bool,
    pub index_type: &'static str,
}
