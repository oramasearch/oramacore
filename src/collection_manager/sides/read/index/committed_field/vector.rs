use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
    sync::RwLock,
    time::UNIX_EPOCH,
};

use anyhow::{anyhow, Context, Result};
use filters::FilterResult;
use invocation_counter::InvocationCounter;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    ai::OramaModel,
    collection_manager::sides::{read::OffloadFieldConfig, write::OramaModelSerializable},
    indexes::hnsw2::HNSW2Index,
    types::DocumentId,
};
use fs::{create_if_not_exists, BufferedFile};

#[derive(Debug)]
pub enum InnerCommittedVectorField {
    Loaded(LoadedCommittedVectorField),
    Unloaded(UnloadedCommittedVectorField),
}

#[derive(Debug)]
pub struct CommittedVectorField {
    inner: RwLock<InnerCommittedVectorField>,
    offload_config: OffloadFieldConfig,
}

#[derive(Debug)]
pub struct UnloadedCommittedVectorField {
    field_path: Box<[String]>,
    data_dir: PathBuf,
    model: OramaModel,
    stats: CommittedVectorFieldStats,
}

pub struct LoadedCommittedVectorField {
    field_path: Box<[String]>,
    inner: HNSW2Index,
    data_dir: PathBuf,
    model: OramaModel,
    counter: InvocationCounter,
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
        model: OramaModel,
        data_dir: PathBuf,
        offload_config: &OffloadFieldConfig,
    ) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let loaded = LoadedCommittedVectorField::from_iter(
            field_path,
            iter,
            model,
            data_dir,
            offload_config,
        )?;

        Ok(Self {
            inner: RwLock::new(InnerCommittedVectorField::Loaded(loaded)),
            offload_config: offload_config.clone(),
        })
    }

    pub fn from_dump_and_iter(
        field_path: Box<[String]>,
        data_dir: PathBuf,
        iter: impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        model: OramaModel,
        new_data_dir: PathBuf,
        offload_config: &OffloadFieldConfig,
    ) -> Result<Self> {
        let loaded = LoadedCommittedVectorField::from_dump_and_iter(
            field_path,
            data_dir,
            iter,
            uncommitted_document_deletions,
            model,
            new_data_dir,
            offload_config,
        )?;

        Ok(Self {
            inner: RwLock::new(InnerCommittedVectorField::Loaded(loaded)),
            offload_config: offload_config.clone(),
        })
    }

    pub fn try_load(info: VectorFieldInfo, offload_config: &OffloadFieldConfig) -> Result<Self> {
        let loaded = LoadedCommittedVectorField::try_load(info, offload_config)?;
        Ok(Self {
            inner: RwLock::new(InnerCommittedVectorField::Loaded(loaded)),
            offload_config: offload_config.clone(),
        })
    }

    fn loaded(&self) -> bool {
        matches!(
            *self.inner.read().unwrap(),
            InnerCommittedVectorField::Loaded(_)
        )
    }

    fn load(&self) {
        let mut inner = self.inner.write().unwrap();
        if let InnerCommittedVectorField::Unloaded(unloaded) = &*inner {
            let loaded = LoadedCommittedVectorField::try_load(
                unloaded.get_field_info(),
                &self.offload_config,
            )
            .expect("Cannot load committed vector field");
            *inner = InnerCommittedVectorField::Loaded(loaded);
        }
    }

    pub fn unload_if_not_used(&self) {
        let lock = self.inner.read().unwrap();
        let loaded = match &*lock {
            InnerCommittedVectorField::Loaded(loaded) => loaded,
            InnerCommittedVectorField::Unloaded(_) => {
                return;
            }
        };

        let now = now();
        let start = now - self.offload_config.unload_window.as_secs();
        let c = loaded.counter.count_in(start, now);
        if c != 0 {
            return;
        }

        drop(lock);
        self.unload();
    }

    fn unload(&self) {
        let mut inner = self.inner.write().unwrap();
        if let InnerCommittedVectorField::Loaded(loaded) = &*inner {
            let field_path = loaded.field_path();
            debug!("Unloading committed vector field {:?}", field_path);
            let stats = loaded.stats().expect("Cannot get stats during unload");
            let unloaded = UnloadedCommittedVectorField {
                field_path: loaded.field_path.clone(),
                data_dir: loaded.data_dir.clone(),
                model: loaded.model,
                stats,
            };
            *inner = InnerCommittedVectorField::Unloaded(unloaded);

            info!("Committed vector field {:?} unloaded", field_path);
        }
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        let inner = self.inner.read().unwrap();
        match &*inner {
            InnerCommittedVectorField::Loaded(loaded) => loaded.get_field_info(),
            InnerCommittedVectorField::Unloaded(unloaded) => VectorFieldInfo {
                field_path: unloaded.field_path.clone(),
                data_dir: unloaded.data_dir.clone(),
                model: OramaModelSerializable(unloaded.model),
            },
        }
    }

    pub fn field_path(&self) -> Box<[String]> {
        let inner = self.inner.read().unwrap();
        match &*inner {
            InnerCommittedVectorField::Loaded(loaded) => loaded.field_path().clone(),
            InnerCommittedVectorField::Unloaded(unloaded) => unloaded.field_path.clone(),
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
        if !self.loaded() {
            self.load();
        }

        let inner = self.inner.read().unwrap();
        match &*inner {
            InnerCommittedVectorField::Loaded(loaded) => loaded.search(
                target,
                similarity,
                limit,
                filtered_doc_ids,
                output,
                uncommitted_deleted_documents,
            ),
            InnerCommittedVectorField::Unloaded(_) => Ok(()),
        }
    }

    pub fn stats(&self) -> Result<CommittedVectorFieldStats> {
        let inner = self.inner.read().unwrap();
        match &*inner {
            InnerCommittedVectorField::Loaded(loaded) => loaded.stats(),
            InnerCommittedVectorField::Unloaded(unloaded) => Ok(unloaded.get_stats()),
        }
    }
}

impl UnloadedCommittedVectorField {
    pub fn get_field_info(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
            model: OramaModelSerializable(self.model),
        }
    }

    pub fn get_stats(&self) -> CommittedVectorFieldStats {
        CommittedVectorFieldStats {
            dimensions: self.stats.dimensions,
            vector_count: self.stats.vector_count,
            loaded: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
    pub model: OramaModelSerializable,
}

impl LoadedCommittedVectorField {
    pub fn from_iter<I>(
        field_path: Box<[String]>,
        iter: I,
        model: OramaModel,
        data_dir: PathBuf,
        offload_config: &OffloadFieldConfig,
    ) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let mut inner = HNSW2Index::new(model.dimensions());
        for (doc_id, vectors) in iter {
            for vector in vectors {
                inner.add(&vector, doc_id).context("Cannot add vector")?;
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
            counter: create_counter(offload_config.slot_count_exp, offload_config.slot_size_exp),
        })
    }

    pub fn from_dump_and_iter(
        field_path: Box<[String]>,
        data_dir: PathBuf,
        iter: impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        model: OramaModel,
        new_data_dir: PathBuf,
        offload_config: &OffloadFieldConfig,
    ) -> Result<Self> {
        let dump_file_path = data_dir.join("index.hnsw");

        let inner: HNSW2Index = BufferedFile::open(dump_file_path)
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
            counter: create_counter(offload_config.slot_count_exp, offload_config.slot_size_exp),
        })
    }

    pub fn try_load(info: VectorFieldInfo, offload_config: &OffloadFieldConfig) -> Result<Self> {
        let dump_file_path = info.data_dir.join("index.hnsw");

        let inner: HNSW2Index = BufferedFile::open(dump_file_path)
            .map_err(|e| anyhow!("Cannot open hnsw file: {}", e))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read hnsw file: {}", e))?;

        Ok(Self {
            field_path: info.field_path,
            inner,
            data_dir: info.data_dir,
            model: info.model.0,
            counter: create_counter(offload_config.slot_count_exp, offload_config.slot_size_exp),
        })
    }

    pub fn field_path(&self) -> Box<[String]> {
        self.field_path.clone()
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
            model: OramaModelSerializable(self.model),
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
        self.counter.register(now());

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

        for (doc_id, score) in data {
            if filtered_doc_ids.is_some_and(|ids| !ids.contains(&doc_id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(&doc_id) {
                continue;
            }

            if score >= similarity {
                let v = output.entry(doc_id).or_insert(0.0);
                *v += score;
            }
        }

        Ok(())
    }

    pub fn stats(&self) -> Result<CommittedVectorFieldStats> {
        Ok(CommittedVectorFieldStats {
            dimensions: self.inner.dim(),
            vector_count: self.inner.len(),
            loaded: true,
        })
    }
}

#[derive(Serialize, Debug)]
pub struct CommittedVectorFieldStats {
    pub dimensions: usize,
    pub vector_count: usize,
    pub loaded: bool,
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn create_counter(slot_count_exp: u8, slot_size_exp: u8) -> InvocationCounter {
    // 2^8 * 2^4 = 4096 seconds = 1 hour, 6 minutes, and 36 seconds
    let counter = InvocationCounter::new(slot_count_exp, slot_size_exp);

    // Avoid to unload committed vector field if it is created right now.
    counter.register(now());

    counter
}
