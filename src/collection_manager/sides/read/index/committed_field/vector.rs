use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use invocation_counter::InvocationCounter;
use oramacore_lib::{
    data_structures::{hnsw2::HNSW2Index, vector_bruteforce::VectorBruteForce, ShouldInclude},
    fs::{create_if_not_exists, BufferedFile},
};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::{
        index::{
            committed_field::offload_utils::{
                create_counter, should_offload, update_invocation_counter, Cow,
            },
            merge::{CommittedField, FieldMetadata},
            uncommitted_field::UncommittedVectorField,
        },
        search::SearchDocumentContext,
        OffloadFieldConfig,
    },
    lock::OramaSyncLock,
    python::embeddings::Model,
    types::DocumentId,
};

// From benchmarks, the brute force index is "ok" up to 10k documents.
const MIN_HNSW_DOCS: usize = 10_000;
const BRUTE_FORCE_INDEX_FILE_NAME: &str = "index.vec";
const HNSW_INDEX_FILE_NAME: &str = "index.hnsw";

#[derive(Serialize, Deserialize, Debug)]
pub struct CommittedVectorFieldStats {
    pub dimensions: usize,
    pub vector_count: usize,
    pub loaded: AtomicBool,
    pub layout: VectorLayoutType,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum VectorLayoutType {
    #[serde(rename = "hnsw")]
    Hnsw,
    #[serde(rename = "plain")]
    Plain,
}
impl Display for VectorLayoutType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorLayoutType::Hnsw => write!(f, "hnsw"),
            VectorLayoutType::Plain => write!(f, "plain"),
        }
    }
}

enum VectorLayout {
    Hnsw(Box<HNSW2Index<DocumentId>>),
    Plain(VectorBruteForce<DocumentId>),
}
impl VectorLayout {
    fn as_layout_type(&self) -> VectorLayoutType {
        match self {
            VectorLayout::Hnsw(_) => VectorLayoutType::Hnsw,
            VectorLayout::Plain(_) => VectorLayoutType::Plain,
        }
    }
    fn len(&self) -> usize {
        match self {
            VectorLayout::Hnsw(index) => index.len(),
            VectorLayout::Plain(index) => index.len(),
        }
    }
    fn dim(&self) -> usize {
        match self {
            VectorLayout::Hnsw(index) => index.dim(),
            VectorLayout::Plain(index) => index.dim(),
        }
    }
    fn add(&mut self, vector: &[f32], doc_id: DocumentId) -> Result<()> {
        match self {
            VectorLayout::Hnsw(index) => {
                index.add(vector, doc_id).context("Cannot add vector")?;
            }
            VectorLayout::Plain(index) => {
                index.add_owned(vector.to_vec(), doc_id);
            }
        }
        Ok(())
    }
    fn add_owned(&mut self, vector: Vec<f32>, doc_id: DocumentId) -> Result<()> {
        match self {
            VectorLayout::Hnsw(index) => {
                index
                    .add_owned(vector, doc_id)
                    .context("Cannot add vector")?;
            }
            VectorLayout::Plain(index) => {
                index.add_owned(vector, doc_id);
            }
        }
        Ok(())
    }
    fn get_data(&self) -> Box<dyn Iterator<Item = (DocumentId, &[f32])> + '_> {
        match self {
            VectorLayout::Hnsw(index) => Box::new(index.get_data()),
            VectorLayout::Plain(index) => Box::new(index.get_data()),
        }
    }
    fn save(&self, data_dir: &PathBuf) -> Result<()> {
        create_if_not_exists(data_dir).context("Cannot create data directory")?;
        match self {
            Self::Hnsw(index) => {
                let dump_file_path = data_dir.join(HNSW_INDEX_FILE_NAME);
                BufferedFile::create_or_overwrite(dump_file_path)
                    .context("Cannot create hnsw file")?
                    .write_bincode_data(&index)
                    .context("Cannot write hnsw file")?;
            }
            Self::Plain(index) => {
                let dump_file_path = data_dir.join(BRUTE_FORCE_INDEX_FILE_NAME);
                BufferedFile::create_or_overwrite(dump_file_path)
                    .context("Cannot create vector file")?
                    .write_bincode_data(&index)
                    .context("Cannot write vector file")?;
            }
        }
        Ok(())
    }
    fn search(
        &self,
        target: &[f32],
        similarity: f32,
        limit: usize,
        search_document_context: &SearchDocumentContext<'_, DocumentId>,
        model: Model,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        match self {
            VectorLayout::Hnsw(index) => {
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
                let limit = if search_document_context.has_filtered() {
                    limit * 2
                } else {
                    limit
                };

                let data = index.search(target, limit);

                for (doc_id, score) in data {
                    if search_document_context.should_exclude(&doc_id) {
                        continue;
                    }
                    let score = model.rescale_score(score);

                    if score >= similarity {
                        let v = output.entry(doc_id).or_insert(0.0);
                        *v += score;
                    }
                }
            }
            VectorLayout::Plain(index) => {
                let data = index.search(target, limit, similarity, search_document_context);

                for (doc_id, score) in data {
                    let score = model.rescale_score(score);
                    if score >= similarity {
                        let v = output.entry(doc_id).or_insert(0.0);
                        *v += score;
                    }
                }
            }
        }

        Ok(())
    }
}

enum VectorStatus {
    Loaded(VectorLayout),
    Unloaded,
}

pub struct CommittedVectorField {
    metadata: VectorFieldInfo,
    stats: CommittedVectorFieldStats,
    status: OramaSyncLock<VectorStatus>,
    invocation_counter: InvocationCounter,
    unload_window: Duration,
}

impl CommittedField for CommittedVectorField {
    type FieldMetadata = VectorFieldInfo;
    type Uncommitted = UncommittedVectorField;

    fn try_load(metadata: VectorFieldInfo, offload_config: OffloadFieldConfig) -> Result<Self> {
        let layout = load_layout(&metadata.data_dir).context("Cannot load vector layout")?;

        Ok(Self {
            metadata,
            stats: CommittedVectorFieldStats {
                dimensions: layout.dim(),
                vector_count: layout.len(),
                loaded: AtomicBool::new(true),
                layout: layout.as_layout_type(),
            },
            status: OramaSyncLock::new("vector_inner", VectorStatus::Loaded(layout)),
            invocation_counter: create_counter(offload_config),
            unload_window: offload_config.unload_window.into(),
        })
    }

    fn from_uncommitted(
        uncommitted: &UncommittedVectorField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let iter = uncommitted.iter();

        let layout = if uncommitted.len() < MIN_HNSW_DOCS {
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
            BufferedFile::create_or_overwrite(data_dir.join(BRUTE_FORCE_INDEX_FILE_NAME))
                .context("Cannot create vector file")?
                .write_bincode_data(&new_index)
                .context("Cannot write vector file")?;

            VectorLayout::Plain(new_index)
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
            BufferedFile::create_or_overwrite(data_dir.join(HNSW_INDEX_FILE_NAME))
                .context("Cannot create hnsw file")?
                .write_bincode_data(&new_index)
                .context("Cannot write hnsw file")?;

            VectorLayout::Hnsw(Box::new(new_index))
        };

        Ok(Self {
            metadata: VectorFieldInfo {
                field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
                data_dir,
                model: uncommitted.get_model(),
            },
            stats: CommittedVectorFieldStats {
                dimensions: layout.dim(),
                vector_count: layout.len(),
                loaded: AtomicBool::new(true),
                layout: layout.as_layout_type(),
            },
            status: OramaSyncLock::new("vector_inner", VectorStatus::Loaded(layout)),
            invocation_counter: create_counter(offload_config),
            unload_window: offload_config.unload_window.into(),
        })
    }

    fn metadata(&self) -> VectorFieldInfo {
        self.metadata.clone()
    }

    fn add_uncommitted(
        &self,
        uncommitted: &UncommittedVectorField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        debug_assert_eq!(uncommitted.field_path(), self.metadata.field_path.as_ref(),);
        debug_assert_eq!(uncommitted.dimension, self.stats.dimensions,);
        debug_assert_eq!(uncommitted.get_model(), self.metadata.model,);

        let dim = self.stats.dimensions;

        let total_docs = uncommitted.len() + self.stats.vector_count;
        let mut new_layout = if total_docs < MIN_HNSW_DOCS {
            let mut brute_force = VectorBruteForce::new(dim);
            brute_force.set_capacity(total_docs);
            VectorLayout::Plain(brute_force)
        } else {
            let hnsw = HNSW2Index::new(dim);
            VectorLayout::Hnsw(Box::new(hnsw))
        };

        let lock = self.status.read("commit").unwrap();
        let vector_status = &**lock;
        let current_layout = match vector_status {
            VectorStatus::Loaded(layout) => Cow::Borrowed(layout),
            VectorStatus::Unloaded => Cow::Owned(load_layout(&self.metadata.data_dir)?),
        };

        let old_data = current_layout.get_data();
        for (doc_id, vector) in old_data {
            if uncommitted_document_deletions.contains(&doc_id) {
                continue;
            }
            new_layout
                .add(vector, doc_id)
                .context("Cannot add vector")?;
        }

        for (doc_id, vectors) in uncommitted.iter() {
            if uncommitted_document_deletions.contains(&doc_id) {
                continue;
            }
            for vector in vectors {
                new_layout
                    .add_owned(vector, doc_id)
                    .context("Cannot add vector")?;
            }
        }
        if let VectorLayout::Hnsw(hnsw) = &mut new_layout {
            hnsw.build().context("Cannot build hnsw index")?;
        }

        new_layout.save(&data_dir)?;

        Ok(Self {
            metadata: VectorFieldInfo {
                field_path: self.metadata.field_path.clone(),
                data_dir,
                model: self.metadata.model,
            },
            stats: CommittedVectorFieldStats {
                dimensions: self.stats.dimensions,
                vector_count: new_layout.len(),
                loaded: AtomicBool::new(true),
                layout: new_layout.as_layout_type(),
            },
            status: OramaSyncLock::new("vector_inner", VectorStatus::Loaded(new_layout)),
            invocation_counter: create_counter(offload_config),
            unload_window: offload_config.unload_window.into(),
        })
    }
}

impl CommittedVectorField {
    pub fn stats(&self) -> &CommittedVectorFieldStats {
        &self.stats
    }

    pub fn search(
        &self,
        target: &[f32],
        similarity: f32,
        limit: usize,
        search_document_context: &SearchDocumentContext<'_, DocumentId>,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        update_invocation_counter(&self.invocation_counter);

        let lock = self.status.read("search").unwrap();
        let lock = if let VectorStatus::Unloaded = &**lock {
            drop(lock); // Release the read lock before loading

            let layout = load_layout(&self.metadata.data_dir)?;
            let mut write_lock = self.status.write("load").unwrap();
            **write_lock = VectorStatus::Loaded(layout);
            self.stats
                .loaded
                .store(true, std::sync::atomic::Ordering::Release);
            drop(write_lock); // Release the write lock

            self.status.read("search").unwrap()
        } else {
            lock
        };
        let vector_status = &**lock;
        let layout = match vector_status {
            VectorStatus::Loaded(layout) => layout,
            VectorStatus::Unloaded => {
                // This never happens because of the logic above.
                return Err(anyhow!("Cannot search unloaded vector field"));
            }
        };

        layout.search(
            target,
            similarity,
            limit,
            search_document_context,
            self.metadata.model,
            output,
        )?;

        Ok(())
    }

    pub fn unload_if_not_used(&self) {
        // Some invocations happened recently, do not unload.
        if !should_offload(&self.invocation_counter, self.unload_window) {
            return;
        }

        let lock = self.status.read("unload_if_not_used").unwrap();
        // This field is already unloaded. Skip.
        if let VectorStatus::Unloaded = &**lock {
            return;
        }

        drop(lock); // Release the read lock before unloading
        let mut lock = self.status.write("unload").unwrap();
        // Double check if another thread unloaded the field meanwhile.
        if let VectorStatus::Unloaded = &**lock {
            return;
        }

        self.stats
            .loaded
            .store(false, std::sync::atomic::Ordering::Release);

        **lock = VectorStatus::Unloaded;
    }
}

fn load_layout(data_dir: &Path) -> Result<VectorLayout> {
    let is_hnsw = std::fs::exists(data_dir.join(HNSW_INDEX_FILE_NAME))?;
    if is_hnsw {
        let dump_file_path = data_dir.join(HNSW_INDEX_FILE_NAME);
        let inner: HNSW2Index<DocumentId> = BufferedFile::open(dump_file_path)
            .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;
        Ok(VectorLayout::Hnsw(Box::new(inner)))
    } else {
        let dump_file_path = data_dir.join(BRUTE_FORCE_INDEX_FILE_NAME);
        let inner: VectorBruteForce<DocumentId> = BufferedFile::open(dump_file_path)
            .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;
        Ok(VectorLayout::Plain(inner))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
    pub model: Model,
}

impl FieldMetadata for VectorFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
}
