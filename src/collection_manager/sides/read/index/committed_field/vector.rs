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
    filters::FilterResult,
    fs::{create_if_not_exists, BufferedFile},
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    collection_manager::sides::read::{
        index::{
            committed_field::offload_utils::{
                create_counter, should_offload, update_invocation_counter, MutRef,
            },
            merge::{CommittedField, CommittedFieldMetadata, Field},
            uncommitted_field::UncommittedVectorField,
        },
        OffloadFieldConfig,
    },
    embeddings::Model,
    lock::OramaSyncLock,
    types::DocumentId,
};

// From benchmarks, the brute force index is "ok" up to 10k documents.
const MIN_HNSW_DOCS: usize = 10_000;
const BRUTE_FORCE_INDEX_FILE_NAME: &str = "index.vec";
const HNSW_INDEX_FILE_NAME: &str = "index.hnsw";
const HNSW_INDEX_2_FILE_NAME: &str = "index.hnsw2";

#[derive(Serialize, Deserialize, Debug)]
pub struct CommittedVectorFieldStats {
    pub dimensions: usize,
    pub vector_count: usize,
    pub loaded: AtomicBool,
    pub layout: VectorLayoutType,
}

impl Clone for CommittedVectorFieldStats {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            vector_count: self.vector_count,
            loaded: AtomicBool::new(self.loaded.load(std::sync::atomic::Ordering::Acquire)),
            layout: self.layout,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq, Eq)]
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
    BruteForce(VectorBruteForce<DocumentId>),
}
impl VectorLayout {
    fn as_layout_type(&self) -> VectorLayoutType {
        match self {
            VectorLayout::Hnsw(_) => VectorLayoutType::Hnsw,
            VectorLayout::BruteForce(_) => VectorLayoutType::Plain,
        }
    }
    fn len(&self) -> usize {
        match self {
            VectorLayout::Hnsw(index) => index.len(),
            VectorLayout::BruteForce(index) => index.len(),
        }
    }
    fn dim(&self) -> usize {
        match self {
            VectorLayout::Hnsw(index) => index.dim(),
            VectorLayout::BruteForce(index) => index.dim(),
        }
    }
    fn save(&self, data_dir: &PathBuf) -> Result<()> {
        create_if_not_exists(data_dir).context("Cannot create data directory")?;
        match self {
            Self::Hnsw(index) => {
                let dump_file_path = data_dir.join(HNSW_INDEX_2_FILE_NAME);
                BufferedFile::create_or_overwrite(dump_file_path)
                    .context("Cannot create hnsw file")?
                    .write_bincode_data(&index)
                    .context("Cannot write hnsw file")?;
            }
            Self::BruteForce(index) => {
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
        params: &VectorSearchParams,
        model: Model,
        output: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        match self {
            VectorLayout::Hnsw(index) => {
                // We filtered matches by:
                // - `filtered_doc_ids`: removed by `search` method
                // - `similarity` threshold: removed by `search` method
                // If there're not uncomitted deletions or the user doesn't filter, the limit is ok:
                // HNSW returns the most probable matches first, so we can stop when we reach the limit.
                // Otherwise, we should continue the search till reach the limit.
                // Anyway, the implementation below returns a Vec, so we should redo the search till reach the limit.
                // For now, we just double the limit.
                // TODO: implement a better way to handle this.
                let limit = if params.filtered_doc_ids.is_some() {
                    params.limit * 2
                } else {
                    params.limit
                };

                let data = index.search(params.target, limit);

                for (doc_id, score) in data {
                    if let Some(filtered) = params.filtered_doc_ids {
                        if !filtered.contains(&doc_id) {
                            continue;
                        }
                    }
                    let score = model.rescale_score(score);

                    if score >= params.similarity {
                        let v = output.entry(doc_id).or_insert(0.0);
                        *v += score;
                    }
                }
            }
            VectorLayout::BruteForce(index) => {
                let data = index.search(
                    params.target,
                    params.limit,
                    params.similarity,
                    &VectorBruteForceSearchResult {
                        filtered_doc_ids: params.filtered_doc_ids,
                    },
                );

                for (doc_id, score) in data {
                    let score = model.rescale_score(score);
                    if score >= params.similarity {
                        let v = output.entry(doc_id).or_insert(0.0);
                        *v += score;
                    }
                }
            }
        }

        Ok(())
    }
}

pub struct VectorSearchParams<'search> {
    pub target: &'search [f32],
    pub similarity: f32,
    pub limit: usize,
    pub filtered_doc_ids: Option<&'search FilterResult<DocumentId>>,
}

struct VectorBruteForceSearchResult<'search, DocumentId> {
    pub filtered_doc_ids: Option<&'search FilterResult<DocumentId>>,
}
impl ShouldInclude<DocumentId> for VectorBruteForceSearchResult<'_, DocumentId>
where
    DocumentId: Eq + std::hash::Hash,
{
    fn should_include(&self, item: &DocumentId) -> bool {
        self.filtered_doc_ids
            .map(|filtered| filtered.contains(item))
            .unwrap_or(true)
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
            BufferedFile::create_or_overwrite(data_dir.join(BRUTE_FORCE_INDEX_FILE_NAME))
                .context("Cannot create vector file")?
                .write_bincode_data(&new_index)
                .context("Cannot write vector file")?;

            VectorLayout::BruteForce(new_index)
        } else {
            let mut new_index = HNSW2Index::new_with_deletion(uncommitted.dimension);
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

    fn add_uncommitted(
        &self,
        uncommitted: &UncommittedVectorField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        debug_assert_eq!(uncommitted.dimension, self.stats.dimensions);
        debug_assert_eq!(uncommitted.get_model(), self.metadata.model);

        let mut new_field = Self::try_load(self.metadata.clone(), offload_config)?;
        new_field.metadata.data_dir = data_dir.clone();

        {
            let mut a = new_field.status.write("").unwrap();
            let a = &mut **a;
            let mut layout = match a {
                VectorStatus::Unloaded => MutRef::Owned(load_layout(&self.metadata.data_dir)?),
                VectorStatus::Loaded(layout) => MutRef::Borrowed(layout),
            };
            match &mut *layout {
                VectorLayout::BruteForce(brute_force) => {
                    for (doc, vec) in uncommitted.iter() {
                        if uncommitted_document_deletions.contains(&doc) {
                            continue;
                        }
                        for vector in vec {
                            brute_force.add_owned(vector, doc);
                        }
                    }

                    brute_force.delete(uncommitted_document_deletions);

                    if brute_force.len() >= MIN_HNSW_DOCS {
                        info!(
                            "Upgrading vector field {:?} from brute-force to HNSW layout",
                            self.metadata.field_path
                        );
                        let mut hnsw = HNSW2Index::new_with_deletion(uncommitted.dimension);

                        hnsw.batch_add(
                            uncommitted
                                .iter()
                                .filter_map(|(doc, vecs)| {
                                    if uncommitted_document_deletions.contains(&doc) {
                                        None
                                    } else {
                                        Some(vecs.into_iter().map(move |v| (v, doc)))
                                    }
                                })
                                .flatten(),
                        )?;
                        hnsw.delete_batch(uncommitted_document_deletions);
                        hnsw.build()?;

                        *layout = VectorLayout::Hnsw(Box::new(hnsw));
                    }
                }
                VectorLayout::Hnsw(hnsw) => {
                    hnsw.batch_add(
                        uncommitted
                            .iter()
                            .filter_map(|(doc, vecs)| {
                                if uncommitted_document_deletions.contains(&doc) {
                                    None
                                } else {
                                    Some(vecs.into_iter().map(move |v| (v, doc)))
                                }
                            })
                            .flatten(),
                    )?;
                    hnsw.delete_batch(uncommitted_document_deletions);

                    hnsw.build()?;
                    hnsw.rebuild()?;
                }
            };

            layout.save(&data_dir)?;

            // Update stats to reflect the current layout state after potential upgrade
            new_field.stats = CommittedVectorFieldStats {
                dimensions: layout.dim(),
                vector_count: layout.len(),
                loaded: AtomicBool::new(true),
                layout: layout.as_layout_type(),
            };
        }

        Ok(new_field)
    }

    fn metadata(&self) -> VectorFieldInfo {
        self.metadata.clone()
    }
}

impl Field for CommittedVectorField {
    type FieldStats = CommittedVectorFieldStats;

    fn field_path(&self) -> &[String] {
        self.metadata.field_path.as_ref()
    }

    fn stats(&self) -> CommittedVectorFieldStats {
        self.stats.clone()
    }
}

impl CommittedVectorField {
    pub fn search(
        &self,
        params: &VectorSearchParams<'_>,
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

        layout.search(params, self.metadata.model, output)?;

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
    let is_hnsw = is_hnsw || std::fs::exists(data_dir.join(HNSW_INDEX_2_FILE_NAME))?;
    if is_hnsw {
        let dump_file_path = data_dir.join(HNSW_INDEX_FILE_NAME);
        if std::fs::exists(&dump_file_path).unwrap_or(false) {
            let inner = BufferedFile::open(dump_file_path)
                .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
                .read_as_vec()
                .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;
            let inner = HNSW2Index::deserialize_bincode_compat(&inner)
                .map_err(|e| anyhow!("Cannot deserialize hnsw file: {e}"))?;
            Ok(VectorLayout::Hnsw(Box::new(inner)))
        } else {
            let dump_file_path = data_dir.join(HNSW_INDEX_2_FILE_NAME);
            let inner: HNSW2Index<DocumentId> = BufferedFile::open(dump_file_path)
                .map_err(|e| anyhow!("Cannot open hnsw2 file: {e}"))?
                .read_bincode_data()
                .map_err(|e| anyhow!("Cannot read hnsw2 file: {e}"))?;
            Ok(VectorLayout::Hnsw(Box::new(inner)))
        }
    } else {
        let dump_file_path = data_dir.join(BRUTE_FORCE_INDEX_FILE_NAME);
        let inner: VectorBruteForce<DocumentId> = BufferedFile::open(dump_file_path)
            .map_err(|e| anyhow!("Cannot open hnsw file: {e}"))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read hnsw file: {e}"))?;
        Ok(VectorLayout::BruteForce(inner))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VectorFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
    pub model: Model,
}

impl CommittedFieldMetadata for VectorFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
    fn field_path(&self) -> &[String] {
        self.field_path.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use duration_string::DurationString;
    use rand::Rng;

    use super::*;
    use crate::tests::utils::*;

    #[test]
    fn test_from_brute_force_to_hnsw() {
        init_log();

        let data_dir = generate_new_path();
        let first_data_dir = data_dir.join("first");
        let second_data_dir = data_dir.join("second");

        std::fs::create_dir_all(&first_data_dir).unwrap();
        std::fs::create_dir_all(&second_data_dir).unwrap();

        let offload_config = OffloadFieldConfig {
            unload_window: DurationString::from_string("12h".to_string()).unwrap(),
            slot_count_exp: 10,
            slot_size_exp: 8,
        };

        let model = Model::BGESmall;

        // Use random vectors instead of zero vectors to avoid pathological behavior
        // in HNSW construction where all identical vectors cause excessive computation.
        let mut rng = rand::rng();

        let mut uncommitted =
            UncommittedVectorField::empty(Box::new(["field_path".to_string()]), model);
        for doc_id in 1..=(MIN_HNSW_DOCS - 1) {
            let vectors = vec![(0..model.dimensions())
                .map(|_| rng.random::<f32>())
                .collect::<Vec<_>>()];
            uncommitted
                .insert(DocumentId(doc_id as u64), vectors)
                .unwrap();
        }

        let committed = CommittedVectorField::from_uncommitted(
            &uncommitted,
            first_data_dir,
            &HashSet::new(),
            offload_config,
        )
        .unwrap();

        assert_eq!(committed.stats.layout, VectorLayoutType::Plain);

        let mut uncommitted =
            UncommittedVectorField::empty(Box::new(["field_path".to_string()]), model);
        for doc_id in (MIN_HNSW_DOCS - 1)..=(MIN_HNSW_DOCS + 10) {
            let vectors = vec![(0..model.dimensions())
                .map(|_| rng.random::<f32>())
                .collect::<Vec<_>>()];
            uncommitted
                .insert(DocumentId(doc_id as u64), vectors)
                .unwrap();
        }

        let new_committed = committed
            .add_uncommitted(
                &uncommitted,
                second_data_dir,
                &HashSet::new(),
                offload_config,
            )
            .unwrap();

        assert_eq!(new_committed.stats.layout, VectorLayoutType::Hnsw);

        let target = (0..model.dimensions())
            .map(|_| rng.random::<f32>())
            .collect::<Vec<_>>();

        let mut output = HashMap::new();
        let params = VectorSearchParams {
            target: &target,
            similarity: 0.0,
            limit: 5,
            filtered_doc_ids: None,
        };
        new_committed.search(&params, &mut output).unwrap();

        assert!(!output.is_empty());
    }
}
