use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use hora::{
    core::{
        ann_index::{ANNIndex, SerializableIndex},
        metrics::Metric,
        node::IdxType,
    },
    index::{hnsw_idx::HNSWIndex, hnsw_params::HNSWParams},
};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    types::DocumentId,
};

#[derive(
    Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash, Deserialize,
)]
pub struct IdxID(Option<DocumentId>);
impl IdxType for IdxID {}

#[derive(Debug)]
pub struct VectorField {
    inner: HNSWIndex<f32, IdxID>,
    data_dir: PathBuf,
    deleted_documents: HashSet<DocumentId>,
}

impl VectorField {
    pub fn from_iter<I>(iter: I, dimension: usize, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let params = HNSWParams::<f32>::default().max_item(1_000_000_000);
        let inner = HNSWIndex::new(dimension, &params);

        let mut s = Self {
            inner,
            data_dir,
            deleted_documents: HashSet::new(),
        };

        s.add_and_dump(iter)?;

        Ok(s)
    }

    pub fn from_dump_and_iter(
        data_dir: PathBuf,
        iter: impl Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
    ) -> Result<Self> {
        let dump_file_path = data_dir.join("index.hnsw");

        let inner = HNSWIndex::load(
            dump_file_path
                .to_str()
                .ok_or_else(|| anyhow!("Cannot convert path to string"))?,
        )
        .map_err(|e| anyhow!("Cannot load HNSWIndex from {:?}: {}", dump_file_path, e))?;

        let mut s = Self {
            inner,
            data_dir,
            deleted_documents: uncommitted_document_deletions.clone(),
        };

        s.add_and_dump(
            iter.filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id)),
        )?;

        Ok(s)
    }

    pub fn load(info: VectorFieldInfo) -> Result<Self> {
        let dump_file_path = info.data_dir.join("index.hnsw");
        let deleted_documents_file_path = info.data_dir.join("deleted_documents.bin");

        let file_path_str = match dump_file_path.to_str() {
            Some(file_path_str) => file_path_str,
            None => {
                return Err(anyhow!("Cannot convert path to string"));
            }
        };

        let inner = HNSWIndex::load(file_path_str)
            .map_err(|e| anyhow!("Cannot load HNSWIndex from {:?}: {}", dump_file_path, e))?;

        let deleted_documents = BufferedFile::open(deleted_documents_file_path)
            .map_err(|e| anyhow!("Cannot open deleted documents file: {}", e))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read deleted documents file: {}", e))?;

        if inner.dimension() != info.dimension {
            return Err(anyhow!(
                "Dimension mismatch: expected {}, got {}",
                info.dimension,
                inner.dimension()
            ));
        }

        Ok(Self {
            inner,
            data_dir: info.data_dir,
            deleted_documents,
        })
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            dimension: self.inner.dimension(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn search(
        &self,
        target: &[f32],
        limit: usize,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        let search_output = self.inner.search_nodes(target, limit);
        if search_output.is_empty() {
            return Ok(());
        }

        for (node, distance) in search_output {
            let doc_id = match node.idx() {
                Some(idx) => match idx.0 {
                    Some(id) => id,
                    // ???
                    None => {
                        warn!("This should not happen");
                        continue;
                    }
                },
                None => {
                    warn!("This should not happen");
                    continue;
                }
            };

            if filtered_doc_ids.map_or(false, |ids| !ids.contains(&doc_id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(&doc_id) {
                continue;
            }

            // `hora` returns the score as Euclidean distance.
            // That means 0.0 is the best score and the larger the score, the worse.
            // NB: because it is a distance, it is always positive.
            // NB2: the score is capped with a maximum value.
            // So, inverting the score could be a good idea.
            // NB3: we capped the score to "100".
            // TODO: put `0.01` number in config.
            let inc = 1.0 / distance.max(0.01);

            let v = output.entry(doc_id).or_insert(0.0);
            *v += inc;
        }

        Ok(())
    }

    fn add_and_dump(
        &mut self,
        iter: impl Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    ) -> Result<()> {
        self.add(iter)?;

        create_if_not_exists(&self.data_dir)?;

        let dump_file_path = self.data_dir.join("index.hnsw");
        let deleted_documents_file_path = self.data_dir.join("deleted_documents.bin");

        let file_path_str = match dump_file_path.to_str() {
            Some(file_path_str) => file_path_str,
            None => {
                return Err(anyhow!("Cannot convert path to string"));
            }
        };
        self.inner
            .dump(file_path_str)
            .map_err(|e| anyhow!("Cannot dump index to file: {}", e))?;

        BufferedFile::create_or_overwrite(deleted_documents_file_path)
            .context("Cannot create deleted documents file")?
            .write_bincode_data(&self.deleted_documents)
            .context("Cannot serialize deleted documents file")?;

        Ok(())
    }

    pub fn clone_to(&self, data_dir: &Path) -> Result<()> {
        let new_dump_file_path = data_dir.join("index.hnsw");
        let new_deleted_documents_file_path = data_dir.join("deleted_documents.bin");

        let old_dump_file_path = self.data_dir.join("index.hnsw");
        let old_deleted_documents_file_path = self.data_dir.join("deleted_documents.bin");

        std::fs::copy(
            old_deleted_documents_file_path,
            new_deleted_documents_file_path,
        )
        .map_err(|e| anyhow!("Cannot copy deleted documents file: {}", e))?;
        std::fs::copy(old_dump_file_path, new_dump_file_path)
            .map_err(|e| anyhow!("Cannot copy hnsw file: {}", e))?;

        Ok(())
    }

    fn add(&mut self, iter: impl Iterator<Item = (DocumentId, Vec<Vec<f32>>)>) -> Result<()> {
        for (doc_id, vectors) in iter {
            for vector in vectors {
                self.inner
                    .add(&vector, IdxID(Some(doc_id)))
                    .map_err(|e| anyhow!("Cannot add vector to index: {}", e))?;
            }
        }

        self.inner
            .build(Metric::Manhattan)
            .map_err(|e| anyhow!("Cannot build index: {}", e))?;

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorFieldInfo {
    pub dimension: usize,
    pub data_dir: PathBuf,
}
