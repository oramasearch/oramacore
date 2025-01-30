use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{anyhow, Result};
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

use crate::{file_utils::create_if_not_exists, types::DocumentId};

#[derive(
    Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash, Deserialize,
)]
pub struct IdxID(Option<DocumentId>);
impl IdxType for IdxID {}

#[derive(Debug)]
pub struct VectorField {
    inner: HNSWIndex<f32, IdxID>,
    file_path: PathBuf,
}

impl VectorField {
    pub fn from_iter<I>(iter: I, dimension: usize, file_path: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let params = HNSWParams::<f32>::default().max_item(1_000_000_000);
        let inner = HNSWIndex::new(dimension, &params);

        let mut s = Self { inner, file_path };

        s.add_and_dump(iter)?;

        Ok(s)
    }

    pub fn from_dump_and_iter(
        file_path: PathBuf,
        iter: impl Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    ) -> Result<Self> {
        let inner = HNSWIndex::load(
            file_path
                .to_str()
                .ok_or_else(|| anyhow!("Cannot convert path to string"))?,
        )
        .map_err(|e| anyhow!("Cannot load HNSWIndex from {:?}: {}", file_path, e))?;

        let mut s = Self { inner, file_path };

        s.add_and_dump(iter)?;

        Ok(s)
    }

    pub fn load(info: VectorFieldInfo) -> Result<Self> {
        let file_path_str = match info.file_path.to_str() {
            Some(file_path_str) => file_path_str,
            None => {
                return Err(anyhow!("Cannot convert path to string"));
            }
        };

        let inner = HNSWIndex::load(file_path_str)
            .map_err(|e| anyhow!("Cannot load HNSWIndex from {:?}: {}", info.file_path, e))?;

        if inner.dimension() != info.dimension {
            return Err(anyhow!(
                "Dimension mismatch: expected {}, got {}",
                info.dimension,
                inner.dimension()
            ));
        }

        Ok(Self {
            inner,
            file_path: info.file_path,
        })
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            dimension: self.inner.dimension(),
            file_path: self.file_path.clone(),
        }
    }

    pub fn search(
        &self,
        target: &[f32],
        limit: usize,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
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

        create_if_not_exists(
            self.file_path
                .parent()
                .expect("Parent folder should be calculated"),
        )?;

        let file_path_str = match self.file_path.to_str() {
            Some(file_path_str) => file_path_str,
            None => {
                return Err(anyhow!("Cannot convert path to string"));
            }
        };
        self.inner
            .dump(file_path_str)
            .map_err(|e| anyhow!("Cannot dump index to file: {}", e))?;

        Ok(())
    }

    pub fn file_path(&self) -> PathBuf {
        self.file_path.clone()
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
    pub file_path: PathBuf,
}
