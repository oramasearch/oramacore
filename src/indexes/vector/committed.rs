use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, Context, Result};
use hora::{core::{ann_index::{ANNIndex, SerializableIndex}, metrics::Metric, node::IdxType}, index::{hnsw_idx::HNSWIndex, hnsw_params::HNSWParams}};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::document_storage::DocumentId;

#[derive(
    Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash, Deserialize,
)]
pub struct IdxID(Option<DocumentId>);
impl IdxType for IdxID {}

// This enum represents the state of the index.
// It can be in memory or on file.
// If it is in memory, it is faster to search, but it takes more memory.
// So, the size of this enum changes a lot depending on the state.
// Due to the enum nature, the memory used to store "OnFile" variant or "InMemory" is always the same.
// Because "InMemory" variant is larger (408 bytes vs 24 bytes), Rust uses 408 even for "OnFile" variant.
// If there're a lot of indexes, it can be a problem.
// TODO: think about how to optimize it.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum CommittedState {
    InMemory(LoadedInMemoryCommittedIndex),
    OnFile(PathBuf),
}

impl CommittedState {
    pub fn new_in_memory(dimension: usize) -> Self {
        let index = HNSWIndex::<f32, IdxID>::new(
            dimension,
            &HNSWParams::<f32>::default(),
        );
        Self::InMemory(LoadedInMemoryCommittedIndex { index })
    }

    pub fn load_in_memory(&mut self) -> Result<&mut LoadedInMemoryCommittedIndex> {
        let path = match self {
            Self::InMemory(idx) => return Ok(idx),
            Self::OnFile(path) => path,
        };

        let loaded = LoadedInMemoryCommittedIndex::from_path(path)?;
        *self = Self::InMemory(loaded);
        match self {
            Self::InMemory(idx) => Ok(idx),
            Self::OnFile(_) => unreachable!(),
        }
    }

    pub fn unload_from_memory(&mut self, path: PathBuf) -> Result<()> {
        let loaded = match self {
            Self::InMemory(loaded) => loaded,
            Self::OnFile(_) => {
                return Ok(())
            }
        };

        loaded.save_on_fs(path.clone())?;

        *self = Self::OnFile(path);

        Ok(())
    }

    pub fn search(&self, target: &[f32], limit: usize, output: &mut HashMap<DocumentId, f32>) -> Result<()> {
        match self {
            Self::InMemory(idx) => {
                idx.search(target, limit, output);
                Ok(())
            },
            Self::OnFile(path) => {
                let loaded = LoadedInMemoryCommittedIndex::from_path(path)
                    .with_context(|| format!("Cannot load index from '{:?}'", path))?;
                loaded.search(target, limit, output);

                Ok(())
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoadedInMemoryCommittedIndex {
    pub index: HNSWIndex<f32, IdxID>,
}

impl LoadedInMemoryCommittedIndex {
    pub fn add(&mut self, vector: &[f32], id: DocumentId) -> Result<()> {
        self.index.add(vector, IdxID(Some(id)))
            .map_err(|e| anyhow!("Error adding vector: {:?}", e))
    }

    pub fn build(&mut self) -> Result<()> {
        self.index.build(Metric::Manhattan)
            .map_err(|e| anyhow!("Error building index: {:?}", e))
    }

    pub fn search(&self, target: &[f32], limit: usize, output: &mut HashMap<DocumentId, f32>) {
        let search_output = self.index.search(target, limit).into_iter();
        for id in search_output {
            let doc_id = match id.0 {
                Some(id) => id,
                // ???
                None => {
                    warn!("This should not happen");
                    continue;
                }
            };
            let v = output.entry(doc_id).or_insert(0.0);
            *v += 1.0;
        }
    }

    pub fn save_on_fs(&mut self, path: PathBuf) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Cannot create directory '{:?}'", path))?;
        }

        let path: &str = path.to_str()
            .with_context(|| format!("'{:?}' cannot to converted to str", path))?;
        self.index.dump(path)
            .map_err(|e| anyhow!("Error saving index: {:?}", e))
    }

    pub fn from_path(path: &PathBuf) -> Result<Self> {
        let path: &str = path.to_str()
            .with_context(|| format!("'{:?}' cannot to converted to str", path))?;
        let index = HNSWIndex::<f32, IdxID>::load(path)
            .map_err(|e| anyhow!("Error loading index: {:?}", e))?;
        Ok(Self { index })
    }
}

#[cfg(test)]
mod tests {
    use hora::core::ann_index::SerializableIndex;

    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_serialize_deserialize_inner() -> Result<()> {
        let mut index = HNSWIndex::<f32, IdxID>::new(3, &HNSWParams::<f32>::default());
        {

            const DIM: usize = 3;
            const N: usize = 10;
            let data = (0..N)
                .map(|i| {
                    let doc_id = DocumentId(i as u32);
                    let vector: Vec<_> = (0..DIM)
                        .map(|_| {
                            let x = rand::random::<i8>();
                            x as f32 / 10.0
                        })
                        .collect();
                    (vector, doc_id)
                })
                .collect::<Vec<_>>();

            for (vector, doc_id) in data {
                index.add(&vector, IdxID(Some(doc_id))).unwrap();
            }
            index.build(Metric::Manhattan).unwrap();
        }

        let path = generate_new_path();
        let path: &str = path.to_str().unwrap();

        index.dump(path).unwrap();

        let after = HNSWIndex::<f32, IdxID>::load(path).unwrap();

        let before = index.search(&[0.0, 0.0, 0.0], 2);
        let output_after = after.search(&[0.0, 0.0, 0.0], 2);

        assert_eq!(before, output_after);

        Ok(())
    }

    #[test]
    fn test_serialize_deserialize_committed_state() -> Result<()> {
        const DIM: usize = 3;
        const N: usize = 10;

        let mut index = CommittedState::new_in_memory(3);
        {
            
            let data = (0..N)
                .map(|i| {
                    let doc_id = DocumentId(i as u32);
                    let vector: Vec<_> = (0..DIM)
                        .map(|_| {
                            let x = rand::random::<i8>();
                            x as f32 / 10.0
                        })
                        .collect();
                    (vector, doc_id)
                })
                .collect::<Vec<_>>();

            let loaded_index = index.load_in_memory()?;
            for (vector, doc_id) in data {
                loaded_index.add(&vector, doc_id)?;
            }
            loaded_index.build()?;
        }

        let path = generate_new_path();
        index.unload_from_memory(path.clone())?;

        let deserialized = CommittedState::OnFile(path);

        let mut output_before = HashMap::new();
        index.search(&[0.0, 0.0, 0.0], 2, &mut output_before)?;

        let mut output_after = HashMap::new();
        deserialized.search(&[0.0, 0.0, 0.0], 2, &mut output_after)?;

        assert_eq!(output_before, output_after);

        Ok(())
    }
}