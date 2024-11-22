use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::collection_manager::FieldId;
use crate::document_storage::DocumentId;
use crate::embeddings::LoadedModel;
use anyhow::{anyhow, Result};
use dashmap::{DashMap, Entry};
use hora::core::metrics::Metric::Manhattan;
use hora::core::node::IdxType;
use hora::index::hnsw_idx;
use hora::{core::ann_index::ANNIndex, index::hnsw_idx::HNSWIndex};
use serde::Serialize;
use tracing::{trace, warn};

#[derive(Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash)]
struct IdxID(Option<DocumentId>);

pub struct VectorIndex {
    indexes: DashMap<FieldId, (Arc<LoadedModel>, HNSWIndex<f32, IdxID>)>,
}

pub struct VectorIndexConfig {}

impl IdxType for IdxID {}

impl VectorIndex {
    pub fn try_new(_config: VectorIndexConfig) -> Result<Self> {
        Ok(Self {
            indexes: DashMap::new(),
        })
    }

    pub fn add_field(&self, field_id: FieldId, orama_model: Arc<LoadedModel>) -> Result<()> {
        let entry = self.indexes.entry(field_id);
        match entry {
            Entry::Occupied(_) => Err(anyhow!("Field already exists")),
            Entry::Vacant(entry) => {
                let idx = hnsw_idx::HNSWIndex::<f32, IdxID>::new(
                    orama_model.dimensions(),
                    &hora::index::hnsw_params::HNSWParams::<f32>::default(),
                );
                entry.insert((orama_model, idx));

                Ok(())
            }
        }
    }

    pub fn insert_batch(&self, data: Vec<(DocumentId, FieldId, Vec<Vec<f32>>)>) -> Result<()> {
        let mut field_ids = HashSet::new();
        for (id, field_id, vectors) in data {
            let index = self.indexes.get_mut(&field_id);
            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            for vector in vectors {
                index
                    .1
                    .add(&vector, IdxID(Some(id)))
                    .map_err(|e| anyhow!("Error adding vector: {:?}", e))?;
            }

            field_ids.insert(field_id);
        }

        for field_id in field_ids {
            let index = self.indexes.get_mut(&field_id);
            let mut index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            index
                .1
                .build(Manhattan)
                .map_err(|e| anyhow!("Error building index: {:?}", e))?;
        }

        Ok(())
    }

    pub fn search(
        &self,
        field_ids: Vec<FieldId>,
        target: &[f32],
        k: usize,
    ) -> Result<HashMap<DocumentId, f32>> {
        trace!("Searching for target: {:?} in fields {:?}", target, field_ids);

        let mut output = HashMap::new();

        for field_id in field_ids {
            let index = self.indexes.get(&field_id);
            let index = match index {
                Some(index) => index,
                None => return Err(anyhow!("Field {:?} not found", field_id)),
            };

            let search_output = index.1.search(target, k).into_iter();
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

        trace!("VectorIndex output: {:?}", output);

        Ok(output)
    }
}
