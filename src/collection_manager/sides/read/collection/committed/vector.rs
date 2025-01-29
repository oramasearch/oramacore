use std::collections::{HashMap, HashSet};

use anyhow::Result;
use hora::{
    core::{ann_index::ANNIndex, node::IdxType},
    index::hnsw_idx::HNSWIndex,
};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::types::DocumentId;

#[derive(
    Clone, Default, core::fmt::Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Hash, Deserialize,
)]
pub struct IdxID(Option<DocumentId>);
impl IdxType for IdxID {}

#[derive(Debug)]
pub struct VectorField {
    inner: HNSWIndex<f32, IdxID>,
}

impl VectorField {
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
}
