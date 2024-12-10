use std::{cmp::Reverse, collections::HashMap, mem::take, sync::RwLock};

use anyhow::Result;
use ordered_float::NotNan;
use tracing::info;

use crate::{
    capped_heap::CappedHeap, collection_manager::dto::FieldId, document_storage::DocumentId,
};

type VectorWithMagnetude = (f32, Vec<f32>);

pub struct UncommittedVectorIndex {
    data: RwLock<Vec<(DocumentId, FieldId, Vec<VectorWithMagnetude>)>>,
}

impl UncommittedVectorIndex {
    pub fn new() -> Self {
        Self {
            data: Default::default(),
        }
    }

    pub fn insert(&self, data: (DocumentId, FieldId, Vec<Vec<f32>>)) -> Result<()> {
        let mut lock = match self.data.write() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };

        let (id, field_id, vectors) = data;

        let vectors = vectors
            .into_iter()
            .map(|vector| {
                let magnetude = calculate_magnetude(&vector);
                (magnetude, vector)
            })
            .collect();

        lock.push((id, field_id, vectors));
        Ok(())
    }

    pub fn consume(&self) -> Vec<(DocumentId, FieldId, Vec<VectorWithMagnetude>)> {
        let mut lock = match self.data.write() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };

        let v = &mut *lock;
        take(v)
    }

    pub fn search(
        &self,
        field_ids: &[FieldId],
        target: &[f32],
        limit: usize,
    ) -> Result<HashMap<DocumentId, f32>> {
        let lock = match self.data.read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };

        info!(
            "Searching for target: {:?} for fields {:?} in {} vectors",
            target,
            field_ids,
            lock.len()
        );

        let magnetude = calculate_magnetude(target);

        let mut capped_heap = CappedHeap::new(limit);
        for (id, field_id, vectors) in lock.iter() {
            if !field_ids.contains(field_id) {
                continue;
            }

            for (m, vector) in vectors {
                let score = score_vector(vector, target)?;

                if score <= 0.0 {
                    continue;
                }

                let score = score / (m * magnetude);

                let score = match NotNan::new(score) {
                    Ok(score) => score,
                    // Skip vectors with NaN scores
                    Err(_) => continue,
                };

                capped_heap.insert(score, id);
            }
        }

        let result = capped_heap.into_top();

        let result = result
            .into_iter()
            .map(|Reverse((score, id))| (*id, score.into_inner()))
            .collect();

        Ok(result)
    }
}

fn score_vector(vector: &[f32], target: &[f32]) -> Result<f32> {
    debug_assert_eq!(
        vector.len(),
        target.len(),
        "Vector and target must have the same length"
    );

    let score: f32 = vector.iter().zip(target).map(|(a, b)| a * b).sum();

    Ok(score)
}

fn calculate_magnetude(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x.powi(2)).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncommitted_vector_index() -> Result<()> {
        let index = UncommittedVectorIndex::new();

        //      |
        //      2b  a
        //      |
        //      |
        //      |
        // -4-------1-
        //      |
        //      |
        //      |
        //      5
        //      |

        index.insert((DocumentId(1), FieldId(1), vec![vec![1.0, 0.0]]))?;
        index.insert((DocumentId(2), FieldId(1), vec![vec![0.0, 1.0]]))?;
        index.insert((DocumentId(4), FieldId(1), vec![vec![-1.0, 0.0]]))?;
        index.insert((DocumentId(5), FieldId(1), vec![vec![0.0, -1.0]]))?;

        let result_1 = index.search(&[FieldId(1)], &[1.0, 0.0], 4)?;

        assert_eq!(result_1.get(&DocumentId(1)), Some(&1.0));
        // The point 4 has a negative score, so it should not be included.
        // Is correct we exclude it?
        // TODO: think about this.
        assert_eq!(result_1.get(&DocumentId(4)), None);

        let result_a = index.search(&[FieldId(1)], &[1.0, 1.0], 6)?;
        assert_eq!(result_a.get(&DocumentId(1)), Some(&0.5));
        assert_eq!(result_a.get(&DocumentId(2)), Some(&0.5));

        let result_b = index.search(&[FieldId(1)], &[0.25, 1.0], 6)?;
        assert!(result_b.get(&DocumentId(2)).unwrap() > result_b.get(&DocumentId(1)).unwrap());

        Ok(())
    }
}
