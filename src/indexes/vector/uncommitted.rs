use std::{collections::HashMap, sync::RwLock};

use anyhow::Result;
use tracing::warn;

use crate::{collection_manager::sides::Offset, offset_storage::OffsetStorage, types::DocumentId};

type VectorWithMagnetude = (f32, Vec<f32>);

#[derive(Debug, Default, Clone)]
pub struct InnerInnerUncommittedVectorFieldIndex {
    data: Vec<(DocumentId, Vec<VectorWithMagnetude>)>,
}

impl InnerInnerUncommittedVectorFieldIndex {
    pub fn insert(&mut self, data: (DocumentId, Vec<Vec<f32>>)) -> Result<()> {
        let (id, vectors) = data;

        let vectors = vectors
            .into_iter()
            .map(|vector| {
                let magnetude = calculate_magnetude(&vector);
                (magnetude, vector)
            })
            .collect();

        self.data.push((id, vectors));
        Ok(())
    }

    pub fn search(&self, target: &[f32], result: &mut HashMap<DocumentId, f32>) -> Result<()> {
        let magnetude = calculate_magnetude(target);

        for (id, vectors) in &self.data {
            for (m, vector) in vectors {
                let score = score_vector(vector, target)?;

                if score <= 0.0 {
                    continue;
                }

                let score = score / (m * magnetude);

                let s = result.entry(*id).or_insert(0.0);
                *s += score;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct InnerUncommittedVectorFieldIndex {
    left: InnerInnerUncommittedVectorFieldIndex,
    right: InnerInnerUncommittedVectorFieldIndex,
    state: bool,
}

impl InnerUncommittedVectorFieldIndex {
    fn insert(&mut self, data: (DocumentId, Vec<Vec<f32>>)) -> Result<()> {
        let inner = if self.state {
            &mut self.left
        } else {
            &mut self.right
        };
        inner.insert(data)?;
        Ok(())
    }

    fn search(&self, target: &[f32], result: &mut HashMap<DocumentId, f32>) -> Result<()> {
        self.left.search(target, result)?;
        self.right.search(target, result)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct UncommittedVectorFieldIndex {
    inner: RwLock<(OffsetStorage, InnerUncommittedVectorFieldIndex)>,
    dimension: usize,
}

impl UncommittedVectorFieldIndex {
    pub fn new(dimension: usize, offset: Offset) -> Self {
        let offset_storage = OffsetStorage::new();
        offset_storage.set_offset(offset);
        Self {
            inner: RwLock::new((offset_storage, InnerUncommittedVectorFieldIndex::default())),
            dimension,
        }
    }

    pub fn insert(&self, offset: Offset, data: (DocumentId, Vec<Vec<f32>>)) -> Result<()> {
        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        if offset <= inner.0.get_offset() {
            warn!("Skip insert vector with lower offset");
            return Ok(());
        }
        inner.1.insert(data)?;
        inner.0.set_offset(offset);

        Ok(())
    }

    pub fn take(&self) -> UncommittedVectorFieldIndexTaken {
        // There is a subtle bug here: this method should be called only once at time
        // If this method is called twice at the same time, the result will be wrong
        // This is because:
        // - the state is flipped
        // - the tree is cleared on `UncommittedVectorFieldIndexTaken` drop (which happens at the end of the commit)
        // So, if we flipped the state twice, the tree will be cleared twice and we could lose data
        // TODO: investigate how to make this method safe to be called multiple times

        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        let tree = if inner.1.state {
            inner.1.left.clone()
        } else {
            inner.1.right.clone()
        };
        let current_state = inner.1.state;

        let current_offset = inner.0.get_offset();

        // Route the writes to the other side
        inner.1.state = !current_state;
        drop(inner);

        UncommittedVectorFieldIndexTaken {
            data: tree.data,
            dimension: self.dimension,
            index: self,
            state: current_state,
            current_offset,
        }
    }

    pub fn search(&self, target: &[f32], result: &mut HashMap<DocumentId, f32>) -> Result<()> {
        let inner = match self.inner.read() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };

        inner.1.search(target, result)
    }
}

pub struct UncommittedVectorFieldIndexTaken<'index> {
    pub data: Vec<(DocumentId, Vec<VectorWithMagnetude>)>,
    pub dimension: usize,
    index: &'index UncommittedVectorFieldIndex,
    state: bool,
    current_offset: Offset,
}

impl UncommittedVectorFieldIndexTaken<'_> {
    pub fn data(&mut self) -> Vec<(DocumentId, Vec<VectorWithMagnetude>)> {
        std::mem::take(&mut self.data)
    }
    pub fn current_offset(&self) -> Offset {
        self.current_offset
    }
    pub fn close(self) {
        let mut lock = match self.index.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        let inner = if self.state {
            &mut lock.1.left
        } else {
            &mut lock.1.right
        };
        inner.data.clear();
    }
}

fn score_vector(vector: &[f32], target: &[f32]) -> Result<f32> {
    debug_assert_eq!(
        vector.len(),
        target.len(),
        "Vector and target must have the same length"
    );
    let distance: f32 = vector
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let distance = distance.sqrt();

    let score = 1.0 / distance.max(0.01);

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
        let index = UncommittedVectorFieldIndex::new(2, Offset(0));

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

        index.insert(Offset(1), (DocumentId(1), vec![vec![1.0, 0.0]]))?;
        index.insert(Offset(2), (DocumentId(2), vec![vec![0.0, 1.0]]))?;
        index.insert(Offset(3), (DocumentId(4), vec![vec![-1.0, 0.0]]))?;
        index.insert(Offset(4), (DocumentId(5), vec![vec![0.0, -1.0]]))?;

        let mut result_1 = HashMap::new();
        index.search(&[1.0, 0.0], &mut result_1)?;

        assert_eq!(result_1.get(&DocumentId(1)), Some(&100.0));
        assert!(*result_1.get(&DocumentId(2)).unwrap() < 1.0);
        assert!(*result_1.get(&DocumentId(4)).unwrap() < 1.0);
        assert!(*result_1.get(&DocumentId(5)).unwrap() < 1.0);

        let mut result_a = HashMap::new();
        index.search(&[1.0, 1.0], &mut result_a)?;
        assert_eq!(result_a.get(&DocumentId(1)), Some(&0.5));
        assert_eq!(result_a.get(&DocumentId(2)), Some(&0.5));

        let mut result_b = HashMap::new();
        index.search(&[0.25, 1.0], &mut result_b)?;
        assert!(result_b.get(&DocumentId(2)).unwrap() > result_b.get(&DocumentId(1)).unwrap());

        Ok(())
    }
}
