use std::{collections::HashMap, sync::RwLock};

use anyhow::Result;

use crate::types::DocumentId;

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

    pub fn search(
        &self,
        target: &[f32],
        result: &mut HashMap<DocumentId, f32>,
    ) -> Result<()>{
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

    fn search(
        &self,
        target: &[f32],
        result: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        self.left.search(target, result)?;
        self.right.search(target, result)?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct UncommittedVectorFieldIndex {
    inner: RwLock<InnerUncommittedVectorFieldIndex>,
    dimension: usize,
}

impl UncommittedVectorFieldIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: Default::default(),
            dimension,
        }
    }

    pub fn insert(&self, data: (DocumentId, Vec<Vec<f32>>)) -> Result<()> {
        let mut inner = match self.inner.write() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };
        inner.insert(data)?;
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
        let tree = if inner.state {
            inner.left.clone()
        } else {
            inner.right.clone()
        };
        let current_state = inner.state;

        // Route the writes to the other side
        inner.state = !current_state;
        drop(inner);

        UncommittedVectorFieldIndexTaken {
            data: tree.data,
            dimension: self.dimension,
        }
    }

    pub fn search(
        &self,
        target: &[f32],
        result: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        let inner = match self.inner.read() {
            Ok(lock) => lock,
            Err(p) => p.into_inner(),
        };

        inner.search(target, result)
    }
}

pub struct UncommittedVectorFieldIndexTaken {
    pub data: Vec<(DocumentId, Vec<VectorWithMagnetude>)>,
    pub dimension: usize,
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

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncommitted_vector_index() -> Result<()> {
        let index = UncommittedVectorFieldIndex::new();

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
*/