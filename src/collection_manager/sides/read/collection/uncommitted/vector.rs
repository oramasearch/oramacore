use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use serde::Serialize;

use crate::types::DocumentId;

type VectorWithMagnetude = (f32, Vec<f32>);

#[derive(Debug)]
pub struct VectorField {
    pub data: Vec<(DocumentId, Vec<VectorWithMagnetude>)>,
    pub dimension: usize,
}

impl VectorField {
    pub fn empty(dimension: usize) -> Self {
        Self {
            data: Vec::new(),
            dimension,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn search(
        &self,
        target: &[f32],
        similarity: f32,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        let magnetude = calculate_magnetude(target);

        for (id, vectors) in &self.data {
            if filtered_doc_ids.is_some_and(|ids| !ids.contains(id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(id) {
                continue;
            }

            for (m, vector) in vectors {
                let score = score_vector(vector, target)?;
                if score < similarity {
                    continue;
                }

                let score = score / (m * magnetude);

                let s = output.entry(*id).or_insert(0.0);
                *s += score;
            }
        }

        Ok(())
    }

    pub fn insert(&mut self, doc_id: DocumentId, vectors: Vec<Vec<f32>>) -> Result<()> {
        let is_different = vectors.iter().any(|v| v.len() != self.dimension);
        if is_different {
            bail!("Vector dimension is different from the field dimension");
        }

        let vectors = vectors
            .into_iter()
            .map(|vector| {
                let magnetude = calculate_magnetude(&vector);
                (magnetude, vector)
            })
            .collect();

        self.data.push((doc_id, vectors));

        Ok(())
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)> + '_ {
        self.data
            .iter()
            .map(|(id, vectors)| (*id, vectors.iter().map(|(_, v)| v.clone()).collect()))
    }

    pub fn get_stats(&self) -> VectorUncommittedFieldStats {
        VectorUncommittedFieldStats {
            len: self.data.len(),
        }
    }
}

fn calculate_magnetude(vector: &[f32]) -> f32 {
    vector.iter().map(|x| x.powi(2)).sum::<f32>()
}

fn score_vector(vector: &[f32], target: &[f32]) -> Result<f32> {
    debug_assert_eq!(
        vector.len(),
        target.len(),
        "Vector and target must have the same length"
    );

    let mag_1 = vector
        .iter()
        .map(|x| x.powi(2))
        .sum::<f32>()
        .sqrt()
        .max(0.0001);
    let mag_2 = target
        .iter()
        .map(|x| x.powi(2))
        .sum::<f32>()
        .sqrt()
        .max(0.0001);
    let dot_product = target.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
    let similarity = dot_product / (mag_1 * mag_2);

    Ok(similarity)
}

#[derive(Serialize)]
pub struct VectorUncommittedFieldStats {
    len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncommitted_vector() {
        let mut index = VectorField::empty(3);
        index
            .insert(DocumentId(0), vec![vec![1.0, 0.0, 0.0]])
            .unwrap();
        index
            .insert(DocumentId(1), vec![vec![1.0, 0.0001, 0.0]])
            .unwrap();
        index
            .insert(DocumentId(2), vec![vec![0.0, 0.0, 1.0]])
            .unwrap();

        // With similarity
        let mut output = HashMap::new();
        index
            .search(
                &[1.0, 0.0, 0.0],
                0.6,
                None,
                &mut output,
                &Default::default(),
            )
            .unwrap();
        assert_eq!(
            HashSet::from([DocumentId(0), DocumentId(1)]),
            output.keys().cloned().collect()
        );

        // With similarity to 0
        let mut output = HashMap::new();
        index
            .search(
                &[1.0, 0.0, 0.0],
                0.0,
                None,
                &mut output,
                &Default::default(),
            )
            .unwrap();
        assert_eq!(
            HashSet::from([DocumentId(0), DocumentId(1), DocumentId(2)]),
            output.keys().cloned().collect()
        );
    }

    #[test]
    fn test_score_vector() {
        let vector1 = vec![0.1, 0.0, 0.0];
        let vector2 = vec![0.1, 0.001, 0.0];
        let vector3 = vec![0.0, 1.0, 0.0];

        assert!(
            score_vector(&vector1, &vector1).unwrap() > score_vector(&vector1, &vector2).unwrap()
        );
        assert!(
            score_vector(&vector1, &vector2).unwrap() > score_vector(&vector1, &vector3).unwrap()
        );
    }
}
