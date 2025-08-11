use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use filters::FilterResult;
use serde::Serialize;

use crate::{ai::OramaModel, types::DocumentId};

type VectorWithMagnetude = (f32, Vec<f32>);

#[derive(Debug)]
pub struct UncommittedVectorField {
    field_path: Box<[String]>,
    model: OramaModel,

    pub data: Vec<(DocumentId, Vec<VectorWithMagnetude>)>,
    pub dimension: usize,
}

impl UncommittedVectorField {
    pub fn empty(field_path: Box<[String]>, model: OramaModel) -> Self {
        Self {
            field_path,
            data: Vec::new(),
            dimension: model.dimensions(),
            model,
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.data = Default::default();
    }

    pub fn get_model(&self) -> OramaModel {
        self.model
    }

    fn is_e5_model(&self) -> bool {
        matches!(
            self.model,
            OramaModel::MultilingualE5Small
                | OramaModel::MultilingualE5Base
                | OramaModel::MultilingualE5Large
        )
    }

    pub fn search(
        &self,
        target: &[f32],
        similarity: f32,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
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
                let mut score = score_vector(vector, target)?;

                // Rescale E5 model scores from [0.7, 1.0] to [0.0, 1.0]
                if self.is_e5_model() {
                    score = rescale_e5_similarity_score(score);
                }

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

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)> + '_ {
        self.data
            .iter()
            .map(|(id, vectors)| (*id, vectors.iter().map(|(_, v)| v.clone()).collect()))
    }

    pub fn stats(&self) -> UncommittedVectorFieldStats {
        UncommittedVectorFieldStats {
            document_count: self.data.len(),
            vector_count: self.data.iter().map(|(_, v)| v.len()).sum(),
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

/// Rescales E5 embedding model cosine similarity scores from their typical range [0.7, 1.0] to [0.0, 1.0].
/// E5 models produce similarity scores that rarely go below 0.7, making the effective range much narrower.
/// This rescaling helps normalize the scores to use the full [0.0, 1.0] range for better search ranking.
fn rescale_e5_similarity_score(score: f32) -> f32 {
    const E5_MIN_SCORE: f32 = 0.7;
    const E5_MAX_SCORE: f32 = 1.0;

    // Clamp the score to the expected E5 range to handle edge cases
    let clamped_score = score.clamp(E5_MIN_SCORE, E5_MAX_SCORE);

    // Rescale from [0.7, 1.0] to [0.0, 1.0]
    (clamped_score - E5_MIN_SCORE) / (E5_MAX_SCORE - E5_MIN_SCORE)
}

#[derive(Serialize, Debug)]
pub struct UncommittedVectorFieldStats {
    pub document_count: usize,
    pub vector_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncommitted_vector() {
        let mut index = UncommittedVectorField::empty(
            vec!["".to_string()].into_boxed_slice(),
            OramaModel::BgeSmall,
        );
        let mut v1 = vec![0.0; 384];
        v1[0] = 1.0;
        let mut v2 = vec![0.0; 384];
        v2[0] = 1.0;
        v2[1] = 0.0001;
        let mut v3 = vec![0.0; 384];
        v3[2] = 1.0;
        index.insert(DocumentId(0), vec![v1]).unwrap();
        index.insert(DocumentId(1), vec![v2]).unwrap();
        index.insert(DocumentId(2), vec![v3]).unwrap();

        let mut target = vec![0.0; 384];
        target[0] = 1.0;

        // With similarity
        let mut output = HashMap::new();
        index
            .search(&target, 0.6, None, &mut output, &Default::default())
            .unwrap();
        assert_eq!(
            HashSet::from([DocumentId(0), DocumentId(1)]),
            output.keys().cloned().collect()
        );

        // With similarity to 0
        let mut output = HashMap::new();
        index
            .search(&target, 0.0, None, &mut output, &Default::default())
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
