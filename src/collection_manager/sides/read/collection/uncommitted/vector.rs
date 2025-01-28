use std::collections::HashMap;

use anyhow::Result;

use crate::types::DocumentId;

type VectorWithMagnetude = (f32, Vec<f32>);

#[derive(Debug)]
pub struct VectorField {
    pub data: Vec<(DocumentId, Vec<VectorWithMagnetude>)>,
}

impl VectorField {
    pub fn empty() -> Self {
        Self { data: Vec::new() }
    }

    pub fn search(&self, target: &[f32], output: &mut HashMap<DocumentId, f32>) -> Result<()> {
        let magnetude = calculate_magnetude(target);

        for (id, vectors) in &self.data {
            for (m, vector) in vectors {
                let score = score_vector(vector, target)?;

                if score <= 0.0 {
                    continue;
                }

                let score = score / (m * magnetude);

                let s = output.entry(*id).or_insert(0.0);
                *s += score;
            }
        }

        Ok(())
    }

    pub fn insert(&mut self, doc_id: DocumentId, vectors: Vec<Vec<f32>>) {
        let vectors = vectors
            .into_iter()
            .map(|vector| {
                let magnetude = calculate_magnetude(&vector);
                (magnetude, vector)
            })
            .collect();

        self.data.push((doc_id, vectors));
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
    let distance: f32 = vector
        .iter()
        .zip(target)
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let distance = distance.sqrt();

    let score = 1.0 / distance.max(0.01);

    Ok(score)
}
