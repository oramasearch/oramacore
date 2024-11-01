use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

pub const DEFAULT_SUBVECTORS: usize = 16;
pub const DEFAULT_CENTROIDS: usize = 256;

pub fn train_pq_codebooks(
    embeddings: &Array2<f32>,
    num_subvectors: usize,
    num_centroids: usize,
) -> Vec<Array2<f32>> {
    let embedding_dim = embeddings.shape()[1];
    let subvector_dim = embedding_dim / num_subvectors;

    let mut codebooks = Vec::new();

    for i in 0..num_subvectors {
        let subvector_data = embeddings.slice(s![.., i * subvector_dim..(i + 1) * subvector_dim]);
        let codebook = train_codebook(subvector_data, num_centroids);
        codebooks.push(codebook);
    }

    codebooks
}

pub fn quantize_embedding(
    embedding: &Vec<f32>,
    codebooks: &Vec<Array2<f32>>,
) -> Vec<usize> {
    let embedding_dim = embedding.len();
    let num_subvectors = codebooks.len();
    let subvector_dim = embedding_dim / num_subvectors;

    let embedding_array = ArrayView1::from(embedding);

    let mut quantized_embedding = Vec::new();
    for (i, codebook) in codebooks.iter().enumerate() {
        let subvector = embedding_array.slice(s![i * subvector_dim..(i + 1) * subvector_dim]);
        let code_idx = quantize_subvector(subvector, codebook);
        quantized_embedding.push(code_idx);
    }
    quantized_embedding
}

pub fn quantize_embeddings(
    embeddings: &Array2<f32>,
    codebooks: &Vec<Array2<f32>>,
) -> Vec<Vec<usize>> {
    embeddings
        .axis_iter(Axis(0))
        .map(|embedding| {
            let embedding_vec = embedding.to_vec();
            quantize_embedding(&embedding_vec, codebooks)
        })
        .collect()
}

fn train_codebook(data: ArrayView2<f32>, num_centroids: usize) -> Array2<f32> {
    let labels = Array1::from_elem(data.nrows(), 0);
    let dataset = DatasetBase::new(data.to_owned(), labels);
    let model = KMeans::params(num_centroids)
        .fit(&dataset)
        .expect("KMeans training failed");

    model.centroids().to_owned()
}

fn quantize_subvector(subvector: ArrayView1<f32>, codebook: &Array2<f32>) -> usize {
    codebook
        .axis_iter(Axis(0))
        .enumerate()
        .min_by(|(_, centroid_a), (_, centroid_b)| {
            let dist_a = subvector
                .iter()
                .zip(centroid_a.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            let dist_b = subvector
                .iter()
                .zip(centroid_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .map(|(idx, _)| idx)
        .unwrap()
}
