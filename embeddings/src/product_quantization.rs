// Ported from https://github.com/shubham0204/pq.rs/blob/main/src/pq.rs
// Apache 2.0 license at the moment of porting https://github.com/shubham0204/pq.rs/commit/bf00d22c1e6374116ca987ef4b3e6cbdc4b9c374.

use num_traits::cast::FromPrimitive;
use num_traits::sign::Unsigned;
use num_traits::ToPrimitive;
use rand::seq::SliceRandom;
use std::vec;

pub const DEFAULT_N_SUBVECTORS: usize = 32; // usually we want dimensions/32 or dimensions/16
pub const DEFAULT_N_CODES: usize = 256;

pub enum DistanceMetric {
    Euclidean,
    Dot,
}

pub struct ProductQuantizer {
    n_subvectors: usize,
    n_codes: usize,
    src_vec_dims: usize,
    codewords: Vec<Vec<Vec<f32>>>,
    distance_metric: fn(&[f32], &[f32]) -> f32,
}

impl ProductQuantizer {
    pub fn new(
        n_subvectors: usize,
        n_codes: usize,
        src_vec_dims: usize,
        metric: DistanceMetric,
    ) -> Self {
        assert!(
            n_subvectors <= src_vec_dims,
            "`n_subvectors` has to be smaller than or equal to `src_vec_dims`."
        );
        match metric {
            DistanceMetric::Euclidean => ProductQuantizer {
                n_subvectors,
                n_codes,
                src_vec_dims,
                codewords: Vec::new(),
                distance_metric: ProductQuantizer::euclid_distance,
            },
            DistanceMetric::Dot => ProductQuantizer {
                n_subvectors,
                n_codes,
                src_vec_dims,
                codewords: Vec::new(),
                distance_metric: ProductQuantizer::dot,
            },
        }
    }

    pub fn fit(self: &mut ProductQuantizer, src_vectors: &Vec<Vec<f32>>, iterations: usize) {
        for vec in src_vectors {
            assert!(
                vec.len() == self.src_vec_dims,
                "Each vector in `src_vectors` must have dims equal to `src_vec_dims`"
            );
        }

        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors;

        self.codewords = Vec::new();
        for m in 0..self.n_subvectors {
            let mut sub_vectors_m: Vec<Vec<f32>> = Vec::new();
            for vec in src_vectors {
                sub_vectors_m.push(vec[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec());
            }

            self.codewords
                .push(self.kmeans(&sub_vectors_m, self.n_codes, iterations));
        }
    }

    pub fn encode<T>(self: &ProductQuantizer, vectors: &Vec<Vec<f32>>) -> Vec<Vec<T>>
    where
        T: Unsigned + FromPrimitive,
    {
        for vec in vectors {
            assert!(
                vec.len() == self.src_vec_dims,
                "Each vector in `vectors` must have dims equal to `src_vec_dims`"
            );
        }
        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors;
        let mut vector_codes: Vec<Vec<T>> = Vec::new();
        for vec in vectors {
            let mut subvectors: Vec<Vec<f32>> = Vec::new();
            for m in 0..self.n_subvectors {
                subvectors.push(vec[(m * sub_vec_dims)..((m + 1) * sub_vec_dims)].to_vec());
            }
            vector_codes.push(self.vector_quantize(&subvectors));
        }
        vector_codes
    }

    pub fn search<T>(self: &ProductQuantizer, queries: &[Vec<f32>], codes: &[Vec<T>]) -> Vec<usize>
    where
        T: Unsigned + FromPrimitive + ToPrimitive,
    {
        for vec in queries {
            assert!(
                vec.len() == self.src_vec_dims,
                "Each vector in `queries` must have dims equal to `src_vec_dims`"
            );
        }
        for code in codes {
            assert!(
                code.len() == self.n_subvectors,
                "Each code in `codes` must have dims equal to `n_codes`"
            );
        }
        let sub_vec_dims: usize = self.src_vec_dims / self.n_subvectors;
        let mut distances: Vec<usize> = Vec::new();
        for query in queries {
            let mut min_distance = f32::MAX;
            let mut min_distance_index = 0;
            for (n, code) in codes.iter().enumerate() {
                let mut distance = 0.0;
                for m in 0..self.n_subvectors {
                    let query_sub: Vec<f32> =
                        query[m * sub_vec_dims..((m + 1) * sub_vec_dims)].to_vec();

                    distance += (self.distance_metric)(
                        &query_sub,
                        &self.codewords[m][code[m].to_usize().unwrap()],
                    );
                }
                if min_distance > distance {
                    min_distance = distance;
                    min_distance_index = n;
                }
            }
            distances.push(min_distance_index);
        }
        distances
    }

    fn vector_quantize<T>(self: &ProductQuantizer, vector: &[Vec<f32>]) -> Vec<T>
    where
        T: FromPrimitive + Unsigned,
    {
        let mut codes: Vec<T> = Vec::new();
        for (m, subvector) in vector.iter().enumerate() {
            let mut min_distance: f32 = f32::MAX;
            let mut min_distance_code_index: T = T::from_u8(0).unwrap();
            for (k, code) in self.codewords[m].iter().enumerate() {
                let distance = (self.distance_metric)(subvector, code);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_code_index = T::from_usize(k).unwrap();
                }
            }
            codes.push(min_distance_code_index);
        }

        codes
    }

    fn kmeans(
        self: &ProductQuantizer,
        vecs: &Vec<Vec<f32>>,
        n_clusters: usize,
        iter: usize,
    ) -> Vec<Vec<f32>> {
        let mut assigned_centroids: Vec<Vec<f32>> = vec![Vec::new(); vecs.len()];
        let mut centroids: Vec<Vec<f32>> = vecs
            .choose_multiple(&mut rand::thread_rng(), n_clusters)
            .cloned()
            .collect();

        let vec_dims: usize = vecs[0].len();

        for _ in 0..iter {
            for i in 0..vecs.len() {
                let mut min_centroid_distance: f32 = f32::MAX;
                let mut min_centroid: Vec<f32> = centroids[0].clone();
                for centroid in &centroids {
                    // Calculate distance between ith vector and `centroid`
                    let distance: f32 = (self.distance_metric)(&vecs[i], centroid);
                    if distance < min_centroid_distance {
                        min_centroid_distance = distance;
                        min_centroid = centroid.clone();
                    }
                }
                assigned_centroids[i] = min_centroid;
            }

            for i in 0..n_clusters {
                let mut vec_sum: Vec<f32> = vec![0.0; vec_dims];
                let mut count: usize = 0;

                for j in 0..assigned_centroids.len() {
                    if assigned_centroids[j] == centroids[i] {
                        ProductQuantizer::vec_add(&mut vec_sum, &vecs[j]);
                        count += 1;
                    }
                }
                ProductQuantizer::vec_scale(&mut vec_sum, 1.0 / (count as f32));
                centroids[i] = vec_sum;
            }
        }

        centroids
    }

    fn vec_scale(vec: &mut Vec<f32>, scale: f32) {
        for element in vec.iter_mut() {
            *element *= scale;
        }
    }

    fn vec_add(vec1: &mut Vec<f32>, vec2: &[f32]) {
        for i in 0..vec1.len() {
            vec1[i] += vec2[i];
        }
    }

    fn euclid_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut squared_diff_sum = 0.0;
        for i in 0..vec1.len() {
            squared_diff_sum += (vec1[i] - vec2[i]).powi(2);
        }
        squared_diff_sum.sqrt()
    }

    fn dot(vec1: &[f32], vec2: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        for i in 0..vec1.len() {
            dot_product += vec1[i] * vec2[i];
        }
        dot_product
    }
}
