use anyhow::Result;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use reductive::pq::Pq;
use reductive::pq::{QuantizeVector, TrainPq};

pub const DEFAULT_N_SUBVECTORS: usize = 32; // usually we want dimensions/32 or dimensions/16
pub const DEFAULT_N_CODES: usize = 256;

pub struct ProductQuantizer {
    pub pq: Pq<f32>,
}

impl ProductQuantizer {
    pub fn try_new(dataset: Vec<Vec<f32>>) -> Result<Self> {
        let mut rng = ChaCha8Rng::from_entropy();
        let to_array2 = Self::vec_to_array2(dataset);

        let pq = Pq::<f32>::train_pq_using(
            DEFAULT_N_SUBVECTORS,
            2,
            200,
            1,
            to_array2.unwrap().view(),
            &mut rng,
        )?;

        Ok(Self { pq })
    }

    pub fn quantize(&self, dataset: Vec<Vec<f32>>) -> Array2<u8> {
        let new_vector_to_array2 = Self::vec_to_array2(dataset).unwrap();
        self.pq.quantize_batch::<u8, _>(new_vector_to_array2.view())
    }

    fn vec_to_array2<T: Clone>(vec: Vec<Vec<T>>) -> Option<Array2<T>> {
        if vec.is_empty() {
            return None;
        }

        let rows = vec.len();
        let cols = vec[0].len();

        if !vec.iter().all(|row| row.len() == cols) {
            return None;
        }

        let flat: Vec<T> = vec.into_iter().flatten().collect();
        Some(Array2::from_shape_vec((rows, cols), flat).unwrap())
    }
}
