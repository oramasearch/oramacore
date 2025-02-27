use criterion::Criterion;
use criterion::{black_box, criterion_group, criterion_main};

use hnsw2::core::metrics::Metric;
use oramacore::indexes::hnsw::Search;
use oramacore::{
    indexes::{
        hnsw::{Builder, HnswMap, Point},
        hnsw2::HNSW2Index,
    },
    types::{Document, DocumentId},
};
use rand::distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

const DIM: usize = 64;
const N: usize = 10_000;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct PPoint(Vec<f32>);

impl Point for PPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance metric
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

fn generate_sample() -> Vec<Vec<f32>> {
    let normal = Uniform::new(0.0, 10.0).unwrap();
    let n = N;
    let dimension = DIM;
    (0..n)
        .map(|_| {
            (0..dimension)
                .map(|_| normal.sample(&mut rand::rng()))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn build_hnsw(_dim: usize, data: Vec<Vec<f32>>) -> HnswMap<PPoint, DocumentId> {
    let values = (0..data.len())
        .map(|x| DocumentId(x as u64))
        .collect::<Vec<DocumentId>>();
    let points = data.into_iter().map(|x| PPoint(x)).collect::<Vec<PPoint>>();

    let map: HnswMap<PPoint, DocumentId> = Builder::default().build(points, values);

    map
}

fn build_hnsw2(dim: usize, data: Vec<Vec<f32>>) -> HNSW2Index {
    let mut index = HNSW2Index::new(dim);
    for (i, sample) in data.into_iter().enumerate() {
        index.add(&sample, DocumentId(i as u64)).unwrap();
    }
    index.build().unwrap();

    index
}

pub fn hnsw(c: &mut Criterion) {
    let data = generate_sample();
    let data2 = data.clone();

    c.bench_function("build hnsw", |b| {
        b.iter(|| {
            let _ = build_hnsw(DIM, black_box(data.clone()));
        });
    });
    c.bench_function("build hnsw2", |b| {
        b.iter(|| {
            let _ = build_hnsw2(DIM, black_box(data2.clone()));
        });
    });

    c.bench_function("search hnsw", |b| {
        let index = build_hnsw(DIM, data.clone());
        b.iter(|| {
            let mut search = Search::default();
            let iter = index.search(black_box(&PPoint(data[0].clone())), &mut search);
            let _ = iter.take(10).collect::<Vec<_>>();
        });
    });
    c.bench_function("search hnsw2", |b| {
        let index = build_hnsw2(DIM, data.clone());
        b.iter(|| {
            let _ = index.search(black_box(data[0].clone()), 10);
        });
    });
}

criterion_group!(benches, hnsw);
criterion_main!(benches);
