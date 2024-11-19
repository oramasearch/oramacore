use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;
use std::sync::Arc;
use storage::Storage;
use string_index::scorer::bm25::BM25Score;
use string_index::StringIndex;
use types::{DocumentId, FieldId};

#[derive(Debug)]
struct BenchmarkResults {
    docs_count: usize,
    index_size_bytes: u64,
    indexing_time_ms: u64,
    avg_search_time_ms: f64,
    memory_usage_mb: f64,
    throughput_docs_per_sec: f64,
}

fn generate_test_data(
    num_docs: usize,
) -> Vec<(DocumentId, Vec<(FieldId, Vec<(String, Vec<String>)>)>)> {
    let mut rng = rand::thread_rng();

    let vocabulary: Vec<String> = (0..5000)
        .map(|_| {
            let len = rng.gen_range(3..8);
            (0..len)
                .map(|_| (b'a' + rng.gen_range(0..26)) as char)
                .collect()
        })
        .collect();

    (0..num_docs)
        .map(|i| {
            let doc_id = DocumentId(i as u64);
            let num_fields = rng.gen_range(1..=3);

            let fields = (0..num_fields)
                .map(|field_i| {
                    let field_id = FieldId(field_i);
                    let num_terms = rng.gen_range(5..50);

                    let terms = (0..num_terms)
                        .map(|_| {
                            let word = vocabulary.choose(&mut rng).unwrap().clone();
                            let stems = (0..rng.gen_range(0..2))
                                .map(|_| vocabulary.choose(&mut rng).unwrap().clone())
                                .collect();
                            (word, stems)
                        })
                        .collect();

                    (field_id, terms)
                })
                .collect();

            (doc_id, fields)
        })
        .collect()
}

fn benchmark_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(50);

    for &size in &[1000, 5000, 10_000] {
        group.bench_with_input(BenchmarkId::new("simple_index", size), &size, |b, &size| {
            let data = generate_test_data(size);
            let batch: HashMap<_, _> = data.into_iter().collect();

            b.iter(|| {
                let tmp_dir = tempdir::TempDir::new("bench_index").unwrap();
                let storage = Arc::new(Storage::from_path(tmp_dir.path().to_str().unwrap()));
                let index = StringIndex::new(Arc::clone(&storage));

                index
                    .insert_multiple(batch.clone())
                    .expect("Insertion failed");
            });
        });
    }

    group.finish();
}

fn benchmark_batch_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_indexing");
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(30);

    for &size in &[1000, 5000, 10_000] {
        for &batch_size in &[100, 500] {
            group.bench_with_input(
                BenchmarkId::new(format!("batch_size_{}", batch_size), size),
                &size,
                |b, &size| {
                    let data = generate_test_data(size);
                    let batch: HashMap<_, _> = data.into_iter().collect();

                    b.iter(|| {
                        let tmp_dir = tempdir::TempDir::new("bench_index").unwrap();
                        let storage =
                            Arc::new(Storage::from_path(tmp_dir.path().to_str().unwrap()));
                        let index = StringIndex::new(Arc::clone(&storage));
                        index
                            .insert_multiple(batch.clone())
                            .expect("Batch insertion failed");
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    group.measurement_time(std::time::Duration::from_secs(20));

    let mut rng = rand::thread_rng();
    let vocabulary: Vec<String> = (0..500)
        .map(|_| {
            let len = rng.gen_range(3..8);
            (0..len)
                .map(|_| (b'a' + rng.gen_range(0..26)) as char)
                .collect()
        })
        .collect();

    let queries: Vec<Vec<String>> = (0..100)
        .map(|_| {
            let num_terms = rng.gen_range(1..3);
            (0..num_terms)
                .map(|_| vocabulary.choose(&mut rng).unwrap().clone())
                .collect()
        })
        .collect();

    for &size in &[1000, 5000] {
        group.bench_with_input(BenchmarkId::new("search", size), &size, |b, &size| {
            let tmp_dir = tempdir::TempDir::new("bench_index").unwrap();
            let storage = Arc::new(Storage::from_path(tmp_dir.path().to_str().unwrap()));
            let index = StringIndex::new(Arc::clone(&storage));

            let data = generate_test_data(size);
            let batch: HashMap<_, _> = data.into_iter().collect();
            index
                .insert_multiple(batch)
                .expect("Initial data insertion failed");

            b.iter(|| {
                for query in queries.iter() {
                    let _ = index
                        .search(
                            query.clone(),
                            None,
                            Default::default(),
                            BM25Score::default(),
                            None,
                        )
                        .expect("Search failed");
                }
            });
        });
    }

    group.finish();
}

fn benchmark_concurrent_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(std::time::Duration::from_secs(40));

    for &size in &[5000] {
        group.bench_with_input(BenchmarkId::new("concurrent", size), &size, |b, &size| {
            let tmp_dir = tempdir::TempDir::new("bench_index").unwrap();
            let storage = Arc::new(Storage::from_path(tmp_dir.path().to_str().unwrap()));
            let index = Arc::new(StringIndex::new(Arc::clone(&storage)));

            let data = generate_test_data(size);
            let batch: HashMap<_, _> = data.into_iter().collect();
            index
                .insert_multiple(batch)
                .expect("Initial data insertion failed");

            let queries = generate_test_data(50)
                .into_iter()
                .map(|(_, fields)| {
                    fields
                        .into_iter()
                        .flat_map(|(_, terms)| terms)
                        .map(|(term, _)| term)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            b.iter(|| {
                use std::thread;

                let search_threads: Vec<_> = (0..2)
                    .map(|_| {
                        let index = Arc::clone(&index);
                        let queries = queries.clone();
                        thread::spawn(move || {
                            for query in &queries {
                                let _ = index
                                    .search(
                                        query.clone(),
                                        None,
                                        Default::default(),
                                        BM25Score::default(),
                                        None,
                                    )
                                    .expect("Concurrent search failed");
                            }
                        })
                    })
                    .collect();

                for thread in search_threads {
                    thread.join().expect("Search thread join failed");
                }
            });
        });
    }

    group.finish();
}
criterion_group!(
    benches,
    benchmark_indexing,
    benchmark_batch_indexing,
    benchmark_search,
    benchmark_concurrent_ops
);
criterion_main!(benches);
