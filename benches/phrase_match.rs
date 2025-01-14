use criterion::{criterion_group, criterion_main};

#[cfg(feature = "benchmarking")]
mod bench {

    use criterion::{black_box, Criterion};
    use rustorama::indexes::string::BM25Scorer;
    use rustorama::test_utils::create_committed_string_field_index;
    use rustorama::types::Document;
    use serde_json::json;

    #[derive(Debug)]
    #[allow(dead_code)]
    struct BenchmarkResults {
        docs_count: usize,
        index_size_bytes: u64,
        indexing_time_ms: u64,
        avg_search_time_ms: f64,
        memory_usage_mb: f64,
        throughput_docs_per_sec: f64,
    }

    fn generate_test_data() -> Vec<Document> {
        let path = "src/bin/imdb_top_1000_tv_series.json";
        let docs: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        docs.as_array()
            .unwrap()
            .iter()
            .map(|d| {
                json!({
                    "field": d["plot"].as_str().unwrap(),
                })
                .try_into()
                .unwrap()
            })
            .collect()
    }

    pub fn one_word(c: &mut Criterion) {
        let data = generate_test_data();

        let mut inputs = None;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();

        runtime.block_on(async {
            inputs = create_committed_string_field_index(data).await.unwrap();
        });

        c.bench_function("one word - phrase", |b| {
            b.iter(|| {
                let string_index = inputs.as_ref().unwrap();
                string_index
                    .search_with_phrase_match(
                        black_box(&["control".to_string()]),
                        1.0,
                        &mut BM25Scorer::new(),
                        None,
                        &string_index.get_global_info(),
                    )
                    .unwrap();
            });
        });
        c.bench_function("one word - no phrase", |b| {
            b.iter(|| {
                let string_index = inputs.as_ref().unwrap();
                string_index
                    .search_without_phrase_match(
                        black_box(&["control".to_string()]),
                        1.0,
                        &mut BM25Scorer::new(),
                        None,
                        &string_index.get_global_info(),
                    )
                    .unwrap();
            });
        });

        c.bench_function("two word - phrase", |b| {
            b.iter(|| {
                let string_index = inputs.as_ref().unwrap();
                string_index
                    .search_with_phrase_match(
                        black_box(&["control".to_string(), "people".to_string()]),
                        1.0,
                        &mut BM25Scorer::new(),
                        None,
                        &string_index.get_global_info(),
                    )
                    .unwrap();
            });
        });
        c.bench_function("two word - no phrase", |b| {
            b.iter(|| {
                let string_index = inputs.as_ref().unwrap();
                string_index
                    .search_without_phrase_match(
                        black_box(&["control".to_string(), "people".to_string()]),
                        1.0,
                        &mut BM25Scorer::new(),
                        None,
                        &string_index.get_global_info(),
                    )
                    .unwrap();
            });
        });
    }
}

#[cfg(not(feature = "benchmarking"))]
mod bench {
    use criterion::Criterion;

    pub fn one_word(c: &mut Criterion) {
        panic!("re-run with `cargo bench --features benchmarking`");
    }
}

criterion_group!(benches, bench::one_word);
criterion_main!(benches);
