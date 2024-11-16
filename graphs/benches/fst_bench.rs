use criterion::{black_box, criterion_group, criterion_main, Criterion};
use graphs::fst::FST;
use string_index::Posting;
use types::{DocumentId, FieldId};

fn setup_test_data() -> Vec<(&'static str, Posting)> {
    vec![
        (
            "apple",
            Posting {
                document_id: DocumentId(1),
                field_id: FieldId(1),
                positions: vec![0],
                term_frequency: 1.0,
                doc_length: 10,
            },
        ),
        (
            "application",
            Posting {
                document_id: DocumentId(2),
                field_id: FieldId(1),
                positions: vec![0, 5],
                term_frequency: 2.0,
                doc_length: 15,
            },
        ),
        (
            "appreciate",
            Posting {
                document_id: DocumentId(3),
                field_id: FieldId(1),
                positions: vec![0],
                term_frequency: 1.0,
                doc_length: 12,
            },
        ),
        (
            "book",
            Posting {
                document_id: DocumentId(4),
                field_id: FieldId(1),
                positions: vec![0],
                term_frequency: 1.0,
                doc_length: 8,
            },
        ),
        (
            "booking",
            Posting {
                document_id: DocumentId(5),
                field_id: FieldId(1),
                positions: vec![0],
                term_frequency: 1.0,
                doc_length: 10,
            },
        ),
    ]
}

fn bench_fst_insertion(c: &mut Criterion) {
    let test_data = setup_test_data();

    c.bench_function("fst_insertion", |b| {
        b.iter(|| {
            let mut fst = FST::new();
            for (word, posting) in test_data.iter() {
                fst.insert(black_box(word), black_box(posting.clone()));
            }
        })
    });
}

fn bench_fst_exact_search(c: &mut Criterion) {
    let test_data = setup_test_data();
    let mut fst = FST::new();

    for (word, posting) in test_data.iter() {
        fst.insert(word, posting.clone());
    }

    c.bench_function("fst_exact_search", |b| {
        b.iter(|| {
            for (word, _) in test_data.iter() {
                fst.search(black_box(word));
            }
        })
    });
}

fn bench_fst_with_different_doc_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("document_lengths");

    for doc_length in [10, 100, 1000, 10000].iter() {
        let posting = Posting {
            document_id: DocumentId(1),
            field_id: FieldId(1),
            positions: vec![0],
            term_frequency: 1.0,
            doc_length: *doc_length as u16,
        };

        group.bench_function(format!("doc_length_{}", doc_length), |b| {
            b.iter(|| {
                let mut fst = FST::new();
                fst.insert("test", black_box(posting.clone()));
                fst.search("test");
            })
        });
    }

    group.finish();
}

fn bench_fst_with_multiple_positions(c: &mut Criterion) {
    let mut group = c.benchmark_group("position_counts");

    for position_count in [1, 10, 100].iter() {
        let posting = Posting {
            document_id: DocumentId(1),
            field_id: FieldId(1),
            positions: (0..*position_count).collect(),
            term_frequency: *position_count as f32,
            doc_length: 1000,
        };

        group.bench_function(format!("positions_{}", position_count), |b| {
            b.iter(|| {
                let mut fst = FST::new();
                fst.insert("test", black_box(posting.clone()));
                fst.search("test");
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fst_insertion,
    bench_fst_exact_search,
    bench_fst_with_different_doc_lengths,
    bench_fst_with_multiple_positions,
);
criterion_main!(benches);
