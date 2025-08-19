use serde_json::json;
use std::time::Instant;

use crate::tests::utils::{init_log, TestContext};

/// Quick validation test for the fulltext benchmark setup
#[tokio::test(flavor = "multi_thread")]
async fn test_fulltext_benchmark_validation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert a small set of test documents to validate the benchmark approach
    let documents = (0..1000)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "title": format!("Document {}", i),
                "content": format!("technology innovation development software engineering article {}", i),
                "category": match i % 5 {
                    0 => "technology",
                    1 => "business", 
                    2 => "science",
                    3 => "engineering",
                    _ => "general"
                },
            })
        })
        .collect::<Vec<_>>();

    index_client
        .insert_documents(json!(documents).try_into().unwrap())
        .await
        .unwrap();

    // Test various query types to ensure they work for benchmarking
    let test_queries = vec![
        ("single_word", json!({"term": "technology"})),
        (
            "multi_word",
            json!({"term": "artificial intelligence machine learning"}),
        ),
        ("exact_match", json!({"term": "Document", "exact": true})),
        (
            "with_threshold",
            json!({"term": "software engineering development", "threshold": 0.8}),
        ),
        (
            "pagination",
            json!({"term": "technology", "limit": 10, "offset": 5}),
        ),
    ];

    println!("Uncommitted index performance:");
    for (name, query) in &test_queries {
        let start = Instant::now();
        let result = collection_client
            .search(query.clone().try_into().unwrap())
            .await
            .unwrap();
        let duration = start.elapsed();

        println!("  {}: {:?} ({} results)", name, duration, result.count);
        // Some queries might not return results, which is fine for benchmarking
        println!("    Query {} returned {} results", name, result.count);
    }

    // Test committed index performance
    test_context.commit_all().await.unwrap();

    println!("Committed index performance:");
    for (name, query) in &test_queries {
        let start = Instant::now();
        let result = collection_client
            .search(query.clone().try_into().unwrap())
            .await
            .unwrap();
        let duration = start.elapsed();

        println!("  {}: {:?} ({} results)", name, duration, result.count);
        // Some queries might not return results, which is fine for benchmarking
        println!(
            "    Query {} returned {} results after commit",
            name, result.count
        );
    }

    drop(test_context);
}
