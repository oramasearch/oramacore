use serde_json::json;
use std::fs;
use std::time::Instant;

use crate::tests::utils::{init_log, TestContext};

/// Quick fulltext search benchmark with reduced scope
#[tokio::test(flavor = "multi_thread")]
async fn test_quick_fulltext_benchmark() {
    init_log();
    println!("Quick Fulltext Search Benchmark");

    // Load games.json data - handle different working directories
    let games_path = if std::path::Path::new("benches/games.json").exists() {
        "benches/games.json".to_string()
    } else if std::path::Path::new("../benches/games.json").exists() {
        "../benches/games.json".to_string()
    } else {
        // Try to find games.json from project root
        let current_dir = std::env::current_dir().expect("Failed to get current directory");
        let mut path = current_dir;
        loop {
            let games_file = path.join("benches/games.json");
            if games_file.exists() {
                break games_file.to_str().expect("Invalid path").to_string();
            }
            if !path.pop() {
                panic!("Could not find games.json file in project tree");
            }
        }
    };
    let games_data = fs::read_to_string(&games_path)
        .unwrap_or_else(|e| panic!("Failed to read games.json from {}: {}", games_path, e));
    let games: Vec<serde_json::Value> =
        serde_json::from_str(&games_data).expect("Failed to parse games.json");

    println!("Loaded {} games from games.json", games.len());

    // Use smaller subsets for quicker test execution
    let test_scales = vec![(100, "Quick test"), (250, "Medium test")];

    for (doc_count, scale_name) in test_scales {
        let doc_count = doc_count.min(games.len());
        let documents = games.iter().take(doc_count).cloned().collect::<Vec<_>>();

        println!("\n=== {} ({} games) ===", scale_name, doc_count);

        // Create fresh context for each scale
        let test_context = TestContext::new().await;
        let collection_client = test_context.create_collection().await.unwrap();
        let index_client = collection_client.create_index().await.unwrap();

        let insert_start = Instant::now();
        index_client
            .insert_documents(json!(documents).try_into().unwrap())
            .await
            .unwrap();
        println!("  Insert time: {:?}", insert_start.elapsed());

        // Create realistic game-based queries (reduced for speed)
        let queries = vec![
            ("RPG", "Genre search"),
            ("adventure", "Genre lowercase"),
            ("fantasy action", "Multi-genre"),
            ("Zelda", "Game title"),
        ];

        println!("  Uncommitted Index Search Performance:");
        for (query_term, description) in &queries {
            let mut total_time = std::time::Duration::ZERO;
            let iterations = 3; // Reduced for quicker test

            for _ in 0..iterations {
                let start = Instant::now();
                let result = collection_client
                    .search(json!({"term": query_term}).try_into().unwrap())
                    .await
                    .unwrap();
                total_time += start.elapsed();

                // Track first result for validation
                if iterations == 3 {
                    println!(
                        "    {}: {:?} ({} results)",
                        description,
                        start.elapsed(),
                        result.count
                    );
                }
            }

            let avg_time = total_time / iterations;
            println!("    {} average: {:?}", description, avg_time);
        }

        // Test committed index performance for this scale
        println!("  Committing {} documents...", doc_count);
        let commit_start = Instant::now();
        test_context.commit_all().await.unwrap();
        println!("  Commit time: {:?}", commit_start.elapsed());

        println!("  Committed Index Search Performance:");
        for (query_term, description) in &queries {
            let start = Instant::now();
            let result = collection_client
                .search(json!({"term": query_term}).try_into().unwrap())
                .await
                .unwrap();
            let duration = start.elapsed();
            println!(
                "    {}: {:?} ({} results)",
                description, duration, result.count
            );
        }

        // Filter operations benchmark
        println!("  Filter Operations Performance:");

        let filter_queries = vec![
            (
                "rating_high",
                json!({
                    "term": "",
                    "where": {
                        "rating": {"gt": 4.0}
                    }
                }),
                "Rating > 4.0",
            ),
            (
                "combined_filter",
                json!({
                    "term": "fantasy",
                    "where": {
                        "rating": {"gte": 4.0}
                    }
                }),
                "Fantasy + Rating ≥ 4.0",
            ),
        ];

        for (_query_name, query, description) in &filter_queries {
            let start = Instant::now();
            let result = collection_client
                .search(query.clone().try_into().unwrap())
                .await;

            match result {
                Ok(result) => {
                    let duration = start.elapsed();
                    println!(
                        "    {}: {:?} ({} results)",
                        description, duration, result.count
                    );
                }
                Err(e) => {
                    println!("    {}: Error - {:?}", description, e);
                }
            }
        }

        // Clean up for next scale test
        drop(test_context);
    }
}
