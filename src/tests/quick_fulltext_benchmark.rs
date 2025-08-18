use serde_json::json;
use std::fs;
use std::time::Instant;

use crate::tests::utils::{init_log, TestContext};

/// Quick fulltext search benchmark with reduced scope
#[tokio::test(flavor = "multi_thread")]
async fn test_quick_fulltext_benchmark() {
    init_log();
    println!("Quick Fulltext Search Benchmark");

    // Load games.json data
    let games_path = "benches/games.json";
    let games_data = fs::read_to_string(games_path).expect("Failed to read games.json");
    let games: Vec<serde_json::Value> =
        serde_json::from_str(&games_data).expect("Failed to parse games.json");

    println!("Loaded {} games from games.json", games.len());

    // Use different subsets for different benchmark scales
    let test_scales = vec![
        (500, "Small scale"),
        (1000, "Medium scale"),
        (games.len(), "Full dataset"),
    ];

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

        // Create realistic game-based queries
        let queries = vec![
            ("RPG", "Genre search"),
            ("adventure", "Genre lowercase"),
            ("fantasy action", "Multi-genre"),
            ("Zelda", "Game title"),
            ("combat exploration", "Gameplay features"),
            ("indie platformer", "Indie genre combo"),
            ("dungeon crawler", "Specific type"),
            ("open world", "Popular feature"),
        ];

        println!("  Uncommitted Index Search Performance:");
        for (query_term, description) in &queries {
            let mut total_time = std::time::Duration::ZERO;
            let iterations = 5; // Reduced for multiple scales

            for _ in 0..iterations {
                let start = Instant::now();
                let result = collection_client
                    .search(json!({"term": query_term}).try_into().unwrap())
                    .await
                    .unwrap();
                total_time += start.elapsed();

                // Track first result for validation
                if iterations == 5 {
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
                "rating_excellent",
                json!({
                    "term": "",
                    "where": {
                        "rating": {"gte": 4.5}
                    }
                }),
                "Rating ≥ 4.5",
            ),
            (
                "rating_range",
                json!({
                    "term": "",
                    "where": {
                        "rating": {"between": [3.5, 4.5]}
                    }
                }),
                "Rating 3.5-4.5",
            ),
            (
                "rating_low",
                json!({
                    "term": "",
                    "where": {
                        "rating": {"lt": 3.0}
                    }
                }),
                "Rating < 3.0",
            ),
            (
                "rating_poor",
                json!({
                    "term": "",
                    "where": {
                        "rating": {"lte": 2.5}
                    }
                }),
                "Rating ≤ 2.5",
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
            (
                "combined_range",
                json!({
                    "term": "adventure",
                    "where": {
                        "rating": {"between": [4.0, 5.0]}
                    }
                }),
                "Adventure + High Rating",
            ),
            (
                "exact_rating",
                json!({
                    "term": "",
                    "where": {
                        "rating": {"eq": 4.5}
                    }
                }),
                "Exact Rating = 4.5",
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
