use serde_json::json;
use std::fs;
use std::time::Instant;

use crate::tests::utils::{init_log, TestContext};

/// Dedicated filter operations benchmark using games.json data
#[tokio::test(flavor = "multi_thread")]
async fn test_filter_operations_benchmark() {
    init_log();
    println!("Filter Operations Benchmark with Games Data");

    // Load games.json data
    let games_path = "benches/games.json";
    let games_data = fs::read_to_string(games_path).expect("Failed to read games.json");
    let games: Vec<serde_json::Value> =
        serde_json::from_str(&games_data).expect("Failed to parse games.json");

    println!("Loaded {} games from games.json", games.len());

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert full dataset for comprehensive filter testing
    let doc_count = 1000; // Use subset for faster testing
    let documents = games.iter().take(doc_count).cloned().collect::<Vec<_>>();

    println!("Inserting {doc_count} games for filter testing...");
    let insert_start = Instant::now();
    index_client
        .insert_documents(json!(documents).try_into().unwrap())
        .await
        .unwrap();
    println!("Insert time: {:?}\n", insert_start.elapsed());

    // Comprehensive filter operation tests
    let filter_tests = vec![
        // Rating filters
        (
            "rating_gt_4",
            json!({
                "term": "",
                "where": {"rating": {"gt": 4.0}}
            }),
            "Rating > 4.0",
        ),
        (
            "rating_gte_45",
            json!({
                "term": "",
                "where": {"rating": {"gte": 4.5}}
            }),
            "Rating ≥ 4.5",
        ),
        (
            "rating_lt_3",
            json!({
                "term": "",
                "where": {"rating": {"lt": 3.0}}
            }),
            "Rating < 3.0",
        ),
        (
            "rating_lte_25",
            json!({
                "term": "",
                "where": {"rating": {"lte": 2.5}}
            }),
            "Rating ≤ 2.5",
        ),
        (
            "rating_eq_45",
            json!({
                "term": "",
                "where": {"rating": {"eq": 4.5}}
            }),
            "Rating = 4.5",
        ),
        (
            "rating_between",
            json!({
                "term": "",
                "where": {"rating": {"between": [3.5, 4.5]}}
            }),
            "Rating 3.5-4.5",
        ),
        // Combined fulltext + filter
        (
            "fantasy_high_rating",
            json!({
                "term": "fantasy",
                "where": {"rating": {"gte": 4.0}}
            }),
            "Fantasy + Rating ≥ 4.0",
        ),
        (
            "action_mid_rating",
            json!({
                "term": "action",
                "where": {"rating": {"between": [3.0, 4.0]}}
            }),
            "Action + Mid Rating",
        ),
        (
            "rpg_excellent",
            json!({
                "term": "RPG",
                "where": {"rating": {"gte": 4.5}}
            }),
            "RPG + Excellent Rating",
        ),
        // Edge cases
        (
            "no_term_high_rating",
            json!({
                "term": "",
                "where": {"rating": {"gt": 4.8}}
            }),
            "Very High Rating",
        ),
        (
            "adventure_exact_rating",
            json!({
                "term": "adventure",
                "where": {"rating": {"eq": 4.4}}
            }),
            "Adventure + Exact Rating",
        ),
    ];

    // Test uncommitted index filters
    println!("Uncommitted Index Filter Performance:");
    for (_name, query, description) in &filter_tests {
        let start = Instant::now();
        let result = collection_client
            .search(query.clone().try_into().unwrap())
            .await;

        match result {
            Ok(result) => {
                let duration = start.elapsed();
                println!(
                    "  {}: {:?} ({} results)",
                    description, duration, result.count
                );
            }
            Err(e) => {
                println!("  {description}: Error - {e:?}");
            }
        }
    }

    // Commit and test committed index filters
    println!("\nCommitting data...");
    let commit_start = Instant::now();
    test_context.commit_all().await.unwrap();
    println!("Commit time: {:?}", commit_start.elapsed());

    println!("\nCommitted Index Filter Performance:");
    for (_name, query, description) in &filter_tests {
        let start = Instant::now();
        let result = collection_client
            .search(query.clone().try_into().unwrap())
            .await;

        match result {
            Ok(result) => {
                let duration = start.elapsed();
                println!(
                    "  {}: {:?} ({} results)",
                    description, duration, result.count
                );
            }
            Err(e) => {
                println!("  {description}: Error - {e:?}");
            }
        }
    }

    // Performance comparison analysis
    println!("\nFilter Performance Analysis:");
    println!("- Numeric filters (rating) should be fastest");
    println!("- Range queries (between) may be slower than simple comparisons");
    println!("- Combined fulltext + filter should show additive performance impact");
    println!("- Committed index filters may be slower but more consistent");
    println!("- Filter selectivity affects performance (fewer results = faster)");

    drop(test_context);
}
