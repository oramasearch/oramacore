use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

/// Tests that geopoint fields in a temp index are correctly promoted to the runtime index.
///
/// The promotion flow:
/// 1. Create a runtime index with some documents (including a geopoint field).
/// 2. Create a temp index from the runtime index.
/// 3. Insert documents with geopoint fields into the temp index.
/// 4. Commit the temp index to persist GeoPointStorage to temp directory.
/// 5. Replace (promote) the temp index into the runtime index.
/// 6. Verify the geopoint field is searchable after promotion via radius filter.
/// 7. Commit again and verify data survives.
/// 8. Reload the system and verify geopoint field data persists after reload.
#[tokio::test(flavor = "multi_thread")]
async fn test_geopoint_field_promotion_from_temp_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    // Insert an initial document with a geopoint field in the runtime index.
    // This uses coordinates near Rome, Italy.
    runtime_index_client
        .insert_documents(
            json!([
                {
                    "id": "original-1",
                    "name": "Original",
                    "location": { "lat": 41.9028, "lon": 12.4964 },
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Create a temp index based on the runtime index.
    let temp_index_client = collection_client
        .create_temp_index(runtime_index_client.index_id)
        .await
        .unwrap();

    // Insert documents with geopoint fields into the temp index.
    // Cities spread across Europe for spatial filtering tests.
    let docs: Vec<_> = vec![
        ("0", "Rome", 41.9028, 12.4964),
        ("1", "Milan", 45.4642, 9.1900),
        ("2", "Naples", 40.8518, 14.2681),
        ("3", "Florence", 43.7696, 11.2558),
        ("4", "Venice", 45.4408, 12.3155),
        ("5", "Turin", 45.0703, 7.6869),
        ("6", "Bologna", 44.4949, 11.3426),
        ("7", "Palermo", 38.1157, 13.3615),
        ("8", "Genoa", 44.4056, 8.9463),
        ("9", "Bari", 41.1171, 16.8719),
        ("10", "Paris", 48.8566, 2.3522),
        ("11", "Berlin", 52.5200, 13.4050),
        ("12", "Madrid", 40.4168, -3.7038),
        ("13", "London", 51.5074, -0.1278),
        ("14", "Amsterdam", 52.3676, 4.9041),
        ("15", "Vienna", 48.2082, 16.3738),
        ("16", "Prague", 50.0755, 14.4378),
        ("17", "Warsaw", 52.2297, 21.0122),
        ("18", "Brussels", 50.8503, 4.3517),
        ("19", "Lisbon", 38.7223, -9.1393),
    ]
    .into_iter()
    .map(|(id, name, lat, lon)| {
        json!({
            "id": id,
            "name": name,
            "location": { "lat": lat, "lon": lon },
        })
    })
    .collect();

    temp_index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Commit to persist the geopoint field data in the temp directory.
    test_context.commit_all().await.unwrap();

    // Verify the original runtime index still returns its own data through search.
    // Search with a radius around Rome (500km) - should find just the original document.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 500,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 1,
        "Runtime index should still have only its own document"
    );

    // Promote: replace the runtime index with the temp index.
    collection_client
        .replace_index(runtime_index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    // After promotion, search with a 500km radius around Rome.
    // Should find Italian cities within range: Rome(0), Naples(2), Florence(3), Bologna(6), Bari(9)
    // and possibly others depending on exact distances.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 500,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // All 10 Italian cities (0-9) should be within 500km of Rome.
    // Non-Italian cities are much farther away.
    assert!(
        result.count >= 8,
        "After promotion, should find at least 8 Italian cities within 500km of Rome, got {}",
        result.count
    );
    let italian_count = result.count;

    // Search with a larger radius (2000km) around Rome to get all European cities.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 2000,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 15,
        "After promotion, 2000km radius from Rome should find most European cities, got {}",
        result.count
    );

    // Commit after promotion to verify compaction works at the new path.
    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 500,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, italian_count,
        "After post-promotion commit, geopoint filter should still work"
    );

    // Reload the entire system and verify data persists.
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 500,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, italian_count,
        "After reload, geopoint field data should persist"
    );

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 2000,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        result.count >= 15,
        "After reload, 2000km radius filter should persist, got {}",
        result.count
    );

    drop(test_context);
}

/// Tests that inserting new documents with geopoint fields into a promoted index works correctly.
///
/// After promotion, the geopoint field storage points to the new runtime path.
/// New inserts and subsequent commits must operate on the promoted path without errors.
#[tokio::test(flavor = "multi_thread")]
async fn test_geopoint_field_insert_after_promotion() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let runtime_index_client = collection_client.create_index().await.unwrap();

    runtime_index_client
        .insert_documents(
            json!([{
                "id": "seed",
                "name": "seed",
                "location": { "lat": 41.9028, "lon": 12.4964 },
            }])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let temp_index_client = collection_client
        .create_temp_index(runtime_index_client.index_id)
        .await
        .unwrap();

    temp_index_client
        .insert_documents(
            json!([
                { "id": "1", "name": "Rome", "location": { "lat": 41.9028, "lon": 12.4964 } },
                { "id": "2", "name": "Paris", "location": { "lat": 48.8566, "lon": 2.3522 } },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Promote the temp index.
    collection_client
        .replace_index(runtime_index_client.index_id, temp_index_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    // Insert new documents into the now-promoted runtime index.
    runtime_index_client
        .insert_documents(
            json!([
                { "id": "3", "name": "Milan", "location": { "lat": 45.4642, "lon": 9.1900 } },
                { "id": "4", "name": "Berlin", "location": { "lat": 52.5200, "lon": 13.4050 } },
                { "id": "5", "name": "Naples", "location": { "lat": 40.8518, "lon": 14.2681 } },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Wait for eventual consistency after inserting into the promoted index.
    wait_for(&collection_client, |c| {
        async move {
            let result = c
                .search(json!({ "term": "" }).try_into().unwrap())
                .await
                .unwrap();
            if result.count != 5 {
                return Err(anyhow::anyhow!(
                    "Expected 5 documents, got {}",
                    result.count
                ));
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

    // Search with 500km radius around Rome - should find Italian cities.
    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 500,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // Rome(1), Milan(3), Naples(5) should be within 500km. Paris(2) and Berlin(4) are farther.
    assert!(
        result.count >= 3,
        "Should find at least 3 Italian cities within 500km of Rome after post-promotion insert, got {}",
        result.count
    );

    // Commit to persist the post-promotion data.
    test_context.commit_all().await.unwrap();

    // Reload and verify everything is correct.
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "location": {
                        "radius": {
                            "coordinates": { "lat": 41.9028, "lon": 12.4964 },
                            "unit": "km",
                            "value": 2000,
                            "inside": true,
                        }
                    }
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        result.count, 5,
        "After reload, all 5 documents should be within 2000km of Rome"
    );

    drop(test_context);
}
