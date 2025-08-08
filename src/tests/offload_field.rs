use std::time::Duration;

use duration_string::DurationString;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    collection_manager::sides::read::{CollectionStats, IndexFieldStatsType, OffloadFieldConfig},
    tests::utils::{create_oramacore_config, init_log, TestContext},
    types::IndexId,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_offload_string_field() {
    init_log();

    // Create config with short unload window for testing
    let mut config = create_oramacore_config();
    config.reader_side.config.offload_field = OffloadFieldConfig {
        unload_window: DurationString::from_string("2s".to_string()).unwrap(),
        slot_count_exp: 8, // 2^8 groups
        slot_size_exp: 1,  // group by 2 second
    };

    let test_context = TestContext::new_with_config(config).await;

    // Create collection and index
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert test documents
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "Lorem ipsum dolor sit amet consectetur adipiscing elit",
                },
                {
                    "id": "2",
                    "text": "Curabitur sem tortor interdum in rutrum dignissim vestibulum metus",
                },
                {
                    "id": "3",
                    "text": "Pellentesque habitant morbi tristique senectus et netus",
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Perform initial search to ensure the field is loaded
    let search_result = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);
    assert_eq!(
        search_result.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    // Commit the index to create committed fields
    test_context.commit_all().await.unwrap();

    // Check field stats after commit - field should be loaded
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::CommittedString(string_field_stats) =
        get_field_stats(&stats, index_client.index_id, "text", "committed_string")
            .expect("Field stats should exist after commit")
    else {
        panic!("Expected committed string field stats");
    };
    assert!(
        string_field_stats.loaded,
        "Field should be loaded after commit"
    );
    assert!(
        string_field_stats.key_count > 0,
        "Field should contain indexed terms"
    );

    // Perform search to verify committed field works
    let search_result = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);

    // Wait for automatic field unloading due to the short unload window
    // The unload happens during commits, so trigger another commit after waiting
    sleep(Duration::from_millis(5_000)).await;

    // Trigger commit to check for unloading
    test_context.commit_all().await.unwrap();

    // Verify field was unloaded
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::CommittedString(string_field_stats) =
        get_field_stats(&stats, index_client.index_id, "text", "committed_string")
            .expect("Field stats should exist after commit")
    else {
        panic!("Expected committed string field stats");
    };
    assert!(
        !string_field_stats.loaded,
        "Field should be unloaded after timeout"
    );

    // Perform search - this should trigger automatic reloading of the field
    let search_result = collection_client
        .search(
            json!({
                "term": "Curabitur sem",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);
    assert_eq!(
        search_result.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    // Check field stats after search - field should be loaded again
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::CommittedString(string_field_stats) =
        get_field_stats(&stats, index_client.index_id, "text", "committed_string")
            .expect("Field stats should exist after commit")
    else {
        panic!("Expected committed string field stats");
    };
    assert!(
        string_field_stats.loaded,
        "Field should be loaded again after search"
    );

    // Verify field still works correctly with different search
    let search_result = collection_client
        .search(
            json!({
                "term": "Pellentesque habitant",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);
    assert_eq!(
        search_result.hits[0].id,
        format!("{}:{}", index_client.index_id, "3")
    );

    drop(test_context);
}

fn get_field_stats<'s>(
    collection_stats: &'s CollectionStats,
    index_id: IndexId,
    field_name: &str,
    t: &'static str,
) -> Option<&'s IndexFieldStatsType> {
    let index_stats = &collection_stats
        .indexes_stats
        .iter()
        .find(|i| i.id == index_id)
        .unwrap();
    index_stats
        .fields_stats
        .iter()
        .find(|f| {
            if f.field_path != field_name {
                return false;
            }

            match t {
                "committed_string" => {
                    matches!(f.stats, IndexFieldStatsType::CommittedString(_))
                }
                _ => unimplemented!("Unsupported field type: {}. Implement me", t),
            }
        })
        .map(|f| &f.stats)
}
