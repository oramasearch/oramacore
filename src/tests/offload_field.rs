use anyhow;
use duration_string::DurationString;
use futures::FutureExt;
use serde_json::json;

use crate::{
    collection_manager::sides::read::{CollectionStats, IndexFieldStatsType, OffloadFieldConfig},
    tests::utils::{create_oramacore_config, init_log, wait_for, TestContext},
    types::IndexId,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_offload_string_field() {
    init_log();

    // Create config with short unload window for testing
    let mut config = create_oramacore_config();
    config.reader_side.config.offload_field = OffloadFieldConfig {
        unload_window: DurationString::from_string("3s".to_string()).unwrap(),
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

    // Perform search immediately after commit to register activity on the committed field
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

    // Check field stats after search - field should exist and have activity
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::StringFieldStorage(string_field_stats) =
        get_field_stats(&stats, index_client.index_id, "text", "string")
            .expect("Field stats should exist after commit")
    else {
        panic!("Expected string field stats");
    };
    assert!(
        string_field_stats.unique_terms_count > 0,
        "Field should contain indexed terms"
    );

    // StringStorage uses mmap-based segments and does not need manual unloading.
    // Simply verify that search still works after commit.
    test_context.commit_all().await.unwrap();

    // Perform search - should still work after commit
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

    println!("Search after commit successful\n\n\n\n\n\n {search_result:#?}");

    // Check field stats still exist
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::StringFieldStorage(string_field_stats) =
        get_field_stats(&stats, index_client.index_id, "text", "string")
            .expect("Field stats should exist after commit")
    else {
        panic!("Expected string field stats");
    };
    println!("Field stats after commit: stats={string_field_stats:#?}",);
    assert!(
        string_field_stats.total_documents > 0,
        "Field should still have documents"
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
                "string" => {
                    matches!(f.stats, IndexFieldStatsType::StringFieldStorage(_))
                }
                "embedding" => {
                    matches!(f.stats, IndexFieldStatsType::EmbeddingFieldStorage(_))
                }
                _ => unimplemented!("Unsupported field type: {}. Implement me", t),
            }
        })
        .map(|f| &f.stats)
}

/// Test that embedding fields work correctly across commits.
/// EmbeddingFieldStorage uses mmap-based segments so memory is managed by the OS,
/// no manual load/unload cycle is needed.
#[tokio::test(flavor = "multi_thread")]
async fn test_embedding_field_across_commits() {
    init_log();

    let test_context = TestContext::new().await;

    // Create collection and index
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert test documents with embeddings
    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "Document about artificial intelligence and machine learning",
                },
                {
                    "id": "2",
                    "text": "Research paper on neural networks and deep learning",
                },
                {
                    "id": "3",
                    "text": "Study on computer vision and image processing",
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // Perform initial vector search before commit
    let search_result = wait_for(&collection_client, move |collection_client| {
        async move {
            let search_result = collection_client
                .search(
                    json!({
                        "mode": "vector",
                        "term": "Document about artificial intelligence and machine learning",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();

            if search_result.count == 0 {
                return Err(anyhow::anyhow!(
                    "No results found, waiting for index to load"
                ));
            }

            Ok(search_result)
        }
        .boxed()
    })
    .await
    .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);
    assert_eq!(
        search_result.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    // Commit the index to persist embedding field data
    test_context.commit_all().await.unwrap();

    // Verify search works after commit
    let search_result = collection_client
        .search(
            json!({
                "mode": "vector",
                "term": "Document about artificial intelligence and machine learning",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);

    // Check field stats after commit - embedding field should report embeddings
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::EmbeddingFieldStorage(embedding_stats) = get_field_stats(
        &stats,
        index_client.index_id,
        "___orama_auto_embedding",
        "embedding",
    )
    .expect("Field stats should exist after commit") else {
        panic!("Expected embedding field stats");
    };
    assert!(
        embedding_stats.vector_count > 0,
        "Field should contain embeddings"
    );

    // Verify search with a different query still works
    let search_result = wait_for(&collection_client, |collection_client| {
        async move {
            let search_result = collection_client
                .search(
                    json!({
                        "mode": "vector",
                        "term": "Research paper on neural networks and deep learning",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();

            if search_result.count > 0 {
                Ok(search_result)
            } else {
                Err(anyhow::anyhow!("result not found"))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    assert_eq!(search_result.hits.len(), 2);
    assert_eq!(
        search_result.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        search_result.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}
