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

    // Check field stats after search - field should be loaded and have activity
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::CommittedString(string_field_stats) =
        get_field_stats(&stats, index_client.index_id, "text", "committed_string")
            .expect("Field stats should exist after commit")
    else {
        panic!("Expected committed string field stats");
    };
    assert!(
        string_field_stats.loaded,
        "Field should be loaded after commit and search"
    );
    assert!(
        string_field_stats.key_count > 0,
        "Field should contain indexed terms"
    );

    // Wait for automatic field unloading using wait_for pattern
    // This continuously commits and checks until the field is unloaded
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;
    wait_for(&test_context, |test_context| {
        let index_id = index_client.index_id;
        async move {
            // Trigger commit to check for unloading
            test_context.commit_all().await?;

            // Get the collection client and check if field was unloaded
            let collection_client = test_context.get_test_collection_client(
                collection_id,
                write_api_key,
                read_api_key,
            )?;
            let stats = collection_client.reader_stats().await?;
            let IndexFieldStatsType::CommittedString(string_field_stats) =
                get_field_stats(&stats, index_id, "text", "committed_string")
                    .ok_or_else(|| anyhow::anyhow!("Field stats should exist after commit"))?
            else {
                return Err(anyhow::anyhow!("Expected committed string field stats"));
            };

            if string_field_stats.loaded {
                return Err(anyhow::anyhow!("Field still loaded, waiting for unload"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Field should be unloaded after timeout");

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
                "committed_vector" => {
                    matches!(f.stats, IndexFieldStatsType::CommittedVector(_))
                }
                _ => unimplemented!("Unsupported field type: {}. Implement me", t),
            }
        })
        .map(|f| &f.stats)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_offload_vector_field() {
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

    // Perform initial vector search to ensure the field is loaded
    let search_result = wait_for(&collection_client, move |collection_client| {
        async move {
            let search_result = collection_client
                .search(
                    json!({
                        "mode": "vector",
                        // The first document
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

    // Commit the index to create committed fields
    test_context.commit_all().await.unwrap();

    // Perform search immediately after commit to register activity on the committed field
    let search_result = collection_client
        .search(
            json!({
                "mode": "vector",
                // The first document
                "term": "Document about artificial intelligence and machine learning",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(search_result.count, 1);
    assert_eq!(search_result.hits.len(), 1);

    // Check field stats after search - field should be loaded and have activity
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::CommittedVector(vector_field_stats) = get_field_stats(
        &stats,
        index_client.index_id,
        "___orama_auto_embedding",
        "committed_vector",
    )
    .expect("Field stats should exist after commit") else {
        panic!("Expected committed vector field stats");
    };
    assert!(
        vector_field_stats.loaded,
        "Field should be loaded after commit and search"
    );
    assert!(
        vector_field_stats.vector_count > 0,
        "Field should contain vectors"
    );

    // Wait for automatic field unloading using wait_for pattern
    // This continuously commits and checks until the field is unloaded
    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;
    wait_for(&test_context, |test_context| {
        let index_id = index_client.index_id;
        async move {
            // Trigger commit to check for unloading
            test_context.commit_all().await?;

            // Get the collection client and check if field was unloaded
            let collection_client = test_context.get_test_collection_client(
                collection_id,
                write_api_key,
                read_api_key,
            )?;
            let stats = collection_client.reader_stats().await?;
            let IndexFieldStatsType::CommittedVector(vector_field_stats) = get_field_stats(
                &stats,
                index_id,
                "___orama_auto_embedding",
                "committed_vector",
            )
            .ok_or_else(|| anyhow::anyhow!("Field stats should exist after commit"))?
            else {
                return Err(anyhow::anyhow!("Expected committed vector field stats"));
            };

            if vector_field_stats.loaded {
                return Err(anyhow::anyhow!("Field still loaded, waiting for unload"));
            }

            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Field should be unloaded after timeout");

    // Perform search - this should trigger automatic reloading of the field
    let search_result = collection_client
        .search(
            json!({
                "mode": "vector",
                // The first document
                "term": "Document about artificial intelligence and machine learning",
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

    // Check field stats after search - field should be loaded again
    let stats = collection_client.reader_stats().await.unwrap();
    let IndexFieldStatsType::CommittedVector(vector_field_stats) = get_field_stats(
        &stats,
        index_client.index_id,
        "___orama_auto_embedding",
        "committed_vector",
    )
    .expect("Field stats should exist after commit") else {
        panic!("Expected committed vector field stats");
    };
    assert!(
        vector_field_stats.loaded,
        "Field should be loaded again after search"
    );

    // Verify field still works correctly with different search
    let search_result = collection_client
        .search(
            json!({
                "mode": "vector",
                // The first document
                "term": "Research paper on neural networks and deep learning",
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

    drop(test_context);
}
