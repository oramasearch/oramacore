use crate::tests::utils::{init_log, wait_for, TestContext};
use anyhow::bail;
use futures::FutureExt;

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_set_and_get() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    collection
        .set_value("theme".to_string(), "dark".to_string())
        .await
        .unwrap();

    let value = collection.get_value("theme").await;
    assert_eq!(value, Some("dark".to_string()));
    drop(collection);

    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key.clone();
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, &read_api_key).await?;
            let value = collection.get_value("theme").await;
            if value.is_none() {
                bail!("Collection value not yet available in reader");
            }
            assert_eq!(value, Some("dark".to_string()));
            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Collection value should be available in reader");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_delete() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    collection
        .set_value("key1".to_string(), "value1".to_string())
        .await
        .unwrap();
    collection
        .set_value("key2".to_string(), "value2".to_string())
        .await
        .unwrap();

    let removed = collection.delete_value("key1").await.unwrap();
    assert!(removed, "key1 should have been removed");

    let removed = collection.delete_value("nonexistent").await.unwrap();
    assert!(!removed, "nonexistent key should return false");

    assert_eq!(collection.get_value("key1").await, None);
    assert_eq!(
        collection.get_value("key2").await,
        Some("value2".to_string())
    );
    drop(collection);

    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key.clone();
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, &read_api_key).await?;
            let values = collection.list_values().await;
            if values.len() != 1 {
                bail!("Expected 1 value in reader, got {}", values.len());
            }
            if values.get("key2") != Some(&"value2".to_string()) {
                bail!(
                    "Expected key2=value2 in reader, got {:?}",
                    values.get("key2")
                );
            }
            if values.contains_key("key1") {
                bail!("key1 should have been deleted from reader");
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Collection value deletion should be reflected in reader");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_list() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    let values = collection.list_values().await;
    assert!(values.is_empty());

    collection
        .set_value("color".to_string(), "blue".to_string())
        .await
        .unwrap();
    collection
        .set_value("size".to_string(), "large".to_string())
        .await
        .unwrap();
    collection
        .set_value("mode".to_string(), "production".to_string())
        .await
        .unwrap();

    let values = collection.list_values().await;
    assert_eq!(values.len(), 3);
    assert_eq!(values.get("color"), Some(&"blue".to_string()));
    assert_eq!(values.get("size"), Some(&"large".to_string()));
    assert_eq!(values.get("mode"), Some(&"production".to_string()));
    drop(collection);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_overwrite() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    collection
        .set_value("key".to_string(), "original".to_string())
        .await
        .unwrap();
    assert_eq!(
        collection.get_value("key").await,
        Some("original".to_string())
    );

    collection
        .set_value("key".to_string(), "updated".to_string())
        .await
        .unwrap();
    assert_eq!(
        collection.get_value("key").await,
        Some("updated".to_string())
    );

    assert_eq!(collection.values_count().await, 1);
    drop(collection);

    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key.clone();
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, &read_api_key).await?;
            let value = collection.get_value("key").await;
            if value.as_deref() != Some("updated") {
                bail!("Expected 'updated' in reader, got {value:?}");
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Updated collection value should be reflected in reader");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_commit_and_reload() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    collection
        .set_value("persist_key".to_string(), "persist_value".to_string())
        .await
        .unwrap();
    drop(collection);

    test_context.commit_all().await.unwrap();

    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key.clone();
        let collection_id = c.collection_id;
        async move {
            let collection = reader.get_collection(collection_id, &read_api_key).await?;
            let value = collection.get_value("persist_key").await;
            if value.is_none() {
                bail!("Value not yet in reader");
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Value should reach reader");

    let writer_data_dir = &test_context.config.writer_side.config.data_dir;
    let writer_values_file = writer_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("values.json");
    assert!(
        writer_values_file.exists(),
        "Values file should exist in writer: {writer_values_file:?}"
    );

    let reader_data_dir = &test_context.config.reader_side.config.data_dir;
    let reader_values_file = reader_data_dir
        .join("collections")
        .join(collection_client.collection_id.to_string())
        .join("values.json");
    assert!(
        reader_values_file.exists(),
        "Values file should exist in reader: {reader_values_file:?}"
    );

    let collection_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key.clone();

    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(collection_id, write_api_key, read_api_key)
        .unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();
    let value = collection.get_value("persist_key").await;
    assert_eq!(
        value,
        Some("persist_value".to_string()),
        "Value should persist after reload in writer"
    );
    drop(collection);

    let collection = collection_client
        .reader
        .get_collection(
            collection_client.collection_id,
            &collection_client.read_api_key,
        )
        .await
        .unwrap();
    let value = collection.get_value("persist_key").await;
    assert_eq!(
        value,
        Some("persist_value".to_string()),
        "Value should persist after reload in reader"
    );
    drop(collection);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_validation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    let result = collection
        .set_value("".to_string(), "value".to_string())
        .await;
    assert!(result.is_err(), "Empty key should be rejected");

    let result = collection
        .set_value("key with spaces".to_string(), "value".to_string())
        .await;
    assert!(result.is_err(), "Key with spaces should be rejected");

    let result = collection
        .set_value("key".to_string(), "".to_string())
        .await;
    assert!(result.is_err(), "Empty value should be rejected");

    let large_value = "x".repeat(10 * 1024 + 1);
    let result = collection.set_value("key".to_string(), large_value).await;
    assert!(result.is_err(), "Value exceeding 10KB should be rejected");

    let result = collection
        .set_value("valid_key-1".to_string(), "valid value".to_string())
        .await;
    assert!(result.is_ok(), "Valid key-value should succeed");

    drop(collection);
    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_values_stats() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let _index_client = collection_client.create_index().await.unwrap();

    // Check initial stats
    let stats = collection_client
        .reader
        .collection_stats(
            &collection_client.read_api_key,
            collection_client.collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await
        .unwrap();
    assert_eq!(stats.values_count, 0);

    // Add some values
    let collection = collection_client
        .writer
        .get_collection(
            collection_client.collection_id,
            collection_client.write_api_key,
        )
        .await
        .unwrap();

    collection
        .set_value("k1".to_string(), "v1".to_string())
        .await
        .unwrap();
    collection
        .set_value("k2".to_string(), "v2".to_string())
        .await
        .unwrap();
    drop(collection);

    // Wait for reader to see the values reflected in stats
    wait_for(&collection_client, |c| {
        let reader = c.reader;
        let read_api_key = c.read_api_key.clone();
        let collection_id = c.collection_id;
        async move {
            let stats = reader
                .collection_stats(
                    &read_api_key,
                    collection_id,
                    crate::types::CollectionStatsRequest { with_keys: false },
                )
                .await?;
            if stats.values_count != 2 {
                bail!(
                    "Expected 2 collection values in stats, got {}",
                    stats.values_count
                );
            }
            Ok(())
        }
        .boxed()
    })
    .await
    .expect("Collection values count should be reflected in stats");

    drop(test_context);
}
