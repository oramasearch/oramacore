use std::time::Duration;

use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::{init_log, wait_for, TestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_index_replacement_1() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    index_1_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Tommaso" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let index_2_client = collection_client
        .create_temp_index(index_1_client.index_id)
        .await
        .unwrap();
    index_2_client
        .insert_documents(
            json!([json!({ "id": "1", "text": "Michele" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);

    collection_client
        .replace_index(index_1_client.index_id, index_2_client.index_id)
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    test_context.commit_all().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 0);
    let result = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_replacement_2() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    index_1_client
        .insert_documents(
            json!([json!({ "id": "1", "name": "Tommaso", "surname": "Allevi" })])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let index_2_client = collection_client
        .create_temp_index(index_1_client.index_id)
        .await
        .unwrap();
    index_2_client
        .insert_documents(
            json!([
                json!({ "id": "1", "name": "Tommaso", "surname": "Allevi", "new": true }),
                json!({ "id": "2", "name": "Michele", "surname": "Riva", "new": true }),
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    collection_client
        .replace_index(index_1_client.index_id, index_2_client.index_id)
        .await
        .unwrap();

    wait_for(&collection_client, |c| {
        async move {
            let result = c
                .search(
                    json!({
                        "term": "Tommaso",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();

            if result.hits.len() != 1 {
                return Err(anyhow::anyhow!("Unexpected number of hits"));
            }

            let Some(first) = result.hits.first() else {
                return Err(anyhow::anyhow!("No results found"));
            };
            first
                .document
                .as_ref()
                .and_then(|doc| doc.get("new"))
                .and_then(|v| v.as_bool())
                .ok_or_else(|| anyhow::anyhow!("Document does not contain 'new' field"))
                .map(|is_new| {
                    if is_new {
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("Document 'new' field is not true"))
                    }
                })
        }
        .boxed()
    })
    .await
    .unwrap()
    .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);
    assert_eq!(result.hits.len(), 1);
    assert_eq!(result.hits[0].id, format!("{}:1", index_1_client.index_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_replacement_3() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    index_client
        .insert_documents(
            json!([
                {"id": "1", "name": "Tommaso"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 1);

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(
        output.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "1"
    );

    let temp_coll_client = collection_client
        .create_temp_index(index_client.index_id)
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 1);
    assert_eq!(stats[0].indexes.len(), 2);
    assert_eq!(stats[0].indexes[0].id, index_client.index_id);
    assert_eq!(stats[0].indexes[1].id, temp_coll_client.index_id);

    temp_coll_client
        .insert_documents(
            json!([
                {"id": "2", "name": "Michele"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 2);
    assert_eq!(stats[0].indexes.len(), 2);
    assert_eq!(stats[0].indexes[0].id, index_client.index_id);
    assert_eq!(stats[0].indexes[1].id, temp_coll_client.index_id);

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1); // not yet replaced, so we see only the original document
    assert_eq!(
        output.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "1"
    );

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.document_count, 2);
    assert_eq!(stats.indexes_stats.len(), 2);
    assert_eq!(stats.indexes_stats[0].id, index_client.index_id);
    assert!(!stats.indexes_stats[0].is_temp);
    assert_eq!(stats.indexes_stats[0].document_count, 1);
    assert_eq!(stats.indexes_stats[1].id, temp_coll_client.index_id);
    assert!(stats.indexes_stats[1].is_temp);
    assert_eq!(stats.indexes_stats[1].document_count, 1);

    collection_client
        .replace_index(index_client.index_id, temp_coll_client.index_id)
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 1);
    assert_eq!(stats[0].indexes.len(), 1);
    assert_eq!(stats[0].indexes[0].id, index_client.index_id);

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(
        output.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "2"
    );

    index_client
        .insert_documents(
            json!([
                {"id": "1", "name": "Tommaso"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.document_count, 2);
    assert_eq!(stats.indexes_stats.len(), 1);
    assert_eq!(stats.indexes_stats[0].id, index_client.index_id);
    assert!(!stats.indexes_stats[0].is_temp);
    assert_eq!(stats.indexes_stats[0].document_count, 2);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_replacement_3_with_commit() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    index_client
        .insert_documents(
            json!([
                {"id": "1", "name": "Tommaso"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 1);

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(
        output.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "1"
    );

    let temp_coll_client = collection_client
        .create_temp_index(index_client.index_id)
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 1);
    assert_eq!(stats[0].indexes.len(), 2);
    assert_eq!(stats[0].indexes[0].id, index_client.index_id);
    assert_eq!(stats[0].indexes[1].id, temp_coll_client.index_id);

    temp_coll_client
        .insert_documents(
            json!([
                {"id": "2", "name": "Michele"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 2);
    assert_eq!(stats[0].indexes.len(), 2);
    assert_eq!(stats[0].indexes[0].id, index_client.index_id);
    assert_eq!(stats[0].indexes[1].id, temp_coll_client.index_id);

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1); // not yet replaced, so we see only the original document
    assert_eq!(
        output.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "1"
    );

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.document_count, 2);
    assert_eq!(stats.indexes_stats.len(), 2);
    assert_eq!(stats.indexes_stats[0].id, index_client.index_id);
    assert!(!stats.indexes_stats[0].is_temp);
    assert_eq!(stats.indexes_stats[0].document_count, 1);
    assert_eq!(stats.indexes_stats[1].id, temp_coll_client.index_id);
    assert!(stats.indexes_stats[1].is_temp);
    assert_eq!(stats.indexes_stats[1].document_count, 1);

    collection_client
        .replace_index(index_client.index_id, temp_coll_client.index_id)
        .await
        .unwrap();

    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].document_count, 1);
    assert_eq!(stats[0].indexes.len(), 1);
    assert_eq!(stats[0].indexes[0].id, index_client.index_id);

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);
    assert_eq!(
        output.hits[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "2"
    );

    index_client
        .insert_documents(
            json!([
                {"id": "1", "name": "Tommaso"},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.document_count, 2);
    assert_eq!(stats.indexes_stats.len(), 1);
    assert_eq!(stats.indexes_stats[0].id, index_client.index_id);
    assert!(!stats.indexes_stats[0].is_temp);
    assert_eq!(stats.indexes_stats[0].document_count, 2);

    test_context.commit_all().await.unwrap();

    drop(test_context);
}
