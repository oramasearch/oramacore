use anyhow::bail;
use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_search_ok() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client.insert_documents(json!([
        {
            "id": "1",
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        },
        {
            "id": "2",
            "text": "Curabitur sem tortor, interdum in rutrum in, dignissim vestibulum metus.",
        }
    ]).try_into().unwrap()).await.unwrap();
    index_client
        .delete_documents(vec!["1".to_string()])
        .await
        .unwrap();

    wait_for(&collection_client, |collection_client| {
        Box::pin(async move {
            let output = collection_client
                .search(
                    json!({
                        "term": "Lorem ipsum",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await?;
            if output.count == 0 {
                Ok(())
            } else {
                bail!("Document not deleted yet");
            }
        })
    })
    .await
    .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "Curabitur sem tortor",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_search_unexisting_id() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client.insert_documents(json!([
        {
            "id": "1",
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        },
        {
            "id": "2",
            "text": "Curabitur sem tortor, interdum in rutrum in, dignissim vestibulum metus.",
        }
    ]).try_into().unwrap()).await.unwrap();
    index_client
        .uncheck_delete_documents(vec!["3".to_string()])
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert!(output.hits[0].score > 0.);

    drop(test_context);
}

/// Test to verify that WriteSide correctly decrements the document count in stats
/// after document deletion. This test specifically checks the WriteSide behavior
/// to confirm that document counts are updated immediately after deletion.
#[tokio::test(flavor = "multi_thread")]
async fn test_writeside_stats_decrements_after_delete() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // 1. Insert 3 documents
    index_client
        .insert_documents(
            json!([
                { "id": "1", "text": "First document" },
                { "id": "2", "text": "Second document" },
                { "id": "3", "text": "Third document" }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // 2. Check WriteSide stats shows 3 documents
    let stats = test_context.get_writer_collections().await;
    assert_eq!(stats[0].document_count, 3);
    assert_eq!(stats[0].indexes[0].document_count, 3);

    // 3. Delete one document (use uncheck_delete_documents to test WriteSide directly)
    index_client
        .uncheck_delete_documents(vec!["2".to_string()])
        .await
        .unwrap();

    // 4. Check WriteSide stats shows 2 documents
    let stats = test_context.get_writer_collections().await;
    assert_eq!(
        stats[0].document_count, 2,
        "WriteSide collection stats should show 2 documents after deletion"
    );
    assert_eq!(
        stats[0].indexes[0].document_count, 2,
        "WriteSide index stats should show 2 documents after deletion"
    );

    drop(test_context);
}

/// Regression test for integer underflow in DocumentLengthsPerDocument::remove_doc_ids().
///
/// Scenario: Sparse fields where deleted documents never existed cause total_documents
/// to decrement incorrectly. Example: committed 'artist' field has 1 doc, we delete 2 docs
/// (1 that exists + 1 that doesn't) -> 1 - 2 = underflow panic.
#[tokio::test(flavor = "multi_thread")]
async fn test_delete_document_missing_field_underflow() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {"id": "song_1", "title": "First Song", "artist": "Artist A"},
                {"id": "song_2", "title": "Second Song", "artist": "Artist B"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    // Create the field indexes
    test_context.commit_all().await.unwrap();

    // Remove the docs to make the "artist" doc count to 0
    index_client
        .delete_documents(vec!["song_1".to_string(), "song_2".to_string()])
        .await
        .unwrap();

    index_client
        .insert_documents(
            json!([
                {"id": "song_3", "title": "Third Song"},  // no artist
                {"id": "song_4", "title": "Fourth Song", "artist": "Artist D"}
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    index_client
        .delete_documents(vec!["song_3".to_string()])
        .await
        .unwrap();

    // This insert + commit triggers the panic
    index_client
        .insert_documents(
            json!([{"id": "song_5", "title": "Fifth", "artist": "Artist E"}])
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();
}
