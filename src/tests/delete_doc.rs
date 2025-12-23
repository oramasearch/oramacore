use anyhow::bail;
use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::tests::utils::wait_for;

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
    }).await.unwrap();

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
