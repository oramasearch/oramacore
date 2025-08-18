use anyhow::Result;
use serde_json::json;

use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_search_multi_index_basic() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let index_2_client = collection_client.create_index().await.unwrap();

    let document_count = 10;
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    index_1_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "bool": i % 2 == 0,
            })
        })
        .collect::<Vec<_>>();
    index_2_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Search on both indexes
    let res = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, document_count * 2);

    // Bool is present only on the second index
    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "bool": true,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 5);

    // number is present only on the first index
    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gt": -1
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 10);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_search_multi_index_one_is_empty() -> Result<()> {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let index_2_client = collection_client.create_index().await.unwrap();

    let document_count = 10;
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    println!("{docs:#?}");
    index_1_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // Keep index 2 empty

    // Search on both indexes
    let res = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, document_count);

    // Number is present only on the first index
    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gt": -1
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 10);

    let res = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "text": "text "
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(res.count, 1);

    Ok(())
}
