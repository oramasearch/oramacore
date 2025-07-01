use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::SearchParams;

#[tokio::test(flavor = "multi_thread")]
async fn test_geosearch() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "T",
                    "location": {
                        "lat": 9.0814233,
                        "lon": 45.2623823,
                    },
                },
                {
                    "id": "2",
                    "name": "T",
                    "location": {
                        "lat": 9.0979028,
                        "lon": 45.1995182,
                    },
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let p = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": {
                        "lat": 9.1418481,
                        "lon": 45.2324096
                    },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            },
        },
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p).await.unwrap();

    assert_eq!(output.count, 2);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_geosearch_commit() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "T",
                    "location": {
                        "lat": 9.0814233,
                        "lon": 45.2623823,
                    },
                },
                {
                    "id": "2",
                    "name": "T",
                    "location": {
                        "lat": 9.0979028,
                        "lon": 45.1995182,
                    },
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let p: SearchParams = json!({
        "term": "",
        "where": {
            "location": {
                "radius": {
                    "coordinates": {
                        "lat": 9.1418481,
                        "lon": 45.2324096
                    },
                    "unit": "km",
                    "value": 10,
                    "inside": true,
                }
            },
        },
    })
    .try_into()
    .unwrap();

    let output = collection_client.search(p.clone()).await.unwrap();
    assert_eq!(output.count, 2);

    test_context.commit_all().await.unwrap();

    let test_context = test_context.reload().await;

    let collection_client = test_context.get_test_collection_client(collection_client.collection_id, collection_client.write_api_key, collection_client.read_api_key).unwrap();

    let output = collection_client.search(p).await.unwrap();
    assert_eq!(output.count, 2);

    drop(test_context);
}
