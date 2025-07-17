use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_number() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "year": 1990,
                },
                {
                    "id": "2",
                    "name": "Michele",
                    "year": 1994,
                }
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
                "sortBy": {
                    "property": "year",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "year",
                    "order": "ASC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "year",
                    "order": "DESC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_date() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    // The date is not correct, otherwise we leak personal data
                    "birthday": "1990-01-01T00:00:00Z",
                },
                {
                    "id": "2",
                    "name": "Michele",
                    // The date is not correct, otherwise we leak personal data
                    "birthday": "1994-01-01T00:00:00Z",
                }
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
                "sortBy": {
                    "property": "birthday",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "birthday",
                    "order": "ASC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "birthday",
                    "order": "DESC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_bool() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "is_good": false,
                },
                {
                    "id": "2",
                    "name": "Michele",
                    "is_good": true,
                }
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
                "sortBy": {
                    "property": "is_good",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "is_good",
                    "order": "ASC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "1")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "2")
    );

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "is_good",
                    "order": "DESC"
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.hits.len(), 2);
    assert_eq!(
        output.hits[0].id,
        format!("{}:{}", index_client.index_id, "2")
    );
    assert_eq!(
        output.hits[1].id,
        format!("{}:{}", index_client.index_id, "1")
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_unknown_field() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    collection_client.create_index().await.unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "",
                "sortBy": {
                    "property": "unknown_field",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await;

    let err = output.unwrap_err();
    assert!(format!("{err:?}").contains("Cannot sort by \"unknown_field\": unknown field"));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn tests_sort_on_unsupported_field() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "name": "Tommaso",
                    "position": {
                        "lat": 45.0,
                        "lon": 7.0,
                    }
                },
                {
                    "id": "2",
                    "name": "Michele",
                    "position": {
                        "lat": 46.0,
                        "lon": 8.0,
                    }
                }
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
                "sortBy": {
                    "property": "position",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await;

    let err = output.unwrap_err();
    assert!(format!("{err:?}").contains("Only number, date or boolean fields are supported for sorting, but got GeoPoint for property \"position\""));

    drop(test_context);
}
