use crate::tests::utils::TestContext;
use crate::tests::utils::{extrapolate_ids_from_result, init_log};
use serde_json::json;
use std::convert::TryInto;

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_simple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
              "id": "rule-1",
              "conditions": [
                {
                  "pattern": "c",
                  "anchoring": "is"
                }
              ],
              "consequence": {
                "promote": [
                  {
                    "doc_id": "5",
                    "position": 1
                  },
                  {
                    "doc_id": "7",
                    "position": 2
                  }
                ]
              }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    // The pin rule should promote document with id "5" to position 1
    // The actual ID will be in format "index_id:5"
    assert!(
        result.hits[1].id.ends_with(":5"),
        "Expected document ID ending with ':5', got: {}",
        result.hits[1].id
    );
    assert!(
        result.hits[2].id.ends_with(":7"),
        "Expected document ID ending with ':7', got: {}",
        result.hits[2].id
    );

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "7", "1", "2", "3", "4", "6", "8", "9"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_already_returned() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
              "id": "rule-1",
              "conditions": [
                {
                  "pattern": "c",
                  "anchoring": "is"
                }
              ],
              "consequence": {
                "promote": [
                  {
                    "doc_id": "0",
                    "position": 3
                  },
                ]
              }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2", "3", "0", "4", "5", "6", "7", "8", "9"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_after_the_pagination_window() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
              "id": "rule-1",
              "conditions": [
                {
                  "pattern": "c",
                  "anchoring": "is"
                }
              ],
              "consequence": {
                "promote": [
                  {
                    "doc_id": "0",
                    "position": 3000
                  },
                ]
              }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_pagination() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    index_client
        .insert_pin_rules(
            json!({
              "id": "rule-1",
              "conditions": [
                {
                  "pattern": "c",
                  "anchoring": "is"
                }
              ],
              "consequence": {
                "promote": [
                  {
                    "doc_id": "0",
                    "position": 3
                  },
                ]
              }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2", "3", "0", "4", "5", "6", "7", "8", "9"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 0,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["1", "2"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 1,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["2", "3"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 2,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["3", "0"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 3,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "4"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "limit": 2,
                "offset": 4,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids: Vec<_> = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["4", "5"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_before_insert_simple() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_pin_rules(
            json!({
              "id": "rule-1",
              "conditions": [
                {
                  "pattern": "c",
                  "anchoring": "is"
                }
              ],
              "consequence": {
                "promote": [
                  {
                    "doc_id": "5",
                    "position": 1
                  },
                  {
                    "doc_id": "7",
                    "position": 2
                  }
                ]
              }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c"
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "7", "1", "2", "3", "4", "6", "8", "9"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert_with_sort() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_pin_rules(
            json!({
              "id": "rule-1",
              "conditions": [
                {
                  "pattern": "c",
                  "anchoring": "is"
                }
              ],
              "consequence": {
                "promote": [
                  {
                    "doc_id": "5",
                    "position": 1
                  },
                  {
                    "doc_id": "7",
                    "position": 2
                  }
                ]
              }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let docs: Vec<_> = (0_u8..20_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
                "n": i,
            })
        })
        .collect();
    index_client
        .insert_documents(docs.try_into().unwrap())
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "sortBy": {
                    "order": "ASC",
                    "property": "n",
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(&ids, &["0", "5", "7", "1", "2", "3", "4", "6", "8", "9"]);

    let result = collection_client
        .search(
            json!({
                "term": "c",
                "sortBy": {
                    "order": "DESC",
                    "property": "n",
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&result);
    assert_eq!(
        &ids,
        &["19", "5", "7", "18", "17", "16", "15", "14", "13", "12"]
    );

    drop(test_context);
}
