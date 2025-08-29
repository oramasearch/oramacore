use std::collections::HashSet;
use serde_json::{json, Value};

use crate::tests::utils::{init_log, TestContext};

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by_number() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i % 5,
                "bool": i % 2 == 0,
                "string_filter": format!("s{}", i % 3),
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "groupBy": {
                    "properties": ["number"],
                    "maxResult": 5,
                },
                "limit": 0,
            })
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();
    let Some(groups) = output.groups else {
        panic!("`groups` should be there");
    };
    let values = groups.iter().map(|g| g.values.clone()).collect::<HashSet<_>>();
    assert_eq!(values, HashSet::<Vec<Value>>::from([
        vec![4.into()],
        vec![3.into()],
        vec![2.into()],
        vec![1.into()],
        vec![0.into()],
    ]));
    for g in groups {
        assert!(g.result.len() < 5);
    }

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "groupBy": {
                    "properties": ["number", "bool"],
                    "maxResult": 5,
                },
                "limit": 0,
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let Some(groups) = output.groups else {
        panic!("`groups` should be there");
    };
    let values = groups.iter().map(|g| g.values.clone()).collect::<HashSet<_>>();
    assert_eq!(values, HashSet::<Vec<Value>>::from([
        vec![4.into(), true.into()],
        vec![4.into(), false.into()],
        vec![3.into(), true.into()],
        vec![3.into(), false.into()],
        vec![2.into(), true.into()],
        vec![2.into(), false.into()],
        vec![1.into(), true.into()],
        vec![1.into(), false.into()],
        vec![0.into(), true.into()],
        vec![0.into(), false.into()],
    ]));
    for g in groups {
        assert!(g.result.len() < 5);
    }

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "groupBy": {
                    "properties": ["number", "bool", "string_filter"],
                    "maxResult": 5,
                },
                "limit": 0,
            })
                .try_into()
                .unwrap(),
        )
        .await
        .unwrap();

    let Some(groups) = output.groups else {
        panic!("`groups` should be there");
    };
    let values = groups.iter().map(|g| g.values.clone()).collect::<HashSet<_>>();
    assert_eq!(values, HashSet::<Vec<Value>>::from([
        vec![4.into(), true.into(), "s0".into()],
        vec![4.into(), true.into(), "s1".into()],
        vec![4.into(), true.into(), "s2".into()],
        vec![4.into(), false.into(), "s0".into()],
        vec![4.into(), false.into(), "s1".into()],
        vec![4.into(), false.into(), "s2".into()],
        vec![3.into(), true.into(), "s0".into()],
        vec![3.into(), true.into(), "s1".into()],
        vec![3.into(), true.into(), "s2".into()],
        vec![3.into(), false.into(), "s0".into()],
        vec![3.into(), false.into(), "s1".into()],
        vec![3.into(), false.into(), "s2".into()],
        vec![2.into(), true.into(), "s0".into()],
        vec![2.into(), true.into(), "s1".into()],
        vec![2.into(), true.into(), "s2".into()],
        vec![2.into(), false.into(), "s0".into()],
        vec![2.into(), false.into(), "s1".into()],
        vec![2.into(), false.into(), "s2".into()],
        vec![1.into(), true.into(), "s0".into()],
        vec![1.into(), true.into(), "s1".into()],
        vec![1.into(), true.into(), "s2".into()],
        vec![1.into(), false.into(), "s0".into()],
        vec![1.into(), false.into(), "s1".into()],
        vec![1.into(), false.into(), "s2".into()],
        vec![0.into(), true.into(), "s0".into()],
        vec![0.into(), true.into(), "s1".into()],
        vec![0.into(), true.into(), "s2".into()],
        vec![0.into(), false.into(), "s0".into()],
        vec![0.into(), false.into(), "s1".into()],
        vec![0.into(), false.into(), "s2".into()],
    ]));
    for g in groups {
        assert!(g.result.len() < 5);
    }

    drop(test_context);
}
