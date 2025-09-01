use serde_json::{json, Value};
use std::collections::HashSet;

use crate::tests::utils::{extrapolate_ids_from_result, extrapolate_ids_from_result_hits, init_log, TestContext};

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
                    "max_results": 5,
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
    let values = groups
        .iter()
        .map(|g| g.values.clone())
        .collect::<HashSet<_>>();
    assert_eq!(
        values,
        HashSet::<Vec<Value>>::from([
            vec![4.into()],
            vec![3.into()],
            vec![2.into()],
            vec![1.into()],
            vec![0.into()],
        ])
    );
    for g in groups {
        assert!(g.result.len() < 5);
    }

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "groupBy": {
                    "properties": ["number", "bool"],
                    "max_results": 5,
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
    let values = groups
        .iter()
        .map(|g| g.values.clone())
        .collect::<HashSet<_>>();
    assert_eq!(
        values,
        HashSet::<Vec<Value>>::from([
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
        ])
    );
    for g in groups {
        assert!(g.result.len() < 5);
    }

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "groupBy": {
                    "properties": ["number", "bool", "string_filter"],
                    "max_results": 5,
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
    let values = groups
        .iter()
        .map(|g| g.values.clone())
        .collect::<HashSet<_>>();
    assert_eq!(
        values,
        HashSet::<Vec<Value>>::from([
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
        ])
    );
    for g in groups {
        assert!(g.result.len() < 5);
    }

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_sort_by_score_default() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple fruit", "category": "food", "price": 10},
        {"id": "doc2", "title": "apple phone", "category": "tech", "price": 100},
        {"id": "doc3", "title": "banana fruit", "category": "food", "price": 5},
        {"id": "doc4", "title": "orange tech", "category": "tech", "price": 50}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "apple",
        "groupBy": {
            "properties": ["category"],
            "max_results": 10,
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    let ids = extrapolate_ids_from_result(&results);
    assert_eq!(ids, vec!["doc1", "doc2"]);

    assert!(results.groups.is_some());
    let mut groups = results.groups.unwrap();
    assert_eq!(groups.len(), 2);

    groups.sort_by(|g1, g2| {
        g1.values[0].as_str().cmp(&g2.values[0].as_str())
    });

    assert_eq!(groups[0].values, Vec::from([json!("food")]));
    let ids = extrapolate_ids_from_result_hits(&groups[0].result);
    assert_eq!(ids, vec!["doc1"]);

    assert_eq!(groups[1].values, Vec::from([json!("tech")]));
    let ids = extrapolate_ids_from_result_hits(&groups[1].result);
    assert_eq!(ids, vec!["doc2"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_sort_by_field_ascending() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple", "category": "food", "price": 30},
        {"id": "doc2", "title": "banana", "category": "food", "price": 10},
        {"id": "doc3", "title": "cherry", "category": "food", "price": 20},
        {"id": "doc4", "title": "phone", "category": "tech", "price": 200},
        {"id": "doc5", "title": "laptop", "category": "tech", "price": 100}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "",
        "groupBy": {
            "properties": ["category"],
            "max_results": 10,
        },
        "sortBy": {
            "property": "price",
            "order": "ASC"
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let mut groups = results.groups.unwrap();
    assert_eq!(groups.len(), 2);

    groups.sort_by(|g1, g2| {
        g1.values[0].as_str().cmp(&g2.values[0].as_str())
    });

    assert_eq!(groups[0].values, vec![json!("food")]);
    let ids = extrapolate_ids_from_result_hits(&groups[0].result);
    assert_eq!(ids, vec!["doc2", "doc3", "doc1"]); // banana, cherry, apple

    assert_eq!(groups[1].values, vec![json!("tech")]);
    let ids = extrapolate_ids_from_result_hits(&groups[1].result);
    assert_eq!(ids, vec!["doc5", "doc4"]); // laptop, phone

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_sort_by_field_descending() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple", "category": "food", "price": 30},
        {"id": "doc2", "title": "banana", "category": "food", "price": 10},
        {"id": "doc3", "title": "phone", "category": "tech", "price": 200},
        {"id": "doc4", "title": "laptop", "category": "tech", "price": 100}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "",
        "groupBy": {
            "properties": ["category"],
            "max_results": 10,
        },
        "sortBy": {
            "property": "price",
            "order": "DESC"
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    // Should have groups
    assert!(results.groups.is_some());
    let mut groups = results.groups.unwrap();

    assert_eq!(groups.len(), 2);

    groups.sort_by(|g1, g2| {
        g1.values[0].as_str().cmp(&g2.values[0].as_str())
    });

    assert_eq!(groups[0].values, vec![json!("food")]);
    let ids = extrapolate_ids_from_result_hits(&groups[0].result);
    assert_eq!(ids, vec!["doc1", "doc2"]);

    assert_eq!(groups[1].values, vec![json!("tech")]);
    let ids = extrapolate_ids_from_result_hits(&groups[1].result);
    assert_eq!(ids, vec!["doc3", "doc4"]);

    drop(test_context);
}
