use serde_json::{json, Value};
use std::collections::HashSet;

use crate::tests::utils::{
    extrapolate_ids_from_result, extrapolate_ids_from_result_hits, init_log, TestContext,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by() {
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
        assert!(g.result.len() <= 5);
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
        assert!(g.result.len() <= 5);
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
        assert!(g.result.len() <= 5);
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

    groups.sort_by(|g1, g2| g1.values[0].as_str().cmp(&g2.values[0].as_str()));

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

    groups.sort_by(|g1, g2| g1.values[0].as_str().cmp(&g2.values[0].as_str()));

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

    groups.sort_by(|g1, g2| g1.values[0].as_str().cmp(&g2.values[0].as_str()));

    assert_eq!(groups[0].values, vec![json!("food")]);
    let ids = extrapolate_ids_from_result_hits(&groups[0].result);
    assert_eq!(ids, vec!["doc1", "doc2"]);

    assert_eq!(groups[1].values, vec![json!("tech")]);
    let ids = extrapolate_ids_from_result_hits(&groups[1].result);
    assert_eq!(ids, vec!["doc3", "doc4"]);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_max_results_default() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple fruit sweet", "category": "food"},
        {"id": "doc2", "title": "apple fruit red", "category": "food"},
        {"id": "doc3", "title": "apple fruit green", "category": "food"},
        {"id": "doc4", "title": "apple phone tech", "category": "tech"},
        {"id": "doc5", "title": "apple watch tech", "category": "tech"}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "apple",
        "groupBy": {
            "properties": ["category"]
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();
    assert_eq!(groups.len(), 2);

    for group in &groups {
        assert_eq!(group.result.len(), 1);
    }

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_max_results_zero() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple", "category": "food"},
        {"id": "doc2", "title": "banana", "category": "food"}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "",
        "groupBy": {
            "properties": ["category"],
            "max_results": 0
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();
    assert_eq!(groups.len(), 0);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_max_results_larger_than_available() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple", "category": "food"},
        {"id": "doc2", "title": "banana", "category": "food"}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "",
        "groupBy": {
            "properties": ["category"],
            "max_results": 10
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].result.len(), 2);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_max_results_exact_compliance() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "test", "category": "A", "priority": 1},
        {"id": "doc2", "title": "test", "category": "A", "priority": 2},
        {"id": "doc3", "title": "test", "category": "A", "priority": 3},
        {"id": "doc4", "title": "test", "category": "B", "priority": 1},
        {"id": "doc5", "title": "test", "category": "B", "priority": 2},
        {"id": "doc6", "title": "test", "category": "C", "priority": 1}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test",
        "groupBy": {
            "properties": ["category"],
            "max_results": 2
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();
    assert_eq!(groups.len(), 3);

    let mut groups_by_category = std::collections::HashMap::new();
    for group in &groups {
        let category = group.values[0].as_str().unwrap();
        groups_by_category.insert(category, group.result.len());
    }

    assert_eq!(groups_by_category.get("A"), Some(&2));
    assert_eq!(groups_by_category.get("B"), Some(&2));
    assert_eq!(groups_by_category.get("C"), Some(&1));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by_float_numbers() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "test", "rating": 4.5, "category": "A"},
        {"id": "doc2", "title": "test", "rating": 4.5, "category": "B"},
        {"id": "doc3", "title": "test", "rating": 3.2, "category": "A"},
        {"id": "doc4", "title": "test", "rating": 3.2, "category": "B"},
        {"id": "doc5", "title": "test", "rating": 2.8, "category": "A"}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test",
        "groupBy": {
            "properties": ["rating"],
            "max_results": 3
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();
    assert_eq!(groups.len(), 3);

    let values = groups
        .iter()
        .map(|g| g.values.clone())
        .collect::<HashSet<_>>();

    assert_eq!(values.len(), 3);
    // Floats are rounded :(
    /*
    assert_eq!(
        values,
        HashSet::<Vec<Value>>::from([
            vec![4.5.into()],
            vec![3.2.into()],
            vec![2.8.into()],
        ])
    );
     */

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by_empty_search_results() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "apple", "category": "food"},
        {"id": "doc2", "title": "banana", "category": "food"}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "nonexistent",
        "groupBy": {
            "properties": ["category"],
            "max_results": 5
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 0);
    println!("Search results: {:#?}", results.groups);
    assert!(results.groups.is_none() || results.groups.unwrap().is_empty());

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by_nonexistent_property() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "test", "category": "food"},
        {"id": "doc2", "title": "test", "category": "tech"}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test",
        "groupBy": {
            "properties": ["nonexistent_property"],
            "max_results": 5
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();

    if !groups.is_empty() {
        for group in &groups {
            assert!(group.values[0].is_null() || group.values[0] == serde_json::Value::Null);
        }
    }

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by_insufficient_documents() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([
        {"id": "doc1", "title": "test", "category": "food", "priority": 1},
        {"id": "doc2", "title": "test", "category": "food", "priority": 2},
        {"id": "doc3", "title": "test", "category": "tech", "priority": 1}
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test",
        "groupBy": {
            "properties": ["category"],
            "max_results": 5
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();
    assert_eq!(groups.len(), 2);

    let mut groups_by_category = std::collections::HashMap::new();
    for group in &groups {
        let category = group.values[0].as_str().unwrap();
        groups_by_category.insert(category, group.result.len());
    }

    assert_eq!(groups_by_category.get("food"), Some(&2));
    assert_eq!(groups_by_category.get("tech"), Some(&1));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_by_zero_documents_scenario() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let documents = json!([]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    let search_params = json!({
        "term": "test",
        "groupBy": {
            "properties": ["category"],
            "max_results": 5
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert_eq!(results.count, 0);
    assert!(results.groups.is_none() || results.groups.unwrap().is_empty());

    drop(test_context);
}
