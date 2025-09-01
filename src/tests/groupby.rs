use serde_json::{json, Value};
use std::collections::HashSet;

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

// New tests for group sorting functionality

#[tokio::test(flavor = "multi_thread")]
async fn test_group_sort_by_score_default() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with different relevance
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

    // Search for "apple" and group by category - should sort by relevance score within groups
    let search_params = json!({
        "term": "apple",
        "groupBy": {
            "properties": ["category"],
            "maxResults": 10,
        }
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    // Should have groups
    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();

    // Should have two groups: "food" and "tech"
    assert_eq!(groups.len(), 2);

    // Check that documents within each group are sorted by score (relevance)
    for group in &groups {
        if group.values.contains(&json!("food")) {
            // In food group, "apple fruit" should come first (higher relevance)
            assert_eq!(group.result.len(), 1);
            assert!(group.result[0].id.contains("doc1"));
        } else if group.values.contains(&json!("tech")) {
            // In tech group, "apple phone" should come first (higher relevance)
            assert_eq!(group.result.len(), 1);
            assert!(group.result[0].id.contains("doc2"));
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_sort_by_field_ascending() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with different prices in same categories
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

    // Group by category and sort by price ascending
    let search_params = json!({
        "term": "",
        "groupBy": {
            "properties": ["category"],
            "maxResults": 10,
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

    // Should have groups
    assert!(results.groups.is_some());
    let groups = results.groups.unwrap();

    // Should have two groups
    assert_eq!(groups.len(), 2);

    // Check sorting within groups by checking document ID order
    for group in &groups {
        if group.values.contains(&json!("food")) {
            // Food group should be sorted by price ascending: banana(10) first
            let ids: Vec<String> = group.result.iter().map(|hit| hit.id.clone()).collect();
            assert!(!group.result.is_empty());
            assert!(ids[0].contains("doc2")); // banana (lowest price first)
        } else if group.values.contains(&json!("tech")) {
            // Tech group should be sorted by price ascending: laptop(100) first
            let ids: Vec<String> = group.result.iter().map(|hit| hit.id.clone()).collect();
            assert!(!group.result.is_empty());
            assert!(ids[0].contains("doc5")); // laptop (lowest price first)
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_group_sort_by_field_descending() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents
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

    // Group by category and sort by price descending
    let search_params = json!({
        "term": "",
        "groupBy": {
            "properties": ["category"],
            "maxResults": 10,
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
    let groups = results.groups.unwrap();

    // Check sorting within groups by checking document ID order
    for group in &groups {
        if group.values.contains(&json!("food")) {
            // Food group should be sorted by price descending: apple(30) first
            let ids: Vec<String> = group.result.iter().map(|hit| hit.id.clone()).collect();
            assert!(ids[0].contains("doc1")); // apple (highest price first)
        } else if group.values.contains(&json!("tech")) {
            // Tech group should be sorted by price descending: phone(200) first
            let ids: Vec<String> = group.result.iter().map(|hit| hit.id.clone()).collect();
            assert!(ids[0].contains("doc3")); // phone (highest price first)
        }
    }
}
