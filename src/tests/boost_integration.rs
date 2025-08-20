use crate::tests::utils::{init_log, TestContext};
use serde_json::json;

/// Test boost functionality with real search scenarios
#[tokio::test(flavor = "multi_thread")]
async fn test_boost_no_boost_comparison() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert test documents with different field content
    // doc1 has "machine learning" in title (high density) and content (lower density due to length)
    let documents = json!([
        {
            "id": "doc1",
            "title": "machine learning",
            "content": "This comprehensive document provides detailed information about various algorithms and techniques used in modern data processing applications. The field includes machine learning which encompasses many different approaches and methodologies for analysis and prediction."
        },
        {
            "id": "doc2",
            "title": "data analysis techniques",
            "content": "Advanced machine learning models and frameworks for comprehensive data analysis, statistical processing, and predictive modeling in various business applications and research contexts."
        },
        {
            "id": "doc3",
            "title": "statistical methods",
            "content": "Comprehensive overview of machine learning methodologies combined with traditional statistical approaches for effective data analysis, pattern recognition, and business intelligence applications."
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Test 1: Search without boost (baseline)
    let search_params_no_boost = json!({
        "term": "machine learning",
        "properties": ["title", "content"],
        "boost": {}
    });

    let results_no_boost = collection_client
        .search(search_params_no_boost.try_into().unwrap())
        .await
        .unwrap();

    // Test 2: Search with title boost
    let search_params_title_boost = json!({
        "term": "machine learning",
        "properties": ["title", "content"],
        "boost": {
            "title": 3.0,
            "content": 1.0
        }
    });

    let results_title_boost = collection_client
        .search(search_params_title_boost.try_into().unwrap())
        .await
        .unwrap();

    // Test 3: Search with content boost
    let search_params_content_boost = json!({
        "term": "machine learning",
        "properties": ["title", "content"],
        "boost": {
            "title": 1.0,
            "content": 3.0
        }
    });

    let results_content_boost = collection_client
        .search(search_params_content_boost.try_into().unwrap())
        .await
        .unwrap();

    // Verify all searches return results
    assert!(
        results_no_boost.count > 0,
        "No boost search should return results"
    );
    assert!(
        results_title_boost.count > 0,
        "Title boost search should return results"
    );
    assert!(
        results_content_boost.count > 0,
        "Content boost search should return results"
    );

    // Find specific documents in results
    let find_doc_score = |results: &crate::types::SearchResult, doc_id: &str| -> Option<f32> {
        results
            .hits
            .iter()
            .find(|hit| {
                let document = hit.document.as_ref().unwrap();
                serde_json::from_str::<serde_json::Value>(document.inner.get())
                    .unwrap()
                    .get("id")
                    .and_then(|v| v.as_str())
                    == Some(doc_id)
            })
            .map(|hit| hit.score)
    };

    let doc1_score_no_boost = find_doc_score(&results_no_boost, "doc1").unwrap();
    let doc1_score_title_boost = find_doc_score(&results_title_boost, "doc1").unwrap();
    let doc1_score_content_boost = find_doc_score(&results_content_boost, "doc1").unwrap();

    // doc1 has "machine learning" in title, so title boost should increase its score significantly
    assert!(
        doc1_score_title_boost > doc1_score_no_boost,
        "Title boost should increase doc1 score: {doc1_score_title_boost} vs {doc1_score_no_boost}"
    );

    // The title boost should be more significant than content boost for doc1
    // (since "machine learning" appears in doc1's title)
    assert!(
        doc1_score_title_boost > doc1_score_content_boost,
        "Title boost should be more effective for doc1: {doc1_score_title_boost} vs {doc1_score_content_boost}"
    );

    // Calculate boost effectiveness
    let title_boost_ratio = doc1_score_title_boost / doc1_score_no_boost;
    assert!(
        title_boost_ratio > 1.1,
        "Title boost should provide meaningful score increase: {title_boost_ratio}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multi_field_boost_ranking() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents where the query term appears in different fields
    let documents = json!([
        {
            "id": "title_match",
            "title": "advanced algorithms research",
            "content": "This paper explores various computational methods",
            "category": "research"
        },
        {
            "id": "content_match",
            "title": "computational methods overview",
            "content": "Detailed study of advanced algorithms and their applications",
            "category": "tutorial"
        },
        {
            "id": "category_match",
            "title": "machine learning basics",
            "content": "Introduction to ML concepts and frameworks",
            "category": "advanced algorithms guide"
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Test with different boost configurations
    let search_configs = vec![
        ("no_boost", json!({})),
        (
            "title_heavy",
            json!({"title": 5.0, "content": 1.0, "category": 1.0}),
        ),
        (
            "content_heavy",
            json!({"title": 1.0, "content": 5.0, "category": 1.0}),
        ),
        (
            "category_heavy",
            json!({"title": 1.0, "content": 1.0, "category": 5.0}),
        ),
    ];

    for (config_name, boost_config) in search_configs {
        let search_params = json!({
            "term": "advanced algorithms",
            "properties": ["title", "content", "category"],
            "boost": boost_config
        });

        let results = collection_client
            .search(search_params.try_into().unwrap())
            .await
            .unwrap();

        assert!(
            results.count >= 2,
            "{config_name}: Should find multiple documents"
        );

        // Check ranking based on boost configuration
        let rankings: Vec<String> = results
            .hits
            .iter()
            .map(|hit| {
                let document = hit.document.as_ref().unwrap();
                serde_json::from_str::<serde_json::Value>(document.inner.get())
                    .unwrap()
                    .get("id")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();

        match config_name {
            "title_heavy" => {
                // title_match should rank higher with title boost
                let title_match_pos = rankings.iter().position(|id| id == "title_match").unwrap();
                let content_match_pos = rankings
                    .iter()
                    .position(|id| id == "content_match")
                    .unwrap();
                assert!(
                    title_match_pos < content_match_pos,
                    "Title boost should rank title_match higher: {rankings:?}"
                );
            }
            "content_heavy" => {
                // content_match should rank higher with content boost
                let title_match_pos = rankings.iter().position(|id| id == "title_match").unwrap();
                let content_match_pos = rankings
                    .iter()
                    .position(|id| id == "content_match")
                    .unwrap();
                assert!(
                    content_match_pos < title_match_pos,
                    "Content boost should rank content_match higher: {rankings:?}"
                );
            }
            "category_heavy" => {
                // category_match should rank higher with category boost
                let category_match_pos = rankings.iter().position(|id| id == "category_match");
                assert!(
                    category_match_pos.is_some(),
                    "Category boost should find category_match: {rankings:?}"
                );
            }
            _ => {} // no_boost baseline
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_boost_with_phrase_matching() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents with phrase matches in different fields
    let documents = json!([
        {
            "id": "exact_title",
            "title": "machine learning tutorial",
            "content": "This covers basic concepts"
        },
        {
            "id": "exact_content",
            "title": "AI guide",
            "content": "Complete machine learning tutorial with examples"
        },
        {
            "id": "partial_match",
            "title": "machine intelligence",
            "content": "Learning algorithms and tutorial methods"
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Test phrase search with field boost
    let search_params = json!({
        "term": "machine learning tutorial",
        "properties": ["title", "content"],
        "boost": {
            "title": 4.0,
            "content": 1.0
        },
        "exact": true
    });

    let results = collection_client
        .search(search_params.try_into().unwrap())
        .await
        .unwrap();

    assert!(results.count > 0, "Should find phrase matches");

    // exact_title should score highest due to title boost + exact phrase match
    let rankings: Vec<String> = results
        .hits
        .iter()
        .map(|hit| {
            let document = hit.document.as_ref().unwrap();
            serde_json::from_str::<serde_json::Value>(document.inner.get())
                .unwrap()
                .get("id")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();

    let exact_title_score = results
        .hits
        .iter()
        .find(|hit| {
            let document = hit.document.as_ref().unwrap();
            serde_json::from_str::<serde_json::Value>(document.inner.get())
                .unwrap()
                .get("id")
                .and_then(|v| v.as_str())
                == Some("exact_title")
        })
        .map(|hit| hit.score)
        .unwrap();

    let exact_content_score = results
        .hits
        .iter()
        .find(|hit| {
            let document = hit.document.as_ref().unwrap();
            serde_json::from_str::<serde_json::Value>(document.inner.get())
                .unwrap()
                .get("id")
                .and_then(|v| v.as_str())
                == Some("exact_content")
        })
        .map(|hit| hit.score)
        .unwrap();

    // Title boost should make exact_title score higher than exact_content
    // even though both have exact phrase matches
    assert!(
        exact_title_score > exact_content_score,
        "Title boost should prioritize title matches: {exact_title_score} vs {exact_content_score}"
    );

    println!("Phrase matching test rankings: {rankings:?}");
    println!("exact_title score: {exact_title_score}, exact_content score: {exact_content_score}");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_boost_effectiveness_ratios() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert a simple document for controlled testing
    let documents = json!([
        {
            "id": "test_doc",
            "title": "test query term",
            "content": "some other content here"
        }
    ]);

    index_client
        .insert_documents(documents.try_into().unwrap())
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();

    // Test different boost levels
    let boost_levels = [1.0, 2.0, 3.0, 5.0];
    let mut scores = Vec::new();

    for boost_value in boost_levels {
        let search_params = json!({
            "term": "test query",
            "properties": ["title"],
            "boost": {
                "title": boost_value
            }
        });

        let results = collection_client
            .search(search_params.try_into().unwrap())
            .await
            .unwrap();

        assert!(
            results.count > 0,
            "Should find document with boost {boost_value}"
        );
        let score = results.hits[0].score;
        scores.push(score);

        println!("Boost {boost_value}: Score {score}");
    }

    // Verify that scores increase with boost values
    for i in 1..scores.len() {
        assert!(
            scores[i] > scores[i - 1],
            "Score should increase with boost: {} (boost {}) vs {} (boost {})",
            scores[i],
            boost_levels[i],
            scores[i - 1],
            boost_levels[i - 1]
        );
    }

    // Check that 2x boost gives meaningful improvement
    let ratio_2x = scores[1] / scores[0];
    assert!(
        ratio_2x > 1.2,
        "2x boost should provide substantial score increase: ratio = {ratio_2x}"
    );

    // Check that 5x boost is significantly better than 1x (but with diminishing returns)
    let ratio_5x = scores[3] / scores[0];
    assert!(
        ratio_5x > 1.5,
        "5x boost should provide major score increase: ratio = {ratio_5x}"
    );
}
