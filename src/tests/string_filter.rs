use anyhow::Result;
use redact::Secret;
use serde_json::json;

use crate::{
    collection_manager::{
        dto::ApiKey,
        sides::{
            stats::{StringFilterCommittedFieldStats, StringFilterUncommittedFieldStats},
            FieldStatsType,
        },
    },
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::CollectionId,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_string_filter() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![
            json!({
                "id": "1",
                "name": "John Doe",
                "section": "my-section-1",
            }),
            json!({
                "id": "2",
                "name": "Jane Doe",
                "section": "my-section-2",
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "section": "my-section-1",
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);
    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await
        .unwrap();
    let (uncommitted, committed) = stats
        .fields_stats
        .into_iter()
        .filter_map(|field| {
            if field.name == "section" {
                return None;
            }
            if let FieldStatsType::StringFilter {
                uncommitted,
                committed,
            } = field.stats
            {
                return Some((uncommitted, committed));
            }
            None
        })
        .next()
        .unwrap();
    // It is not committed yet
    assert!(matches!(
        uncommitted,
        Some(StringFilterUncommittedFieldStats {
            variant_count: 2,
            doc_count: 2
        })
    ));
    assert!(committed.is_none());

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "section": "my-section-1",
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await
        .unwrap();
    let (uncommitted, committed) = stats
        .fields_stats
        .into_iter()
        .filter_map(|field| {
            if field.name == "section" {
                return None;
            }
            if let FieldStatsType::StringFilter {
                uncommitted,
                committed,
            } = field.stats
            {
                return Some((uncommitted, committed));
            }
            None
        })
        .next()
        .unwrap();

    // It is committed
    assert!(uncommitted.is_none());
    assert!(matches!(
        committed,
        Some(StringFilterCommittedFieldStats {
            variant_count: 2,
            doc_count: 2
        })
    ));

    let (_, read_side) = create(config.clone()).await?;
    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "section": "my-section-1",
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await
        .unwrap();
    let (uncommitted, committed) = stats
        .fields_stats
        .into_iter()
        .filter_map(|field| {
            if field.name == "section" {
                return None;
            }
            if let FieldStatsType::StringFilter {
                uncommitted,
                committed,
            } = field.stats
            {
                return Some((uncommitted, committed));
            }
            None
        })
        .next()
        .unwrap();
    // It is committed
    assert!(uncommitted.is_none());
    assert!(matches!(
        committed,
        Some(StringFilterCommittedFieldStats {
            variant_count: 2,
            doc_count: 2
        })
    ));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_string_filter_long_text() {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await.unwrap();

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id)
        .await
        .unwrap();

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id,
        vec![json!({
            "title": "Today I want to listen only Max Pezzali.",
        })],
    )
    .await
    .unwrap();

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await
        .unwrap();
    let (uncommitted, committed) = stats
        .fields_stats
        .into_iter()
        .filter_map(|field| {
            if field.name != "title" {
                return None;
            }
            if let FieldStatsType::StringFilter {
                uncommitted,
                committed,
            } = field.stats
            {
                return Some((uncommitted, committed));
            }
            None
        })
        .next()
        .unwrap();
    // Text is too long to be indexed
    assert!(matches!(
        uncommitted,
        Some(StringFilterUncommittedFieldStats {
            variant_count: 0,
            doc_count: 0
        })
    ));
    assert!(committed.is_none());

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await
        .unwrap();
    let (uncommitted, committed) = stats
        .fields_stats
        .into_iter()
        .filter_map(|field| {
            if field.name != "title" {
                return None;
            }
            if let FieldStatsType::StringFilter {
                uncommitted,
                committed,
            } = field.stats
            {
                return Some((uncommitted, committed));
            }
            None
        })
        .next()
        .unwrap();
    // Text is too long to be indexed
    assert!(uncommitted.is_none());
    assert!(matches!(
        committed,
        Some(StringFilterCommittedFieldStats {
            variant_count: 0,
            doc_count: 0
        })
    ));

    let (_, read_side) = create(config.clone()).await.unwrap();

    let stats = read_side
        .collection_stats(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
        )
        .await
        .unwrap();
    let (uncommitted, committed) = stats
        .fields_stats
        .into_iter()
        .filter_map(|field| {
            if field.name != "title" {
                return None;
            }
            if let FieldStatsType::StringFilter {
                uncommitted,
                committed,
            } = field.stats
            {
                return Some((uncommitted, committed));
            }
            None
        })
        .next()
        .unwrap();
    // Text is too long to be indexed
    assert!(uncommitted.is_none());
    assert!(matches!(
        committed,
        Some(StringFilterCommittedFieldStats {
            variant_count: 0,
            doc_count: 0
        })
    ));
}
