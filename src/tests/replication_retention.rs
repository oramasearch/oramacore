use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::json;

use crate::build_orama;
use crate::collection_manager::sides::read::ReplicationStatus;
use crate::collection_manager::sides::{
    validate_delivery_offset, validate_empty_stream, validate_retained_range, Offset,
    ReplicationRefusal,
};
use crate::tests::utils::{create_oramacore_config, init_log, TestCollectionClient, TestContext};
use crate::types::{CollectionStatsRequest, DocumentList};

/// One case per row of the refusal rules: the stream's retained range, an empty stream and
/// the offset the broker actually delivers.
#[test]
fn retained_history_and_actual_replay_fail_closed() {
    // Stream holds 10..=20.
    assert_eq!(
        validate_retained_range(Offset(10), Offset(10), Offset(20)),
        Ok(())
    );
    assert_eq!(
        validate_retained_range(Offset(21), Offset(10), Offset(20)),
        Ok(()),
        "a reader that has applied the head starts at head + 1"
    );
    assert_eq!(
        validate_retained_range(Offset(0), Offset(0), Offset(0)),
        Ok(())
    );

    let expired = validate_retained_range(Offset(9), Offset(10), Offset(20)).unwrap_err();
    assert_eq!(
        expired.to_string(),
        "REPLICATION_HISTORY_EXPIRED: reader requires offset 9 but first retained offset is 10"
    );
    assert_eq!(
        validate_retained_range(Offset(0), Offset(10), Offset(20)),
        Err(ReplicationRefusal::HistoryExpired {
            required: 0,
            first: 10
        }),
        "a new reader cannot start on a stream that lost its first operations"
    );

    let ahead = validate_retained_range(Offset(22), Offset(10), Offset(20)).unwrap_err();
    assert_eq!(
        ahead.to_string(),
        "REPLICATION_STREAM_BEHIND: reader requires offset 22 ahead of stream head 20"
    );

    // Empty stream: only a reader that has applied nothing may start.
    assert_eq!(validate_empty_stream(Offset(0)), Ok(()));
    let empty = validate_empty_stream(Offset(1)).unwrap_err();
    assert_eq!(
        empty.to_string(),
        "REPLICATION_STREAM_BEHIND: reader requires offset 1 but the stream is empty"
    );

    // Deliveries: anything but the expected offset is refused.
    assert_eq!(validate_delivery_offset(Offset(10), Offset(10)), Ok(()));
    let skipped = validate_delivery_offset(Offset(10), Offset(11)).unwrap_err();
    assert_eq!(
        skipped.to_string(),
        "REPLICATION_HISTORY_EXPIRED: reader requires offset 10 but broker delivered 11"
    );
    assert!(
        validate_delivery_offset(Offset(10), Offset(9)).is_err(),
        "a delivery behind the expected offset is a gap too"
    );
}

#[test]
fn readiness_waits_for_initial_observed_stream_head() {
    let status =
        ReplicationStatus::new(Offset(0), Duration::from_secs(300), Duration::from_secs(60));
    status.require_initial_catchup(Offset(2));
    assert!(!status.report().ready);
    status.record_applied(Offset(0));
    assert!(!status.report().ready);
    status.record_applied(Offset(1));
    assert!(!status.report().ready);
    status.record_applied(Offset(2));
    assert!(status.report().ready);

    // A restarted reader that already applied the head is ready immediately.
    let restarted =
        ReplicationStatus::new(Offset(5), Duration::from_secs(300), Duration::from_secs(60));
    restarted.require_initial_catchup(Offset(5));
    assert!(restarted.report().ready);
}

/// The stop path stores an offset only together with the index that contains it: a reader
/// loaded from the same directory serves every document the stopped reader had applied.
#[tokio::test(flavor = "multi_thread")]
async fn stop_commits_every_applied_operation_before_storing_its_offset() -> Result<()> {
    init_log();

    let mut config = create_oramacore_config();
    config.writer_side.master_api_key = TestContext::generate_api_key();
    // Only the stop path commits, so nothing else persists the operations it must cover.
    config
        .reader_side
        .config
        .collection_commit
        .operation_threshold = u64::MAX;
    let test_context = TestContext::new_with_config(config).await;
    let collection_client = test_context.create_collection().await?;
    let index_client = collection_client.create_index().await?;
    test_context.commit_all().await?;

    // Applied by the reader but not committed before `stop`.
    let documents: DocumentList = json!((0..50)
        .map(|i| json!({ "id": i.to_string(), "text": "replicated" }))
        .collect::<Vec<_>>())
    .try_into()?;
    index_client.insert_documents(documents).await?;

    test_context.reader.stop().await?;
    assert_eq!(document_count(&test_context, &collection_client).await?, 50);

    let (_writer, reloaded) = build_orama(test_context.config.clone()).await?;
    let reloaded = TestContext {
        config: test_context.config.clone(),
        reader: reloaded.context("Reader side is enabled")?,
        writer: test_context.writer.clone(),
        master_api_key: test_context.master_api_key,
    };
    assert_eq!(
        document_count(&reloaded, &collection_client).await?,
        50,
        "the stored offset covers operations missing from the committed index"
    );

    Ok(())
}

async fn document_count(
    context: &TestContext,
    collection: &TestCollectionClient<'_>,
) -> Result<usize> {
    let stats = context
        .reader
        .collection_stats(
            &collection.read_api_key,
            collection.collection_id,
            CollectionStatsRequest { with_keys: false },
        )
        .await?;
    Ok(stats.document_count)
}
