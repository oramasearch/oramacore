use anyhow::Result;
use serde_json::json;

use crate::{
    collection_manager::sides::read::IndexFieldStatsType,
    tests::utils::{create_oramacore_config, init_log, TestContext},
    types::{ApiKey, CollectionId, Number},
    OramacoreConfig,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_doc_migration() -> Result<()> {
    init_log();

    let p = std::env::current_dir()
        .unwrap()
        .join("src")
        .join("tests")
        .join("dump-test")
        .join("migration_docs")
        .join("writer");

    let _ = std::fs::remove_dir_all(p.join("documents").join("zebo"));

    let mut config: OramacoreConfig = create_oramacore_config();
    config.writer_side.config.data_dir = p;

    let test_context = TestContext::new_with_config(config.clone()).await;

    let colls = test_context.get_writer_collections().await;

    assert_eq!(colls.len(), 1);
    assert_eq!(colls[0].document_count, 5);
    assert_eq!(colls[0].indexes.len(), 1);
    assert_eq!(colls[0].indexes[0].document_count, 5);

    let test_context = test_context.reload().await;

    let colls = test_context.get_writer_collections().await;

    assert_eq!(colls.len(), 1);
    assert_eq!(colls[0].document_count, 5);
    assert_eq!(colls[0].indexes.len(), 1);
    assert_eq!(colls[0].indexes[0].document_count, 5);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_ordered_key_index() -> Result<()> {
    init_log();

    let writer_path = std::env::current_dir()
        .unwrap()
        .join("src")
        .join("tests")
        .join("dump-test")
        .join("before-remove-odered-key")
        .join(".data")
        .join("writer");
    let reader_path = std::env::current_dir()
        .unwrap()
        .join("src")
        .join("tests")
        .join("dump-test")
        .join("before-remove-odered-key")
        .join(".data")
        .join("reader");

    let new_cwd = std::env::current_dir()
        .unwrap()
        .join("src")
        .join("tests")
        .join("dump-test")
        .join("before-remove-odered-key");
    std::env::set_current_dir(new_cwd).unwrap();

    // Remove old files if they exist
    glob::glob("**/number_vec.bin").unwrap().for_each(|entry| {
        let entry = entry.unwrap();
        if entry.is_file() {
            std::fs::remove_file(entry).unwrap();
        }
    });
    glob::glob("**/date_vec.bin").unwrap().for_each(|entry| {
        let entry = entry.unwrap();
        if entry.is_file() {
            std::fs::remove_file(entry).unwrap();
        }
    });
    glob::glob("**/bool_map.bin").unwrap().for_each(|entry| {
        let entry = entry.unwrap();
        if entry.is_file() {
            std::fs::remove_file(entry).unwrap();
        }
    });

    let mut config: OramacoreConfig = create_oramacore_config();
    config.writer_side.config.data_dir = writer_path;
    config.reader_side.config.data_dir = reader_path;
    let test_context = TestContext::new_with_config(config.clone()).await;

    let colls = test_context.get_writer_collections().await;
    assert_eq!(colls.len(), 1);
    assert_eq!(colls[0].document_count, 1);

    let collection_client = test_context
        .get_test_collection_client(
            CollectionId::try_new("tommaso-1").unwrap(),
            ApiKey::try_new("write").unwrap(),
            ApiKey::try_new("read").unwrap(),
        )
        .unwrap();

    let mut stats = collection_client.reader_stats().await.unwrap();

    let index = stats.indexes_stats.remove(0);

    let bools_field = index
        .fields_stats
        .iter()
        .find(|f| {
            f.field_path == "bools" && matches!(f.stats, IndexFieldStatsType::CommittedBoolean(_))
        })
        .unwrap();
    let IndexFieldStatsType::CommittedBoolean(bools_field) = &bools_field.stats else {
        panic!("Expected CommittedBoolean");
    };
    assert_eq!(bools_field.false_count, 0);
    assert_eq!(bools_field.true_count, 1);

    let bool_field = index
        .fields_stats
        .iter()
        .find(|f| {
            f.field_path == "bool" && matches!(f.stats, IndexFieldStatsType::CommittedBoolean(_))
        })
        .unwrap();
    let IndexFieldStatsType::CommittedBoolean(bool_field) = &bool_field.stats else {
        panic!("Expected CommittedBoolean");
    };
    assert_eq!(bool_field.false_count, 0);
    assert_eq!(bool_field.true_count, 1);

    let res = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "bools": true
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.count, 1);

    let res = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "bool": true
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.count, 1);

    let numbers_field = index
        .fields_stats
        .iter()
        .find(|f| {
            f.field_path == "numbers" && matches!(f.stats, IndexFieldStatsType::CommittedNumber(_))
        })
        .unwrap();
    let IndexFieldStatsType::CommittedNumber(numbers_field) = &numbers_field.stats else {
        panic!("Expected CommittedNumber");
    };
    assert_eq!(numbers_field.min, Number::I32(42));
    assert_eq!(numbers_field.max, Number::I32(42));

    let number_field = index
        .fields_stats
        .iter()
        .find(|f| {
            f.field_path == "number" && matches!(f.stats, IndexFieldStatsType::CommittedNumber(_))
        })
        .unwrap();
    let IndexFieldStatsType::CommittedNumber(number_field) = &number_field.stats else {
        panic!("Expected CommittedNumber");
    };
    assert_eq!(number_field.min, Number::I32(42));
    assert_eq!(number_field.max, Number::I32(42));

    let date_field = index
        .fields_stats
        .iter()
        .find(|f| {
            f.field_path == "date" && matches!(f.stats, IndexFieldStatsType::CommittedDate(_))
        })
        .unwrap();
    let IndexFieldStatsType::CommittedDate(date_field) = &date_field.stats else {
        panic!("Expected CommittedNumber");
    };
    assert_eq!(
        date_field.min,
        Some("2025-06-06T08:57:14.543Z".to_string().try_into().unwrap())
    );
    assert_eq!(
        date_field.max,
        Some("2025-06-06T08:57:14.543Z".to_string().try_into().unwrap())
    );

    Ok(())
}
