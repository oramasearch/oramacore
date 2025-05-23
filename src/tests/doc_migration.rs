use anyhow::Result;

use crate::{
    tests::utils::{create_oramacore_config, init_log, TestContext},
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
