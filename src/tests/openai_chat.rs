use crate::tests::utils::{init_log, TestContext};

// @todo: implement test
#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_module_exists() {
    init_log();

    let test_context = TestContext::new().await;
    let _collection_client = test_context.create_collection().await.unwrap();

    assert!(true);
}