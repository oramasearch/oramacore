use serde_json::json;

use crate::{
    collection_manager::sides::triggers::Trigger,
    tests::utils::{init_log, TestContext},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_trigger() {
    init_log();

    let trigger_id = "my-trigger".to_string();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "Tommaso",
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let _ = collection_client
        .insert_trigger(
            Trigger {
                id: "1".to_string(),
                description: "My trigger".to_string(),
                name: "my-trigger-name".to_string(),
                response: "my-response".to_string(),
                segment_id: None,
            },
            Some(trigger_id.clone()),
        )
        .await
        .unwrap();
    // In http handler there's a mapping that is not present in the `insert_trigger` method
    // That is bad
    // TODO: fix the `insert_trigger` method to return the trigger id and uncomment the line below
    // let trigger_id = trigger.id;

    test_context.commit_all().await.unwrap();

    let trigger = collection_client
        .get_trigger(trigger_id.clone())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(trigger.name, "my-trigger-name");

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let trigger = collection_client
        .get_trigger(trigger_id.clone())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(trigger.name, "my-trigger-name");

    drop(test_context);
}
