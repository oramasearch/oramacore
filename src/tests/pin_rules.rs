use std::convert::TryInto;
use std::time::Duration;
use serde_json::json;
use tokio::time::sleep;
use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_pin_rules_after_insert() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0_u8..10_u8)
        .map(|i| {
            json!({
                "id": format!("{}", i),
                "c": format!("c-{}", i),
            })
        })
        .collect();
    index_client
        .insert_documents(
            docs.try_into().unwrap(),
        )
        .await
        .unwrap();

    index_client.insert_pin_rules(json!({
          "id": "rule-1",
          "conditions": [
            {
              "pattern": "c",
              "anchoring": "is"
            }
          ],
          "consequence": {
            "promote": [
              {
                "doc_id": "5",
                "position": 1
              },
              {
                "doc_id": "7",
                "position": 2
              }
            ]
          }
        }).try_into().unwrap()).await.unwrap();

    sleep(Duration::from_secs(1)).await;

    let result = collection_client.search(json!({
        "term": "c"
    }).try_into().unwrap()).await.unwrap();

    println!("result: {:#?}", result);

    assert_eq!(result.hits[0].id, "5");


    drop(test_context);
}
