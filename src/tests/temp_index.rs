use fake::{Fake, Faker};

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::CreateIndexRequest;
use crate::types::IndexId;

#[tokio::test(flavor = "multi_thread")]
async fn test_temp_index_double_creation() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let copy_from = index_client.index_id;

    let index_id = IndexId::try_new(Faker.fake::<String>()).unwrap();
    collection_client
        .writer
        .create_temp_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            copy_from,
            CreateIndexRequest {
                index_id,
                embedding: None,
                type_strategy: Default::default(),
                datasource: None,
            },
        )
        .await
        .unwrap();

    let res = collection_client
        .writer
        .create_temp_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            copy_from,
            CreateIndexRequest {
                index_id,
                embedding: None,
                type_strategy: Default::default(),
                datasource: None,
            },
        )
        .await;

    println!("Response: {res:?}");

    drop(test_context);
}
