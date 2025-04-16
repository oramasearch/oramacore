use std::collections::HashMap;

use anyhow::Result;
use serde_json::json;

use crate::{
    tests::utils::{create, create_collection, create_oramacore_config, insert_docs},
    types::{ApiKey, CollectionId},
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_string_facets() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId::from("test-collection".to_string());
    create_collection(write_side.clone(), collection_id).await?;
    insert_docs(
        write_side.clone(),
        ApiKey::try_from("my-write-api-key").unwrap(),
        collection_id,
        vec![
            // match "my-section-1"
            json!({
                "id": "1",
                "name": "John Doe",
                "section": "my-section-1",
                "b": true,
            }),
            // match "my-section-2"
            json!({
                "id": "2",
                "name": "Jane Doe",
                "section": "my-section-2",
                "b": true,
            }),
            // match "my-section-2"
            json!({
                "id": "3",
                "name": "Doh",
                "section": "my-section-2",
                "b": true,
            }),
            // not match for term
            json!({
                "id": "4",
                "name": "Foo bar",
                "section": "my-section-2",
                "b": true,
            }),
            // not match for filter
            json!({
                "id": "5",
                "name": "Jane Doe",
                "section": "my-section-2",
                "b": false,
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey::try_from("my-read-api-key").unwrap(),
            collection_id,
            json!({
                "term": "d",
                "where": {
                    "b": true,
                },
                "facets": {
                    "section": {},
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(output.count, 3);
    let facets = output.facets.unwrap();

    assert_eq!(facets.len(), 1);
    assert_eq!(facets["section"].count, 2);
    assert_eq!(
        facets["section"].values,
        HashMap::from([
            ("my-section-1".to_string(), 1),
            ("my-section-2".to_string(), 2),
        ])
    );

    Ok(())
}
