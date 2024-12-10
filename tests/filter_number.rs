use anyhow::Result;
use rustorama::collection_manager::sides::read::{CollectionsReader, DataConfig};
use rustorama::collection_manager::sides::write::CollectionsWriter;
use rustorama::{build_orama, ReadSideConfig, WriteSideConfig};
use serde_json::json;
use tokio::time::sleep;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tempdir::TempDir;

use rustorama::embeddings::{EmbeddingConfig, EmbeddingPreload};

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    tmp_dir.path().to_path_buf()
}

async fn start_all() -> Result<(
    Arc<CollectionsWriter>,
    Arc<CollectionsReader>,
    tokio::task::JoinHandle<()>,
)> {
    let (collections_writer, collections_reader, mut receiver) = build_orama(
        EmbeddingConfig {
            cache_path: std::env::temp_dir().to_str().unwrap().to_string(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        },
        WriteSideConfig {
            output: rustorama::SideChannelType::InMemory,
        },
        ReadSideConfig {
            input: rustorama::SideChannelType::InMemory,
            data: DataConfig {
                data_dir: generate_new_path(),
                max_size_per_chunk: 2048,
            },
        },
    )
    .await?;

    let collections_reader_inner = collections_reader.clone().unwrap();
    let handler = tokio::spawn(async move {
        while let Ok(op) = receiver.recv().await {
            collections_reader_inner.update(op).await.expect("OUCH!");
        }
    });

    Ok((collections_writer.unwrap(), collections_reader.unwrap(), handler))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_number_filters() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let (writer, reader, _) = start_all().await?;

    let collection_id = writer.create_collection(json!({
        "id": "test",
    }).try_into().unwrap()).await?;

    writer.write(collection_id.clone(), json!([
        {
            "id": "doc1",
            "number": 1
        },
        {
            "id": "doc2",
            "number": 2
        },
        {
            "id": "doc3",
            "number": 3
        },
        {
            "id": "doc4",
            "number": 4
        },
        {
            "id": "doc5",
            "number": 5
        },
    ]).try_into().unwrap()).await?;

    loop {
        sleep(Duration::from_millis(100)).await;
        let collection = match reader.get_collection(collection_id.clone()).await {
            Some(collection) => collection,
            None => continue,
        };
        let total = match collection.get_total_documents().await {
            Ok(total) => total,
            Err(_) => continue,
        };
        if total == 5 {
            break;
        }
    }

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection.search(json!({
        "term": "doc",
        "where": {
            "number": {
                "gt": 2
            }
        }
    }).try_into().unwrap()).await?;

    assert_eq!(output.hits.len(), 3);
    assert_eq!(output.count, 3);

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection.search(json!({
        "term": "doc",
        "where": {
            "number": {
                "gte": 2
            }
        }
    }).try_into().unwrap()).await?;

    assert_eq!(output.hits.len(), 4);
    assert_eq!(output.count, 4);

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection.search(json!({
        "term": "doc",
        "where": {
            "number": {
                "lt": 2
            }
        }
    }).try_into().unwrap()).await?;

    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.count, 1);


    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection.search(json!({
        "term": "doc",
        "where": {
            "number": {
                "lte": 2
            }
        }
    }).try_into().unwrap()).await?;

    assert_eq!(output.hits.len(), 2);
    assert_eq!(output.count, 2);

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection.search(json!({
        "term": "doc",
        "where": {
            "number": {
                "eq": 2
            }
        }
    }).try_into().unwrap()).await?;

    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.count, 1);

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection.search(json!({
        "term": "doc",
        "where": {
            "number": {
                "between": [2, 4],
            }
        }
    }).try_into().unwrap()).await?;

    assert_eq!(output.hits.len(), 3);
    assert_eq!(output.count, 3);

    Ok(())
}
