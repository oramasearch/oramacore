use anyhow::Result;
use rustorama::collection_manager::dto::{CreateCollectionOptionDTO, SearchParams};
use rustorama::collection_manager::sides::read::{CollectionsReader, IndexesConfig};
use rustorama::collection_manager::sides::{CollectionsWriterConfig, WriteSide};
use rustorama::embeddings::fe::{FastEmbedModelRepoConfig, FastEmbedRepoConfig};
use rustorama::types::{CollectionId, DocumentList};
use rustorama::{build_orama, ReadSideConfig, RustoramaConfig, WriteSideConfig};
use serde_json::json;
use std::collections::HashMap;
use std::env::temp_dir;
use std::path::PathBuf;
use std::sync::Arc;
use tempdir::TempDir;

use rustorama::embeddings::{EmbeddingConfig, ModelConfig};
use rustorama::web_server::HttpConfig;

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    tmp_dir.path().to_path_buf()
}

async fn start_server() -> Result<(Arc<WriteSide>, Arc<CollectionsReader>)> {
    let (collections_writer, collections_reader, mut receiver) = build_orama(RustoramaConfig {
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        embeddings: EmbeddingConfig {
            preload: vec![],
            grpc: None,
            hugging_face: None,
            fastembed: Some(FastEmbedRepoConfig {
                cache_dir: temp_dir(),
            }),
            models: HashMap::from_iter([(
                "gte-small".to_string(),
                ModelConfig::Fastembed(FastEmbedModelRepoConfig {
                    real_model_name: "Xenova/bge-small-en-v1.5".to_string(),
                    dimensions: 384,
                }),
            )]),
        },
        writer_side: WriteSideConfig {
            output: rustorama::SideChannelType::InMemory,
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
            },
        },
        reader_side: ReadSideConfig {
            input: rustorama::SideChannelType::InMemory,
            config: IndexesConfig {
                data_dir: generate_new_path(),
            },
        },
    })
    .await
    .unwrap();

    let collections_reader = collections_reader.unwrap();
    let collections_reader2 = collections_reader.clone();
    tokio::spawn(async move {
        while let Ok(op) = receiver.recv().await {
            let r = collections_reader2.update(op).await;
            if let Err(e) = r {
                println!("--------");
                eprintln!("Error: {:?}", e);
            }
        }
    });

    Ok((collections_writer.unwrap(), collections_reader))
}

fn generate_test_data() -> DocumentList {
    let path = "src/bin/imdb_top_1000_tv_series.json";
    let docs: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let docs: Vec<_> = docs
        .as_array()
        .unwrap()
        .iter()
        .map(|d| {
            json!({
                "field": d["plot"].as_str().unwrap(),
            })
        })
        .collect();

    docs.try_into().unwrap()
}

async fn run_tests() {
    let (writer, reader) = start_server().await.unwrap();

    let collection_id = CollectionId("collection-test".to_string());

    writer
        .collections()
        .create_collection(CreateCollectionOptionDTO {
            id: collection_id.0.clone(),
            description: None,
            language: None,
            typed_fields: Default::default(),
        })
        .await
        .unwrap();

    writer
        .collections()
        .write(collection_id.clone(), generate_test_data())
        .await
        .unwrap();

    // reader.commit().await.unwrap();

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let param: SearchParams = json!({
        "term": "game love",
    })
    .try_into()
    .unwrap();
    for _ in 0..10_000 {
        collection.search(param.clone()).await.unwrap();
        // println!("Result {:#?}", r)
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_without_webserver() {
    let _ = tracing_subscriber::fmt::try_init();

    run_tests().await;
}
