#![cfg(all(feature = "reader", feature = "writer"))]

use anyhow::Result;
use http::uri::Scheme;
use oramacore::ai::{AIServiceConfig, OramaModel};
use oramacore::collection_manager::dto::{ApiKey, CreateCollection, SearchParams};
use oramacore::collection_manager::sides::{
    CollectionsWriterConfig, OramaModelSerializable, ReadSideConfig, WriteSide, WriteSideConfig,
};
use oramacore::collection_manager::sides::{IndexesConfig, ReadSide};
use oramacore::test_utils::create_grpc_server;
use oramacore::types::{CollectionId, DocumentList};
use oramacore::{build_orama, OramacoreConfig};
use redact::Secret;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempdir::TempDir;

use oramacore::web_server::HttpConfig;

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    tmp_dir.path().to_path_buf()
}

async fn start_server() -> Result<(Arc<WriteSide>, Arc<ReadSide>)> {
    let address = create_grpc_server().await.unwrap();

    let (collections_writer, collections_reader) = build_orama(OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        ai_server: AIServiceConfig {
            scheme: Scheme::HTTP,
            host: address.ip(),
            port: address.port(),
            api_key: None,
            max_connections: 1,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey(Secret::new("my-master-api-key".to_string())),
            output: oramacore::collection_manager::sides::OutputSideChannelType::InMemory {
                capacity: 100,
            },
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: OramaModelSerializable(OramaModel::BgeSmall),
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
        reader_side: ReadSideConfig {
            input: oramacore::collection_manager::sides::InputSideChannelType::InMemory {
                capacity: 100,
            },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                insert_batch_commit_size: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
    })
    .await
    .unwrap();

    Ok((collections_writer.unwrap(), collections_reader.unwrap()))
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
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            CreateCollection {
                id: collection_id.clone(),
                description: None,
                language: None,
                embeddings: None,
                read_api_key: ApiKey(Secret::new("my-read-api-key".to_string())),
                write_api_key: ApiKey(Secret::new("my-write-api-key".to_string())),
            },
        )
        .await
        .unwrap();

    writer
        .write(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id.clone(),
            generate_test_data(),
        )
        .await
        .unwrap();

    reader.commit().await.unwrap();

    let param: SearchParams = json!({
        "term": "game love",
    })
    .try_into()
    .unwrap();
    for _ in 0..10_000 {
        reader
            .search(
                ApiKey(Secret::new("my-read-api-key".to_string())),
                collection_id.clone(),
                param.clone(),
            )
            .await
            .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_without_webserver() {
    let _ = tracing_subscriber::fmt::try_init();

    run_tests().await;
}
