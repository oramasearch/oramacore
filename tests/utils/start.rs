use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::Result;
use oramacore::{
    build_orama,
    collection_manager::sides::{
        read::{CollectionsReader, IndexesConfig},
        CollectionsWriterConfig, WriteSide,
    },
    embeddings::EmbeddingConfig,
    web_server::HttpConfig,
    OramacoreConfig, ReadSideConfig, WriteSideConfig,
};
use tempdir::TempDir;

fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    tmp_dir.path().to_path_buf()
}

pub async fn start_all() -> Result<(
    Arc<WriteSide>,
    Arc<CollectionsReader>,
    tokio::task::JoinHandle<()>,
)> {
    let (collections_writer, collections_reader, mut receiver) = build_orama(OramacoreConfig {
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
            fastembed: None,
            models: HashMap::new(),
        },
        writer_side: WriteSideConfig {
            output: oramacore::SideChannelType::InMemory,
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
            },
        },
        reader_side: ReadSideConfig {
            input: oramacore::SideChannelType::InMemory,
            config: IndexesConfig {
                data_dir: generate_new_path(),
            },
        },
    })
    .await?;

    let collections_reader_inner = collections_reader.clone().unwrap();
    let handler = tokio::spawn(async move {
        while let Ok(op) = receiver.recv().await {
            collections_reader_inner.update(op).await.expect("OUCH!");
        }
    });

    Ok((
        collections_writer.unwrap(),
        collections_reader.unwrap(),
        handler,
    ))
}
