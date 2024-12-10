use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use rustorama::{build_orama, collection_manager::sides::{read::{CollectionsReader, DataConfig}, write::CollectionsWriter}, embeddings::{EmbeddingConfig, EmbeddingPreload}, ReadSideConfig, WriteSideConfig};
use tempdir::TempDir;



fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    tmp_dir.path().to_path_buf()
}

pub async fn start_all() -> Result<(
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
