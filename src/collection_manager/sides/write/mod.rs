mod collection;
mod collections;
mod embedding;
mod fields;
mod operation;

use std::{sync::Arc, time::Duration};

use anyhow::Result;
pub use collections::{CollectionsWriter, CollectionsWriterConfig};
use embedding::{start_calculate_embedding_loop, EmbeddingCalculationRequest};
pub use operation::*;

#[cfg(any(test, feature = "benchmarking"))]
pub use fields::*;
use tokio::sync::broadcast::Sender;

use crate::embeddings::EmbeddingService;

pub struct WriteSide {
    collections: CollectionsWriter,
}

impl WriteSide {
    pub fn new(
        sender: Sender<WriteOperation>,
        config: CollectionsWriterConfig,
        embedding_service: Arc<EmbeddingService>,
    ) -> WriteSide {
        let (sx, rx) = tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(1);

        start_calculate_embedding_loop(embedding_service.clone(), Duration::from_secs(1), rx);

        WriteSide {
            collections: CollectionsWriter::new(sender, config, sx),
        }
    }

    pub async fn load(&mut self) -> Result<()> {
        self.collections.load().await
    }

    pub async fn commit(&self) -> Result<()> {
        self.collections.commit().await
    }

    pub fn collections(&self) -> &CollectionsWriter {
        &self.collections
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Context, Result};
    use serde_json::json;

    use crate::{
        collection_manager::dto::CreateCollectionOptionDTO, test_utils::generate_new_path,
        types::CollectionId,
    };

    use super::*;

    #[tokio::test]
    async fn test_side_writer_serialize() -> Result<()> {
        let config = CollectionsWriterConfig {
            data_dir: generate_new_path(),
        };

        let (sx, _) = tokio::sync::mpsc::channel(1_0000);

        let collection = {
            let (sender, receiver) = tokio::sync::broadcast::channel(1_0000);

            let writer = CollectionsWriter::new(sender, config.clone(), sx);

            let collection_id = CollectionId("test-collection".to_string());
            writer
                .create_collection(CreateCollectionOptionDTO {
                    id: collection_id.0.clone(),
                    description: None,
                    language: None,
                    typed_fields: Default::default(),
                })
                .await?;

            writer
                .write(
                    collection_id,
                    vec![
                        json!({
                            "name": "John Doe",
                        }),
                        json!({
                            "name": "Jane Doe",
                        }),
                    ]
                    .try_into()?,
                )
                .await?;

            let collections = writer.list().await;

            writer.commit().await?;

            drop(receiver);

            collections
        };

        let after = {
            let (sender, receiver) = tokio::sync::broadcast::channel(1_0000);
            let (sx, _) = tokio::sync::mpsc::channel(1_0000);
            let mut writer = CollectionsWriter::new(sender, config, sx);

            writer
                .load()
                .await
                .context("Cannot load collections writer")?;

            let collections = writer.list().await;

            drop(receiver);

            collections
        };

        assert_eq!(collection, after);

        Ok(())
    }
}
