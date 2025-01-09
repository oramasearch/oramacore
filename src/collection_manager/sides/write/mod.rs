mod collection;
mod collections;
mod fields;
mod operation;

pub use collections::{CollectionsWriter, CollectionsWriterConfig};
pub use operation::*;

#[cfg(any(test, feature = "benchmarking"))]
pub use fields::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use anyhow::{Context, Result};
    use serde_json::json;

    use crate::{
        collection_manager::dto::CreateCollectionOptionDTO,
        embeddings::{EmbeddingConfig, EmbeddingPreload, EmbeddingService},
        test_utils::generate_new_path,
        types::CollectionId,
    };

    use super::*;

    #[tokio::test]
    async fn test_side_writer_foo() -> Result<()> {
        let embedding_config = EmbeddingConfig {
            cache_path: generate_new_path(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        };
        let embedding_service = EmbeddingService::try_new(embedding_config)
            .await
            .with_context(|| "Failed to initialize the EmbeddingService")?;
        let embedding_service = Arc::new(embedding_service);

        let config = CollectionsWriterConfig {
            data_dir: generate_new_path(),
        };

        let collection = {
            let (sender, receiver) = tokio::sync::broadcast::channel(1_0000);

            let writer = CollectionsWriter::new(sender, embedding_service.clone(), config.clone());

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

            let mut writer = CollectionsWriter::new(sender, embedding_service.clone(), config);

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
