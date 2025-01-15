mod collection;
mod collections;

mod insert;
mod search;

pub use collection::{CollectionReader, CommitConfig};
pub use collections::{CollectionsReader, IndexesConfig};

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use anyhow::Result;

    use serde_json::json;

    use crate::{
        collection_manager::{
            dto::{FieldId, LanguageDTO, TypedField},
            sides::write::{
                CollectionWriteOperation, DocumentFieldIndexOperation, GenericWriteOperation, Term,
                TermStringField, WriteOperation,
            },
        },
        embeddings::{EmbeddingConfig, EmbeddingService},
        nlp::NLPService,
        test_utils::generate_new_path,
        types::{CollectionId, DocumentId},
    };

    use super::*;

    #[tokio::test]
    async fn test_side_read_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsReader>();
        assert_sync_send::<CollectionReader>();
    }

    #[tokio::test]
    async fn test_side_read_commit_and_load() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let data_dir = generate_new_path();
        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            preload: vec![],
            grpc: None,
            hugging_face: None,
            fastembed: None,
            models: HashMap::new(),
        })
        .await?;
        let embedding_service = Arc::new(embedding_service);
        let collection_id = CollectionId("my-collection-name".to_string());

        {
            let collections = CollectionsReader::try_new(
                embedding_service.clone(),
                Arc::new(NLPService::new()),
                IndexesConfig {
                    data_dir: data_dir.join("indexes"),
                },
            )?;

            collections
                .update(WriteOperation::Generic(
                    GenericWriteOperation::CreateCollection {
                        id: collection_id.clone(),
                    },
                ))
                .await?;

            collections
                .update(WriteOperation::Collection(
                    collection_id.clone(),
                    CollectionWriteOperation::CreateField {
                        field_id: FieldId(0),
                        field_name: "title".to_string(),
                        field: TypedField::Text(LanguageDTO::English),
                    },
                ))
                .await?;

            collections
                .update(WriteOperation::Collection(
                    collection_id.clone(),
                    CollectionWriteOperation::InsertDocument {
                        doc_id: DocumentId(0),
                        doc: json!({
                            "id": "my-id",
                            "title": "hello world",
                        })
                        .try_into()?,
                    },
                ))
                .await?;

            collections
                .update(WriteOperation::Collection(
                    collection_id.clone(),
                    CollectionWriteOperation::Index(
                        DocumentId(0),
                        FieldId(0),
                        DocumentFieldIndexOperation::IndexString {
                            field_length: 2,
                            terms: HashMap::from_iter([
                                (
                                    Term("hello".to_string()),
                                    TermStringField { positions: vec![0] },
                                ),
                                (
                                    Term("world".to_string()),
                                    TermStringField { positions: vec![1] },
                                ),
                            ]),
                        },
                    ),
                ))
                .await?;

            collections.commit().await?;
        }

        let mut collections = CollectionsReader::try_new(
            embedding_service.clone(),
            Arc::new(NLPService::new()),
            IndexesConfig {
                data_dir: data_dir.join("indexes"),
            },
        )?;

        collections.load().await?;

        let reader = collections
            .get_collection(collection_id.clone())
            .await
            .expect("collection not found");

        let result = reader
            .search(
                json!({
                    "term": "hello",
                })
                .try_into()?,
            )
            .await?;

        assert_eq!(result.count, 1);
        assert_eq!(result.hits[0].id, "my-id".to_string());

        Ok(())
    }
}
