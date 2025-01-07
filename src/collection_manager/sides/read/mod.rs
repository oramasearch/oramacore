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
            sides::{
                document_storage::InMemoryDocumentStorage,
                write::{
                    CollectionWriteOperation, DocumentFieldIndexOperation, GenericWriteOperation,
                    Term, TermStringField, WriteOperation,
                },
            },
            CollectionId,
        },
        document_storage::DocumentId,
        embeddings::{EmbeddingConfig, EmbeddingPreload, EmbeddingService},
        test_utils::generate_new_path,
    };

    use super::*;

    #[tokio::test]
    async fn test_reader_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsReader>();
        assert_sync_send::<CollectionReader>();
    }

    #[tokio::test]
    async fn test_foo() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let data_dir = generate_new_path();
        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            preload: EmbeddingPreload::Bool(false),
            cache_path: generate_new_path(),
            hugging_face: None,
        })
        .await?;
        let embedding_service = Arc::new(embedding_service);
        let collection_id = CollectionId("my-collection-name".to_string());

        {
            let collections = CollectionsReader::new(
                embedding_service.clone(),
                Arc::new(InMemoryDocumentStorage::new()),
                IndexesConfig {},
            );

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

            collections.commit(data_dir.clone()).await?;
        }

        let mut collections = CollectionsReader::new(
            embedding_service.clone(),
            Arc::new(InMemoryDocumentStorage::new()),
            IndexesConfig {},
        );

        collections.load_from_disk(data_dir).await?;

        println!("collections: {:#?}", collections);

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

        println!("result: {:#?}", result);

        Ok(())
    }
}
