mod collection;
mod collections;

mod commit;
mod insert;
mod search;

pub use collection::CollectionReader;
pub use collections::{CollectionsReader, IndexesConfig};
pub use commit::CommitConfig;

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf, sync::Arc};

    use anyhow::Result;
    use deno_core::v8::json;
    use serde_json::json;

    use crate::{
        collection_manager::{
            dto::{FieldId, LanguageDTO, TypedField},
            sides::{
                document_storage::InMemoryDocumentStorage,
                write::{
                    CollectionWriteOperation, DocumentFieldIndexOperation, GenericWriteOperation, Term, TermStringField, WriteOperation
                },
            },
            CollectionId,
        },
        document_storage::DocumentId,
        embeddings::{EmbeddingConfig, EmbeddingPreload, EmbeddingService},
        js,
        test_utils::generate_new_path,
        types::Document,
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
        let collections = CollectionsReader::new(
            Arc::new(embedding_service),
            Arc::new(InMemoryDocumentStorage::new()),
            IndexesConfig {
                data_dir: data_dir.clone(),
            },
        );

        let collection_id = CollectionId("my-collection-name".to_string());
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
                            (Term("hello".to_string()), TermStringField { positions: vec![0] }),
                            (Term("world".to_string()), TermStringField { positions: vec![1] }),
                        ]),
                    },
                ),
            ))
            .await?;

        println!("data_dir: {:?}", data_dir);

        collections.commit().await?;

        Ok(())
    }
}
