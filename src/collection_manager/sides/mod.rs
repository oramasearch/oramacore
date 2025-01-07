pub mod document_storage;
pub mod read;
pub mod write;

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::{atomic::AtomicU32, Arc},
    };

    use anyhow::Result;
    use document_storage::{DiskDocumentStorage, DocumentStorage};

    use read::{CollectionsReader, IndexesConfig};
    use serde_json::json;

    use write::CollectionsWriter;

    use crate::{
        collection_manager::{
            dto::{
                CreateCollectionOptionDTO, EmbeddingTypedField, Filter, FulltextMode, Limit,
                SearchMode, SearchParams, TypedField,
            },
            CollectionId,
        },
        embeddings::{
            EmbeddingConfig, EmbeddingPreload, EmbeddingService, OramaFastembedModel, OramaModel,
        },
        indexes::number::{Number, NumberFilter}, test_utils::generate_new_path,
    };

    use super::*;

    #[tokio::test]
    async fn test_sides_error_on_unknow_filter_field() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let (sender, mut rec) = tokio::sync::broadcast::channel(100);

        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            cache_path: std::env::temp_dir(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        })
        .await?;
        let embedding_service = Arc::new(embedding_service);
        let document_id_generator = Arc::new(AtomicU32::new(0));
        let writer =
            CollectionsWriter::new(document_id_generator, sender, embedding_service.clone());
        let document_storage: Arc<dyn DocumentStorage> = Arc::new(DiskDocumentStorage::try_new(generate_new_path())?);

        let reader = CollectionsReader::new(embedding_service, document_storage, IndexesConfig {});

        let create_collection_request: CreateCollectionOptionDTO = CreateCollectionOptionDTO {
            id: "my-collection".to_string(),
            description: None,
            language: None,
            typed_fields: HashMap::from_iter([(
                "embedding".to_string(),
                TypedField::Embedding(EmbeddingTypedField {
                    model_name: OramaModel::Fastembed(OramaFastembedModel::GTESmall),
                    document_fields: vec!["title".to_string()],
                }),
            )]),
        };
        writer.create_collection(create_collection_request).await?;

        while let Ok(op) = rec.try_recv() {
            reader.update(op).await?;
        }

        let collection_id = CollectionId("my-collection".to_string());
        let collection = reader
            .get_collection(collection_id.clone())
            .await
            .ok_or(anyhow::anyhow!("Collection should exists"))?;

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "title".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: HashMap::from_iter([(
                    "wow".to_string(),
                    Filter::Number(NumberFilter::Equal(Number::from(1))),
                )]),
                facets: Default::default(),
            })
            .await;

        let err = output.unwrap_err();
        assert_eq!(format!("{}", err), "Unknown field \"wow\"");

        Ok(())
    }

    #[tokio::test]
    async fn test_sides_error_on_wrong_type_field() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let (sender, mut rec) = tokio::sync::broadcast::channel(100);

        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            cache_path: std::env::temp_dir(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        })
        .await?;
        let embedding_service = Arc::new(embedding_service);
        let document_id_generator = Arc::new(AtomicU32::new(0));
        let writer =
            CollectionsWriter::new(document_id_generator, sender, embedding_service.clone());
        let document_storage: Arc<dyn DocumentStorage> = Arc::new(DiskDocumentStorage::try_new(generate_new_path())?);

        let reader = CollectionsReader::new(embedding_service, document_storage, IndexesConfig {});

        let create_collection_request: CreateCollectionOptionDTO = CreateCollectionOptionDTO {
            id: "my-collection".to_string(),
            description: None,
            language: None,
            typed_fields: HashMap::from_iter([(
                "embedding".to_string(),
                TypedField::Embedding(EmbeddingTypedField {
                    model_name: OramaModel::Fastembed(OramaFastembedModel::GTESmall),
                    document_fields: vec!["title".to_string()],
                }),
            )]),
        };
        writer.create_collection(create_collection_request).await?;

        let collection_id = CollectionId("my-collection".to_string());

        let docs = vec![
            json!({ "id": "1", "title": "title of doc 1", "content": "content of doc 1" }),
            json!({ "id": "2", "title": "title of doc 2", "content": "content of doc 2" }),
        ];

        writer
            .write(collection_id.clone(), docs.try_into()?)
            .await?;

        while let Ok(op) = rec.try_recv() {
            reader.update(op).await?;
        }

        let collection = reader.get_collection(collection_id.clone()).await.unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "title".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: HashMap::from_iter([(
                    "title".to_string(),
                    Filter::Number(NumberFilter::Equal(Number::from(1))),
                )]),
                facets: Default::default(),
            })
            .await;

        let err = output.unwrap_err();
        assert_eq!(
            format!("{}", err),
            "Filter on field \"title\"(Text(English)) not supported"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_sides_fulltext_search() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let (sender, mut rec) = tokio::sync::broadcast::channel(100);

        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            cache_path: std::env::temp_dir(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        })
        .await?;
        let embedding_service = Arc::new(embedding_service);
        let document_id_generator = Arc::new(AtomicU32::new(0));
        let writer =
            CollectionsWriter::new(document_id_generator, sender, embedding_service.clone());
        let document_storage: Arc<dyn DocumentStorage> = Arc::new(DiskDocumentStorage::try_new(generate_new_path())?);

        let reader = CollectionsReader::new(embedding_service, document_storage, IndexesConfig {});

        let create_collection_request: CreateCollectionOptionDTO = CreateCollectionOptionDTO {
            id: "my-collection".to_string(),
            description: None,
            language: None,
            typed_fields: HashMap::from_iter([(
                "embedding".to_string(),
                TypedField::Embedding(EmbeddingTypedField {
                    model_name: OramaModel::Fastembed(OramaFastembedModel::GTESmall),
                    document_fields: vec!["title".to_string()],
                }),
            )]),
        };
        writer.create_collection(create_collection_request).await?;

        let collection_id = CollectionId("my-collection".to_string());

        let docs = vec![
            json!({ "id": "1", "title": "title of doc 1", "content": "content of doc 1" }),
            json!({ "id": "2", "title": "title of doc 2", "content": "content of doc 2" }),
        ];

        writer
            .write(collection_id.clone(), docs.try_into()?)
            .await?;

        while let Ok(op) = rec.try_recv() {
            reader.update(op).await?;
        }

        let collection = reader.get_collection(collection_id.clone()).await.unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "title".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: Default::default(),
                facets: Default::default(),
            })
            .await?;

        assert_eq!(output.count, 2);
        assert_eq!(output.hits.len(), 2);

        let ids: HashSet<_> = output.hits.into_iter().map(|hit| hit.id).collect();

        assert_eq!(ids, ["1", "2"].iter().map(|i| i.to_string()).collect());

        Ok(())
    }
}
