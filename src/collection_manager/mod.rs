use std::{
    collections::{hash_map::Entry, HashMap},
    ops::Deref,
    sync::{
        atomic::{AtomicU32, AtomicU64},
        Arc,
    },
};

use anyhow::{anyhow, Context, Result};
use collection::Collection;
use dto::{CollectionDTO, CreateCollectionOptionDTO, LanguageDTO};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{info, warn};

use crate::{document_storage::DocumentStorage, embeddings::EmbeddingService};

mod collection;
pub mod dto;

pub mod sides;

pub use self::collection::FieldId;

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionId(pub String);

pub struct CollectionsConfiguration {
    pub embedding_service: Arc<EmbeddingService>,
}

pub struct CollectionManager {
    collections: RwLock<HashMap<CollectionId, Collection>>,
    document_storage: Arc<DocumentStorage>,
    id_generator: Arc<AtomicU32>,
    embedding_service: Arc<EmbeddingService>,
}

impl CollectionManager {
    pub fn new(configuration: CollectionsConfiguration) -> Self {
        let id_generator = Arc::new(AtomicU32::new(0));
        CollectionManager {
            collections: Default::default(),
            document_storage: Arc::new(DocumentStorage::new(id_generator.clone())),
            id_generator,
            embedding_service: configuration.embedding_service,
        }
    }

    pub async fn create_collection<S: TryInto<CreateCollectionOptionDTO>>(
        &self,
        collection_option: S,
    ) -> Result<CollectionId>
    where
        anyhow::Error: From<S::Error>,
        S::Error: std::fmt::Display,
    {
        let collection_option = collection_option
            .try_into()
            .map_err(|e| anyhow!("Cannot convert collection_option: {}", e))?;

        info!("Creating collection with id {:?}", collection_option);

        let id = CollectionId(collection_option.id);

        // TODO: fix error handling
        let collection = Collection::try_new(
            id.clone(),
            collection_option.description,
            collection_option
                .language
                .unwrap_or(LanguageDTO::English)
                .into(),
            self.document_storage.clone(),
            collection_option.typed_fields,
            Arc::new(AtomicU32::new(0)),
            self.embedding_service.clone(),
        )
        .await
        .with_context(|| format!("Cannot create collection with id {}", id.0))?;

        let mut collections = self.collections.write().await;
        let entry = collections.entry(id.clone());
        match entry {
            Entry::Occupied(_) => {
                warn!("Collection with id {} already exists", id.0);
                return Err(anyhow!("Id already exists"));
            }
            Entry::Vacant(entry) => entry.insert(collection),
        };

        info!("Collection with id {} created", id.0);

        Ok(id)
    }

    pub async fn list(&self) -> Vec<CollectionDTO> {
        let collections = self.collections.read().await;
        collections.iter().map(|(_, col)| col.as_dto()).collect()
    }

    pub async fn get<'guard, 's>(&'s self, id: CollectionId) -> Option<CollectionReadLock<'guard>>
    where
        's: 'guard,
    {
        CollectionReadLock::try_new(self.collections.read().await, id)
    }
}

pub struct CollectionReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, Collection>>,
    id: CollectionId,
}

impl<'guard> CollectionReadLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<CollectionId, Collection>>,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionReadLock { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for CollectionReadLock<'_> {
    type Target = Collection;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use serde_json::json;

    use crate::{
        embeddings::{EmbeddingConfig, EmbeddingPreload, EmbeddingService},
        indexes::number::{Number, NumberFilter},
    };

    use super::{
        dto::{
            CreateCollectionOptionDTO, FacetDefinition, Filter, FulltextMode, Limit,
            NumberFacetDefinition, NumberFacetDefinitionRange, SearchMode, SearchParams,
        },
        CollectionsConfiguration,
    };

    use super::CollectionManager;

    async fn create_manager() -> CollectionManager {
        let _ = tracing_subscriber::fmt::try_init();

        let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
            preload: EmbeddingPreload::Bool(false),
            cache_path: std::env::temp_dir().to_str().unwrap().to_string(),
            hugging_face: None,
        })
        .await
        .unwrap();
        CollectionManager::new(CollectionsConfiguration {
            embedding_service: Arc::new(embedding_service),
        })
    }

    #[tokio::test]
    async fn create_collection() {
        let manager = create_manager().await;

        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        assert_eq!(collection_id.0, collection_id_str);
    }

    #[tokio::test]
    async fn test_collection_id_already_exists() {
        let manager = create_manager().await;

        let collection_id = "my-test-collection".to_string();

        manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id.clone(),
                description: None,
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let output = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id.clone(),
                description: None,
                language: None,
                typed_fields: Default::default(),
            })
            .await;

        assert_eq!(format!("{}", output.err().unwrap()), "Id already exists");
    }

    #[tokio::test]
    async fn test_get_collections() {
        let manager = create_manager().await;

        let collection_ids: Vec<_> = (0..3)
            .map(|i| format!("my-test-collection-{}", i))
            .collect();
        for id in &collection_ids {
            manager
                .create_collection(CreateCollectionOptionDTO {
                    id: id.clone(),
                    description: None,
                    language: None,
                    typed_fields: Default::default(),
                })
                .await
                .expect("insertion should be successful");
        }

        let collections = manager.list().await;

        assert_eq!(collections.len(), 3);

        let ids: HashSet<_> = collections.into_iter().map(|c| c.id).collect();
        assert_eq!(ids, collection_ids.into_iter().collect());
    }

    #[tokio::test]
    async fn test_insert_documents_into_collection() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id).await.unwrap();
        let output = collection
            .insert_batch(
                vec![
                    json!({
                        "id": "1",
                        "name": "Tommaso",
                        "surname": "Allevi",
                    }),
                    json!({
                        "id": "2",
                        "name": "Michele",
                        "surname": "Riva",
                    }),
                ]
                .try_into()
                .unwrap(),
            )
            .await;

        assert!(matches!(output, Ok(())));
    }

    #[tokio::test]
    async fn test_search_documents() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();
        let output = collection
            .insert_batch(
                vec![
                    json!({
                        "id": "1",
                        "name": "Tommaso",
                        "surname": "Allevi",
                    }),
                    json!({
                        "id": "2",
                        "name": "Michele",
                        "surname": "Riva",
                    }),
                ]
                .try_into()
                .unwrap(),
            )
            .await;

        assert!(matches!(output, Ok(())));

        let collection = manager.get(collection_id).await.unwrap();

        let search_params = SearchParams {
            mode: SearchMode::FullText(FulltextMode {
                term: "Tommaso".to_string(),
            }),
            limit: Limit(10),
            boost: Default::default(),
            properties: Default::default(),
            where_filter: Default::default(),
            facets: Default::default(),
        };
        let output = collection.search(search_params).await;

        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.count, 1);
        assert_eq!(output.hits.len(), 1);
        assert_eq!(output.hits[0].id, "1");
    }

    #[tokio::test]
    async fn test_search_documents_order() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();
        let output = collection
            .insert_batch(
                vec![
                    json!({
                        "id": "1",
                        "text": "This is a long text with a lot of words",
                    }),
                    json!({
                        "id": "2",
                        "text": "This is a smaller text",
                    }),
                ]
                .try_into()
                .unwrap(),
            )
            .await;

        assert!(matches!(output, Ok(())));

        let search_params = SearchParams {
            mode: SearchMode::FullText(FulltextMode {
                term: "text".to_string(),
            }),
            limit: Limit(10),
            boost: Default::default(),
            properties: Default::default(),
            where_filter: Default::default(),
            facets: Default::default(),
        };
        let output = collection.search(search_params).await;

        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.count, 2);
        assert_eq!(output.hits.len(), 2);
        assert_eq!(output.hits[0].id, "2");
        assert_eq!(output.hits[1].id, "1");
    }

    #[tokio::test]
    async fn test_search_documents_limit() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();
        let output = collection
            .insert_batch(
                (0..100)
                    .map(|i| {
                        json!({
                            "id": i.to_string(),
                            "text": "text ".repeat(i + 1),
                        })
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
            .await;

        assert!(matches!(output, Ok(())));

        let search_params = SearchParams {
            mode: SearchMode::FullText(FulltextMode {
                term: "text".to_string(),
            }),
            limit: Limit(10),
            boost: Default::default(),
            properties: Default::default(),
            where_filter: Default::default(),
            facets: Default::default(),
        };
        let output = collection.search(search_params).await;

        assert!(output.is_ok());
        let output = output.unwrap();

        assert_eq!(output.count, 100);
        assert_eq!(output.hits.len(), 10);
        assert_eq!(output.hits[0].id, "99");
        assert_eq!(output.hits[1].id, "98");
        assert_eq!(output.hits[2].id, "97");
        assert_eq!(output.hits[3].id, "96");
        assert_eq!(output.hits[4].id, "95");
    }

    #[tokio::test]
    async fn test_filter_number() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();
        collection
            .insert_batch(
                (0..100)
                    .map(|i| {
                        json!({
                            "id": i.to_string(),
                            "text": "text ".repeat(i + 1),
                            "number": i,
                        })
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
            .await
            .unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "text".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: vec![(
                    "number".to_string(),
                    Filter::Number(NumberFilter::Equal(50.into())),
                )]
                .into_iter()
                .collect(),
                facets: Default::default(),
            })
            .await
            .unwrap();

        assert_eq!(output.count, 1);
        assert_eq!(output.hits.len(), 1);
        assert_eq!(output.hits[0].id, "50");
    }

    #[tokio::test]
    async fn test_facets_number() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id).await.unwrap();
        collection
            .insert_batch(
                (0..100)
                    .map(|i| {
                        json!({
                            "id": i.to_string(),
                            "text": "text ".repeat(i + 1),
                            "number": i,
                        })
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
            .await
            .unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "text".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: Default::default(),
                facets: HashMap::from_iter(vec![(
                    "number".to_string(),
                    FacetDefinition::Number(NumberFacetDefinition {
                        ranges: vec![
                            NumberFacetDefinitionRange {
                                from: Number::from(0),
                                to: Number::from(10),
                            },
                            NumberFacetDefinitionRange {
                                from: Number::from(0.5),
                                to: Number::from(10.5),
                            },
                            NumberFacetDefinitionRange {
                                from: Number::from(-10),
                                to: Number::from(10),
                            },
                            NumberFacetDefinitionRange {
                                from: Number::from(-10),
                                to: Number::from(-1),
                            },
                            NumberFacetDefinitionRange {
                                from: Number::from(1),
                                to: Number::from(100),
                            },
                            NumberFacetDefinitionRange {
                                from: Number::from(99),
                                to: Number::from(105),
                            },
                            NumberFacetDefinitionRange {
                                from: Number::from(102),
                                to: Number::from(105),
                            },
                        ],
                    }),
                )]),
            })
            .await
            .unwrap();

        let facets = output.facets.expect("Facet should be there");
        let number_facet = facets
            .get("number")
            .expect("Facet on field 'number' should be there");

        assert_eq!(number_facet.count, 7);
        assert_eq!(number_facet.values.len(), 7);

        assert_eq!(
            number_facet.values,
            HashMap::from_iter(vec![
                ("-10--1".to_string(), 0),
                ("-10-10".to_string(), 11),
                ("0-10".to_string(), 11),
                ("0.5-10.5".to_string(), 10),
                ("1-100".to_string(), 99),
                ("102-105".to_string(), 0),
                ("99-105".to_string(), 1),
            ])
        );
    }

    #[tokio::test]
    async fn test_filter_bool() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id).await.unwrap();

        collection
            .insert_batch(
                (0..100)
                    .map(|i| {
                        json!({
                            "id": i.to_string(),
                            "text": "text ".repeat(i + 1),
                            "bool": i % 2 == 0,
                        })
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
            .await
            .unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "text".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: vec![("bool".to_string(), Filter::Bool(true))]
                    .into_iter()
                    .collect(),
                facets: Default::default(),
            })
            .await
            .unwrap();

        assert_eq!(output.count, 50);
        assert_eq!(output.hits.len(), 10);
        for hit in output.hits.iter() {
            let id = hit.id.parse::<usize>().unwrap();
            assert_eq!(id % 2, 0);
        }
    }

    #[tokio::test]
    async fn test_facets_bool() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();

        collection
            .insert_batch(
                (0..100)
                    .map(|i| {
                        json!({
                            "id": i.to_string(),
                            "text": "text ".repeat(i + 1),
                            "bool": i % 2 == 0,
                        })
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
            .await
            .unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "text".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: Default::default(),
                facets: HashMap::from_iter(vec![("bool".to_string(), FacetDefinition::Bool)]),
            })
            .await
            .unwrap();

        let facets = output.facets.expect("Facet should be there");
        let bool_facet = facets
            .get("bool")
            .expect("Facet on field 'bool' should be there");

        assert_eq!(bool_facet.count, 2);
        assert_eq!(bool_facet.values.len(), 2);

        assert_eq!(
            bool_facet.values,
            HashMap::from_iter(vec![("true".to_string(), 50), ("false".to_string(), 50),])
        );
    }

    #[tokio::test]
    async fn test_facets_should_based_on_term() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();

        collection
            .insert_batch(
                vec![
                    json!({
                        "id": "1",
                        "text": "text",
                        "bool": true,
                        "number": 1,
                    }),
                    json!({
                        "id": "2",
                        "text": "text text",
                        "bool": false,
                        "number": 2,
                    }),
                    // This document doens't match the term
                    // so it should not be counted in the facets
                    json!({
                        "id": "3",
                        "text": "another",
                        "bool": true,
                        "number": 1,
                    }),
                ]
                .try_into()
                .unwrap(),
            )
            .await
            .unwrap();

        let output = collection
            .search(SearchParams {
                mode: SearchMode::FullText(FulltextMode {
                    term: "text".to_string(),
                }),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
                where_filter: Default::default(),
                facets: HashMap::from_iter(vec![
                    ("bool".to_string(), FacetDefinition::Bool),
                    (
                        "number".to_string(),
                        FacetDefinition::Number(NumberFacetDefinition {
                            ranges: vec![NumberFacetDefinitionRange {
                                from: Number::from(0),
                                to: Number::from(10),
                            }],
                        }),
                    ),
                ]),
            })
            .await
            .unwrap();

        let facets = output.facets.expect("Facet should be there");
        let bool_facet = facets
            .get("bool")
            .expect("Facet on field 'bool' should be there");

        assert_eq!(bool_facet.count, 2);
        assert_eq!(bool_facet.values.len(), 2);

        assert_eq!(
            bool_facet.values,
            HashMap::from_iter(vec![("true".to_string(), 1), ("false".to_string(), 1),])
        );

        let number_facet = facets
            .get("number")
            .expect("Facet on field 'number' should be there");

        assert_eq!(number_facet.count, 1);
        assert_eq!(number_facet.values.len(), 1);

        assert_eq!(
            number_facet.values,
            HashMap::from_iter(vec![("0-10".to_string(), 2),])
        );
    }

    #[tokio::test]
    async fn test_vector_search() {
        let manager = create_manager().await;
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(json!({
                "id": collection_id_str,
                "typed_fields": {
                    "vector": {
                        "type": "embedding",
                        "model_name": "gte-small",
                        "document_fields": ["text"],
                    }
                }
            }))
            .await
            .expect("insertion should be successful");

        let collection = manager.get(collection_id.clone()).await.unwrap();

        collection
            .insert_batch(
                vec![
                    json!({
                        "id": "1",
                        "text": "The cat is sleeping on the table.",
                    }),
                    json!({
                        "id": "2",
                        "text": "A cat rests peacefully on the sofa.",
                    }),
                    json!({
                        "id": "3",
                        "text": "The dog is barking loudly in the yard.",
                    }),
                ]
                .try_into()
                .unwrap(),
            )
            .await
            .unwrap();

        let output = collection
            .search(json!({
                "type": "vector",
                "term": "The feline is napping comfortably indoors.",
            }))
            .await
            .unwrap();

        // Due to the lack of a large enough dataset,
        // the search will not return 2 results as expected.
        // But it should return at least one result.
        // Anyway, the 3th document should not be returned.
        assert_ne!(output.count, 0);
        assert_ne!(output.hits.len(), 0);
        assert!(["1", "2"].contains(&output.hits[0].id.as_str()));
    }
}
