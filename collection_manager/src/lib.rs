use std::sync::Arc;

use collection::Collection;
use dashmap::DashMap;
use document_storage::DocumentStorage;
use dto::{CollectionDTO, CreateCollectionOptionDTO, LanguageDTO};
use storage::Storage;
use thiserror::Error;
use types::CollectionId;

mod collection;
pub mod dto;

pub struct CollectionsConfiguration {
    pub storage: Arc<Storage>,
}

pub struct CollectionManager {
    collections: DashMap<CollectionId, Collection>,
    configuration: CollectionsConfiguration,
    document_storage: Arc<DocumentStorage>,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CreateCollectionError {
    #[error("Id already exists")]
    IdAlreadyExists,
}

impl CollectionManager {
    pub fn new(configuration: CollectionsConfiguration) -> Self {
        CollectionManager {
            collections: DashMap::new(),
            document_storage: Arc::new(DocumentStorage::new(configuration.storage.clone())),
            configuration,
        }
    }

    pub fn create_collection(
        &self,
        collection_option: CreateCollectionOptionDTO,
    ) -> Result<CollectionId, CreateCollectionError> {
        let id = CollectionId(collection_option.id);
        let collection = Collection::new(
            self.configuration.storage.clone(),
            id.clone(),
            collection_option.description,
            collection_option
                .language
                .unwrap_or(LanguageDTO::English)
                .into(),
            self.document_storage.clone(),
        );

        let entry = self.collections.entry(id.clone());
        match entry {
            dashmap::mapref::entry::Entry::Occupied(_) => {
                return Err(CreateCollectionError::IdAlreadyExists)
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => entry.insert(collection),
        };

        Ok(id)
    }

    pub fn list(&self) -> Vec<CollectionDTO> {
        self.collections
            .iter()
            .map(|entry| entry.value().as_dto())
            .collect()
    }

    pub fn get<R>(&self, id: CollectionId, f: impl FnOnce(&Collection) -> R) -> Option<R> {
        let output = self.collections.get(&id);
        output.map(|collection| f(collection.value()))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use rocksdb::OptimisticTransactionDB;
    use serde_json::json;
    use storage::Storage;
    use tempdir::TempDir;

    use crate::dto::{CreateCollectionOptionDTO, SearchParams};

    use super::CollectionManager;

    fn create_manager() -> CollectionManager {
        let tmp_dir = TempDir::new("string_index_test").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
        let db = OptimisticTransactionDB::open_default(tmp_dir).unwrap();
        let storage = Arc::new(Storage::new(db));

        CollectionManager::new(crate::CollectionsConfiguration { storage })
    }

    #[test]
    fn create_collection() {
        let manager = create_manager();

        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
            })
            .expect("insertion should be successful");

        assert_eq!(collection_id.0, collection_id_str);
    }

    #[test]
    fn test_collection_id_already_exists() {
        let manager = create_manager();

        let collection_id = "my-test-collection".to_string();

        manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id.clone(),
                description: None,
                language: None,
            })
            .expect("insertion should be successful");

        let output = manager.create_collection(CreateCollectionOptionDTO {
            id: collection_id.clone(),
            description: None,
            language: None,
        });

        assert_eq!(output, Err(super::CreateCollectionError::IdAlreadyExists));
    }

    #[test]
    fn test_get_collections() {
        let manager = create_manager();

        let collection_ids: Vec<_> = (0..3)
            .map(|i| format!("my-test-collection-{}", i))
            .collect();
        for id in &collection_ids {
            manager
                .create_collection(CreateCollectionOptionDTO {
                    id: id.clone(),
                    description: None,
                    language: None,
                })
                .expect("insertion should be successful");
        }

        let collections = manager.list();

        assert_eq!(collections.len(), 3);

        let ids: HashSet<_> = collections.into_iter().map(|c| c.id).collect();
        assert_eq!(ids, collection_ids.into_iter().collect());
    }

    #[test]
    fn test_insert_documents_into_collection() {
        let manager = create_manager();
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
            })
            .expect("insertion should be successful");

        let output = manager.get(collection_id, |collection| {
            collection.insert_batch(
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
        });

        assert!(matches!(output, Some(Ok(()))));
    }

    #[test]
    fn test_search_documents() {
        let manager = create_manager();
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
            })
            .expect("insertion should be successful");

        let output = manager.get(collection_id.clone(), |collection| {
            collection.insert_batch(
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
        });

        assert!(matches!(output, Some(Ok(()))));

        let output = manager.get(collection_id, |collection| {
            let search_params = SearchParams {
                term: "Tommaso".to_string(),
            };
            collection.search(search_params)
        });

        assert!(matches!(output, Some(Ok(_))));
        let output = output.unwrap().unwrap();

        println!("{:?}", output);
    }
}
