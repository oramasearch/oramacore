use std::{collections::HashMap, sync::Arc};

use code_parser::CodeParser;
use collection::Collection;
use dashmap::DashMap;
use document_storage::DocumentStorage;
use dto::{CollectionDTO, CreateCollectionOptionDTO, LanguageDTO, TypedField};
use nlp::TextParser;
use storage::Storage;
use thiserror::Error;
use types::{CollectionId, StringParser};

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

        let typed_fields: HashMap<String, Box<dyn StringParser>> = collection_option
            .typed_fields
            .into_iter()
            .map(|(key, value)| {
                let parser: Box<dyn StringParser> = match value {
                    TypedField::Text(language) => {
                        Box::new(TextParser::from_language(language.into()))
                    }
                    TypedField::Code(language) => Box::new(CodeParser::from_language(language)),
                };
                (key, parser)
            })
            .collect();

        let collection = Collection::new(
            self.configuration.storage.clone(),
            id.clone(),
            collection_option.description,
            collection_option
                .language
                .unwrap_or(LanguageDTO::English)
                .into(),
            self.document_storage.clone(),
            typed_fields,
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
    use types::CodeLanguage;

    use crate::dto::{CreateCollectionOptionDTO, Limit, SearchParams, TypedField};

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
                typed_fields: Default::default(),
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
                typed_fields: Default::default(),
            })
            .expect("insertion should be successful");

        let output = manager.create_collection(CreateCollectionOptionDTO {
            id: collection_id.clone(),
            description: None,
            language: None,
            typed_fields: Default::default(),
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
                    typed_fields: Default::default(),
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
                typed_fields: Default::default(),
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
                typed_fields: Default::default(),
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
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
            };
            collection.search(search_params)
        });

        assert!(matches!(output, Some(Ok(_))));
        let output = output.unwrap().unwrap();

        assert_eq!(output.count, 1);
        assert_eq!(output.hits.len(), 1);
        assert_eq!(output.hits[0].id, "1");
    }

    #[test]
    fn test_search_documents_order() {
        let manager = create_manager();
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .expect("insertion should be successful");

        let output = manager.get(collection_id.clone(), |collection| {
            collection.insert_batch(
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
        });

        assert!(matches!(output, Some(Ok(()))));

        let output = manager.get(collection_id, |collection| {
            let search_params = SearchParams {
                term: "text".to_string(),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
            };
            collection.search(search_params)
        });

        assert!(matches!(output, Some(Ok(_))));
        let output = output.unwrap().unwrap();

        assert_eq!(output.count, 2);
        assert_eq!(output.hits.len(), 2);
        assert_eq!(output.hits[0].id, "2");
        assert_eq!(output.hits[1].id, "1");
    }

    #[test]
    fn test_search_documents_limit() {
        let manager = create_manager();
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: Default::default(),
            })
            .expect("insertion should be successful");

        let output = manager.get(collection_id.clone(), |collection| {
            collection.insert_batch(
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
        });

        assert!(matches!(output, Some(Ok(()))));

        let output = manager.get(collection_id, |collection| {
            let search_params = SearchParams {
                term: "text".to_string(),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
            };
            collection.search(search_params)
        });

        assert!(matches!(output, Some(Ok(_))));
        let output = output.unwrap().unwrap();

        assert_eq!(output.count, 100);
        assert_eq!(output.hits.len(), 10);
        assert_eq!(output.hits[0].id, "99");
        assert_eq!(output.hits[1].id, "98");
        assert_eq!(output.hits[2].id, "97");
        assert_eq!(output.hits[3].id, "96");
        assert_eq!(output.hits[4].id, "95");
    }

    #[test]
    fn test_foo() {
        let manager = create_manager();
        let collection_id_str = "my-test-collection".to_string();

        let collection_id = manager
            .create_collection(CreateCollectionOptionDTO {
                id: collection_id_str.clone(),
                description: Some("Collection of songs".to_string()),
                language: None,
                typed_fields: vec![("code".to_string(), TypedField::Code(CodeLanguage::TSX))]
                    .into_iter()
                    .collect(),
            })
            .expect("insertion should be successful");

        manager.get(collection_id.clone(), |collection| {
            collection.insert_batch(
                vec![
                    json!({
                        "id": "1",
                        "code": r#"
import { TableController, type SortingState } from '@tanstack/lit-table'
//...
@state()
private _sorting: SortingState = [
  {
    id: 'age', //you should get autocomplete for the `id` and `desc` properties
    desc: true,
  }
]
"#,
                    }),
                    json!({
                        "id": "2",
                        "code": r#"export type RowSelectionState = Record<string, boolean>

export type RowSelectionTableState = {
  rowSelection: RowSelectionState
}"#,
                    }),
                    json!({
                        "id": "3",
                        "code": r#"initialState?: Partial<
  VisibilityTableState &
  ColumnOrderTableState &
  ColumnPinningTableState &
  FiltersTableState &
  SortingTableState &
  ExpandedTableState &
  GroupingTableState &
  ColumnSizingTableState &
  PaginationTableState &
  RowSelectionTableState
>"#,
                    }),
                    json!({
                        "id": "4",
                        "code": r#"setColumnVisibility: (updater: Updater<VisibilityState>) => void"#,
                    })
                ]
                .try_into()
                .unwrap(),
            )
        });

        let output = manager
            .get(collection_id, |collection| {
                let search_params = SearchParams {
                    term: "SelectionTableState".to_string(),
                    limit: Limit(10),
                    boost: Default::default(),
                    properties: Default::default(),
                };
                collection.search(search_params)
            })
            .unwrap()
            .unwrap();

        println!("{:#?}", output);
    }
}
