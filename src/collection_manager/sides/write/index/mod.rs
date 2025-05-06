mod doc_id_storage;
mod fields;

use std::{
    path::PathBuf,
    sync::{atomic::AtomicU16, Arc},
};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use doc_id_storage::DocIdStorage;
use fields::{
    GenericField, IndexFilterField, IndexScoreField, SerializedFilterFieldType,
    SerializedScoreFieldType,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::{mpsc::Sender, RwLock};
use tracing::{info, instrument, trace};

use crate::{
    ai::{automatic_embeddings_selector::AutomaticEmbeddingsSelector, OramaModel},
    collection_manager::sides::{
        field_names_to_paths, hooks::HooksRuntime, CollectionWriteOperation,
        DocumentStorageWriteOperation, IndexWriteOperation, IndexWriteOperationFieldType,
        OperationSender, WriteOperation,
    },
    file_utils::BufferedFile,
    nlp::{locales::Locale, TextParser},
    types::{
        CollectionId, DescribeCollectionIndexResponse, Document, DocumentId, FieldId,
        IndexEmbeddingsCalculation, IndexFieldType, IndexId,
    },
};

use super::embedding::MultiEmbeddingCalculationRequest;
pub use fields::{FieldType, IndexedValue, OramaModelSerializable};

#[derive(Clone)]
pub enum EmbeddingStringCalculation {
    AllProperties,
    Properties(Box<[Box<[String]>]>),
    Hook(Arc<HooksRuntime>),
    Automatic,
}

pub struct Index {
    collection_id: CollectionId,
    index_id: IndexId,
    locale: Locale,
    text_parser: Arc<TextParser>,

    filter_fields: RwLock<Vec<IndexFilterField>>,
    score_fields: RwLock<Vec<IndexScoreField>>,

    doc_id_storage: RwLock<DocIdStorage>,

    field_id_generator: AtomicU16,

    op_sender: OperationSender,
    embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
    hook_runtime: Arc<HooksRuntime>,
    automatically_chosen_properties: Arc<AutomaticEmbeddingsSelector>,

    created_at: DateTime<Utc>,
}

impl Index {
    pub async fn empty(
        index_id: IndexId,
        collection_id: CollectionId,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        text_parser: Arc<TextParser>,
        op_sender: OperationSender,
        hook_runtime: Arc<HooksRuntime>,
        automatically_chosen_properties: Arc<AutomaticEmbeddingsSelector>,
    ) -> Result<Self> {
        let field_id = 0;
        let score_fields = vec![];

        let locale = text_parser.locale();

        Ok(Self {
            collection_id,
            index_id,
            locale,
            text_parser,

            doc_id_storage: RwLock::new(DocIdStorage::empty()),

            filter_fields: RwLock::new(Default::default()),
            score_fields: RwLock::new(score_fields),

            field_id_generator: AtomicU16::new(field_id),

            op_sender,
            hook_runtime,
            embedding_sender,
            automatically_chosen_properties,

            created_at: Utc::now(),
        })
    }

    pub fn try_load(
        collection_id: CollectionId,
        index_id: IndexId,
        data_dir: PathBuf,
        op_sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        hook_runtime: Arc<HooksRuntime>,
        automatically_chosen_properties: Arc<AutomaticEmbeddingsSelector>,
    ) -> Result<Self> {
        let dump: IndexDump = BufferedFile::open(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .read_json_data()
            .context("Cannot serialize collection info")?;
        let IndexDump::V1(dump) = dump;

        let filter_fields = dump
            .filter_fields
            .into_iter()
            .map(IndexFilterField::load_from)
            .collect();
        let score_fields = dump
            .score_fields
            .into_iter()
            .map(|d| {
                IndexScoreField::load_from(
                    d,
                    collection_id,
                    index_id,
                    hooks_runtime.clone(),
                    embedding_sender.clone(),
                    automatically_chosen_properties.clone(),
                )
            })
            .collect();

        let doc_id_storage = DocIdStorage::load(data_dir)?;

        Ok(Self {
            collection_id,
            index_id,
            locale: Locale::EN, // TODO: load from file
            text_parser: Arc::new(TextParser::from_locale(Locale::EN)),

            doc_id_storage: RwLock::new(doc_id_storage),

            filter_fields: RwLock::new(filter_fields),
            score_fields: RwLock::new(score_fields),

            field_id_generator: AtomicU16::new(0),

            op_sender,
            hook_runtime,
            embedding_sender,
            automatically_chosen_properties,

            created_at: dump.created_at,
        })
    }

    pub fn get_locale(&self) -> Locale {
        self.locale
    }

    pub async fn get_embedding_field(
        &self,
        field_path: Box<[String]>,
    ) -> Result<IndexEmbeddingsCalculation> {
        let score_fields = self.score_fields.read().await;
        let field = match get_field_by_path(&field_path, &*score_fields) {
            Some(field) => field,
            None => {
                return Err(anyhow::anyhow!("Field not found"));
            }
        };
        let field = match field {
            IndexScoreField::Embedding(field) => field,
            _ => {
                return Err(anyhow::anyhow!("Field is not an embedding field"));
            }
        };
        match field.get_embedding_calculation() {
            EmbeddingStringCalculation::AllProperties => {
                Ok(IndexEmbeddingsCalculation::AllProperties)
            }
            EmbeddingStringCalculation::Automatic => Ok(IndexEmbeddingsCalculation::Automatic),
            EmbeddingStringCalculation::Properties(v) => {
                let mut results = Vec::new();
                for path in v.iter() {
                    results.push(path.join("."));
                }
                Ok(IndexEmbeddingsCalculation::Properties(results))
            }
            EmbeddingStringCalculation::Hook(_) => Ok(IndexEmbeddingsCalculation::Hook),
        }
    }

    pub async fn add_embedding_field(
        &self,
        field_path: Box<[String]>,
        model: OramaModel,
        embedding_calculation: IndexEmbeddingsCalculation,
    ) -> Result<()> {
        let field_id = self
            .field_id_generator
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let field_id = FieldId(field_id);

        let string_calculation = match embedding_calculation {
            IndexEmbeddingsCalculation::AllProperties => EmbeddingStringCalculation::AllProperties,
            IndexEmbeddingsCalculation::Automatic => EmbeddingStringCalculation::Automatic,
            IndexEmbeddingsCalculation::Properties(v) => {
                EmbeddingStringCalculation::Properties(field_names_to_paths(v))
            }
            IndexEmbeddingsCalculation::Hook => {
                EmbeddingStringCalculation::Hook(self.hook_runtime.clone())
            }
        };

        let field = IndexScoreField::new_embedding(
            self.collection_id,
            self.index_id,
            field_id,
            field_path.clone(),
            model,
            string_calculation,
            self.embedding_sender.clone(),
            self.automatically_chosen_properties.clone(),
        );

        let mut field_lock = self.score_fields.write().await;
        field_lock.push(field);
        drop(field_lock);

        self.op_sender
            .send(WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    self.index_id,
                    IndexWriteOperation::CreateField2 {
                        field_id,
                        field_path,
                        is_array: false,
                        field_type: IndexWriteOperationFieldType::Embedding(
                            OramaModelSerializable(model),
                        ),
                    },
                ),
            ))
            .await
            .context("Cannot send operation")?;

        Ok(())
    }

    pub async fn commit(&self, data_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&data_dir).context("Cannot create data directory")?;

        let filter_fields = self.filter_fields.read().await;
        let score_fields = self.score_fields.read().await;
        let dump = IndexDump::V1(IndexDumpV1 {
            id: self.index_id,
            locale: self.locale,
            field_id_generator: self
                .field_id_generator
                .load(std::sync::atomic::Ordering::Relaxed),
            filter_fields: filter_fields.iter().map(|f| f.serialize()).collect(),
            score_fields: score_fields.iter().map(|f| f.serialize()).collect(),
            created_at: self.created_at,
        });
        drop(filter_fields);
        drop(score_fields);

        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        let doc_id_storage = self.doc_id_storage.read().await;
        doc_id_storage
            .commit(data_dir)
            .context("Cannot commit index")?;

        Ok(())
    }

    pub async fn get_document_ids(&self) -> Vec<DocumentId> {
        let doc_id_storage = self.doc_id_storage.read().await;
        doc_id_storage.get_document_ids().collect()
    }

    pub async fn reindex_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        // The document is already:
        // - indexed (but in another index)
        // - added to the document storage (and shared among all indexes)
        // So, we just need to reindex it and stop.

        // Inspect the doc and create the fields if needed
        self.add_fields_if_needed(&doc)
            .await
            .context("Cannot add fields")?;

        self.process_document(doc_id, doc)
            .await
            .context("Cannot process document")?;

        Ok(())
    }

    pub async fn process_new_document(
        &self,
        doc_id: DocumentId,
        doc: Document,
    ) -> Result<Option<DocumentId>> {
        let mut old_document_id = None;

        // Those `?` is never triggered, but it's here to make the compiler happy:
        // The "id" property is always present in the document.
        // TODO: do this better
        let doc_id_str = doc
            .inner
            .get("id")
            .context("Document does not have an id")?
            .as_str()
            .context("Document id is not a string")?;

        // Check if the document is already indexed. If so, we will replace it
        let mut doc_id_storage = self.doc_id_storage.write().await;
        if let Some(old_doc_id) = doc_id_storage.insert_document_id(doc_id_str.to_string(), doc_id)
        {
            info!("Document already indexed, replacing it.");
            old_document_id = Some(old_doc_id);

            // Remove the old document
            self.op_sender
                .send(WriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::DeleteDocuments {
                        doc_ids: vec![old_doc_id],
                    },
                ))
                .await
                .context("Cannot send operation")?;
            self.op_sender
                .send(WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::IndexWriteOperation(
                        self.index_id,
                        IndexWriteOperation::DeleteDocuments {
                            doc_ids: vec![old_doc_id],
                        },
                    ),
                ))
                .await
                .context("Cannot send operation")?;
        }
        drop(doc_id_storage);

        // Inspect the doc and create the fields if needed
        self.add_fields_if_needed(&doc)
            .await
            .context("Cannot add fields")?;

        self.process_document(doc_id, doc)
            .await
            .context("Cannot process document")?;

        trace!("Document processed");

        Ok(old_document_id)
    }

    async fn process_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        let mut doc_indexed_values: Vec<IndexedValue> = vec![];

        let filter_fields = self.filter_fields.read().await;
        for field in &*filter_fields {
            let indexed_values = match field.index_value(doc_id, &doc.inner).await {
                Ok(indexed_values) => indexed_values,
                Err(_) => continue,
            };

            doc_indexed_values.extend(indexed_values);
        }
        drop(filter_fields);

        let score_fields = self.score_fields.read().await;
        for field in &*score_fields {
            let indexed_values = match field.index_value(doc_id, &doc.inner).await {
                Ok(indexed_values) => indexed_values,
                Err(_) => continue,
            };

            doc_indexed_values.extend(indexed_values);
        }
        drop(score_fields);

        // This case is odd because it means that the document has no fields (just the "id" field)
        // Should we warn about it?
        if !doc_indexed_values.is_empty() {
            self.op_sender
                .send(WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::IndexWriteOperation(
                        self.index_id,
                        IndexWriteOperation::Index {
                            doc_id,
                            indexed_values: doc_indexed_values,
                        },
                    ),
                ))
                .await
                .context("Cannot send operation")?;
        }

        Ok(())
    }

    #[instrument(skip(self, doc_ids), fields(collection_id = ?self.collection_id, index_id = ?self.index_id))]
    pub async fn delete_documents(&self, doc_ids: Vec<String>) -> Result<Vec<DocumentId>> {
        info!("Deleting documents: {:?}", doc_ids);
        let mut doc_id_storage = self.doc_id_storage.write().await;
        let doc_ids = doc_id_storage.remove_document_ids(doc_ids);

        self.op_sender
            .send(WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    self.index_id,
                    IndexWriteOperation::DeleteDocuments {
                        doc_ids: doc_ids.clone(),
                    },
                ),
            ))
            .await
            .context("Cannot send operation")?;
        self.op_sender
            .send(WriteOperation::DocumentStorage(
                DocumentStorageWriteOperation::DeleteDocuments {
                    doc_ids: doc_ids.clone(),
                },
            ))
            .await
            .context("Cannot send operation")?;
        drop(doc_id_storage);

        Ok(doc_ids)
    }

    pub async fn describe(&self) -> DescribeCollectionIndexResponse {
        let filter_fields = self.filter_fields.read().await;
        let score_fields = self.score_fields.read().await;

        let mut fields = Vec::new();
        fields.extend(filter_fields.iter().map(|f| IndexFieldType {
            field_id: f.field_id(),
            field_path: f.field_path().to_vec().join("."),
            field_type: f.field_type(),
            is_array: f.is_array(),
        }));
        fields.extend(score_fields.iter().map(|f| IndexFieldType {
            field_id: f.field_id(),
            field_path: f.field_path().to_vec().join("."),
            field_type: f.field_type(),
            is_array: f.is_array(),
        }));

        // Sort by field id to a consistent order
        fields.sort_by_key(|a| a.field_id.0);

        let document_count = self.doc_id_storage.read().await.len();

        // FieldId(0) is the default embedding field by construction
        let automatically_chosen_properties = if let Some(IndexScoreField::Embedding(field)) =
            score_fields.iter().find(|f| f.field_id() == FieldId(0))
        {
            Some(field.get_automatic_embeddings_selector().await)
        } else {
            None
        };

        DescribeCollectionIndexResponse {
            id: self.index_id,
            document_count,
            fields,
            automatically_chosen_properties,
            created_at: self.created_at,
        }
    }

    #[instrument(skip(self, doc))]
    async fn add_fields_if_needed(&self, doc: &Document) -> Result<()> {
        let flatten = calculate_fields_from_doc(&doc.inner);

        let score_fields = self.score_fields.read().await;
        let filter_fields = self.filter_fields.read().await;

        let mut fields_to_add: (Vec<IndexFilterField>, Vec<IndexScoreField>) = Default::default();

        // For every field path, we need to check if it is already indexed
        // If not, we need to create it
        for field_path in &flatten.0 {
            // We don't index the "id" field at all.
            if field_path == &["id"] {
                continue;
            }

            if get_field_by_path(field_path, &*score_fields).is_some() {
                continue;
            }
            if get_field_by_path(field_path, &*filter_fields).is_some() {
                continue;
            }

            let value = get_value(&doc.inner, field_path);
            let value = if let Some(value) = value {
                if value.is_null() {
                    continue;
                }
                value
            } else {
                continue;
            };

            let (filter_field, score_field) = calculate_fields_for(
                field_path,
                value,
                &self.text_parser,
                &self.field_id_generator,
            );
            if let Some(filter_field) = filter_field {
                self.op_sender
                    .send(WriteOperation::Collection(
                        self.collection_id,
                        CollectionWriteOperation::IndexWriteOperation(
                            self.index_id,
                            IndexWriteOperation::CreateField2 {
                                field_id: filter_field.field_id(),
                                field_path: filter_field.field_path().to_vec().into_boxed_slice(),
                                is_array: filter_field.is_array(),
                                field_type: match &filter_field {
                                    IndexFilterField::Bool(_) => IndexWriteOperationFieldType::Bool,
                                    IndexFilterField::Number(_) => {
                                        IndexWriteOperationFieldType::Number
                                    }
                                    IndexFilterField::String(_) => {
                                        IndexWriteOperationFieldType::StringFilter
                                    }
                                },
                            },
                        ),
                    ))
                    .await
                    .context("Cannot send operation")?;
                fields_to_add.0.push(filter_field);
            }
            if let Some(score_field) = score_field {
                self.op_sender
                    .send(WriteOperation::Collection(
                        self.collection_id,
                        CollectionWriteOperation::IndexWriteOperation(
                            self.index_id,
                            IndexWriteOperation::CreateField2 {
                                field_id: score_field.field_id(),
                                field_path: score_field.field_path().to_vec().into_boxed_slice(),
                                is_array: score_field.is_array(),
                                field_type: match &score_field {
                                    IndexScoreField::String(_) => {
                                        IndexWriteOperationFieldType::String
                                    }
                                    IndexScoreField::Embedding(f) => {
                                        IndexWriteOperationFieldType::Embedding(
                                            OramaModelSerializable(f.get_model()),
                                        )
                                    }
                                },
                            },
                        ),
                    ))
                    .await
                    .context("Cannot send operation")?;

                fields_to_add.1.push(score_field);
            }
        }
        drop(score_fields);
        drop(filter_fields);

        if !fields_to_add.0.is_empty() {
            let mut filter_fields = self.filter_fields.write().await;
            for field in fields_to_add.0 {
                // We calculate concurrently the fields to add
                // But we need to check if the field is already indexed
                // before adding it to the index
                // Here is the right place to do it because we have the write lock
                if get_field_by_path(field.field_path(), &*filter_fields).is_some() {
                    continue;
                }
                filter_fields.push(field);
            }
            drop(filter_fields);
        }
        if !fields_to_add.1.is_empty() {
            let mut score_fields = self.score_fields.write().await;
            for field in fields_to_add.1 {
                // We calculate concurrently the fields to add
                // But we need to check if the field is already indexed
                // before adding it to the index
                // Here is the right place to do it because we have the write lock
                if get_field_by_path(field.field_path(), &*score_fields).is_some() {
                    continue;
                }
                score_fields.push(field);
            }
            drop(score_fields);
        }

        Ok(())
    }

    pub async fn switch_to_embedding_hook(&self, hooks_runtime: Arc<HooksRuntime>) -> Result<()> {
        let mut score_fields = self.score_fields.write().await;

        for field in score_fields.iter_mut() {
            if let IndexScoreField::Embedding(field) = field {
                field.switch_to_embedding_hook(hooks_runtime.clone());
            }
        }

        Ok(())
    }
}

fn calculate_fields_for(
    field_path: &[&str],
    value: &Value,
    text_parser: &Arc<TextParser>,
    field_id_generator: &AtomicU16,
) -> (Option<IndexFilterField>, Option<IndexScoreField>) {
    let mut filter_field = None;
    let mut score_field = None;

    let field_path = field_path
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let generate_id = || {
        let id = field_id_generator.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        FieldId(id)
    };

    match value {
        Value::Bool(_) => {
            let field = IndexFilterField::new_bool(generate_id(), field_path);
            filter_field = Some(field);
        }
        Value::Number(_) => {
            let field = IndexFilterField::new_number(generate_id(), field_path);
            filter_field = Some(field);
        }
        Value::String(_) => {
            let field = IndexFilterField::new_string(generate_id(), field_path.clone());
            filter_field = Some(field);

            let field = IndexScoreField::new_string(generate_id(), field_path, text_parser.clone());
            score_field = Some(field);
        }
        Value::Array(arr) => {
            let first_non_null_value = arr.iter().find(|v| !matches!(v, Value::Null));
            let Some(first_non_null_value) = first_non_null_value else {
                // If all values are null, we don't know which type to use
                // So we just return None for both
                // If another document will specify the type, we will create the field then
                return (None, None);
            };

            match first_non_null_value {
                Value::Bool(_) => {
                    let field = IndexFilterField::new_bool_arr(generate_id(), field_path);
                    filter_field = Some(field);
                }
                Value::Number(_) => {
                    let field = IndexFilterField::new_number_arr(generate_id(), field_path);
                    filter_field = Some(field);
                }
                Value::String(_) => {
                    let field = IndexFilterField::new_string_arr(generate_id(), field_path.clone());
                    filter_field = Some(field);

                    let field = IndexScoreField::new_string_arr(
                        generate_id(),
                        field_path,
                        text_parser.clone(),
                    );
                    score_field = Some(field);
                }
                Value::Array(_) | Value::Object(_) | Value::Null => {
                    // We don't support nested arrays or objects
                    // So we just return None for both
                    return (None, None);
                }
            }
        }
        Value::Null => {}
        Value::Object(_) => {
            use debug_panic::debug_panic;
            eprintln!("Path: {:?}", field_path);
            debug_panic!("Something is wrong: we should never have an object here.");
        }
    };

    (filter_field, score_field)
}

fn get_field_by_path<'doc, 'index, X: AsRef<str>, T: GenericField>(
    field_path: &[X],
    fields: &'index Vec<T>,
) -> Option<&'index T> {
    fields
        .iter()
        .find(|f| path_is_equal(f.field_path(), field_path))
}

#[inline]
fn path_is_equal<X: AsRef<str>>(path: &[String], field_path: &[X]) -> bool {
    if path.len() != field_path.len() {
        return false;
    }
    for (p1, p2) in path.iter().zip(field_path.iter()) {
        if p1 != p2.as_ref() {
            return false;
        }
    }
    true
}

fn get_value<'doc, X: AsRef<str>>(
    doc: &'doc Map<String, Value>,
    field_path: &[X],
) -> Option<&'doc Value> {
    fn recursive_object_inspection<'doc, X: AsRef<str>>(
        obj: &'doc Map<String, Value>,
        field_path: &[X],
        index: usize,
    ) -> Option<&'doc Value> {
        if field_path.is_empty() {
            return None;
        }

        let key = &field_path[index];
        let key = key.as_ref();
        let value = obj.get(key)?;

        if let Value::Object(obj) = value {
            return recursive_object_inspection(obj, field_path, index + 1);
        }
        // We threat null values as empty values
        // This is because we don't index null values
        if matches!(value, Value::Null) {
            return None;
        }

        Some(value)
    }

    recursive_object_inspection(doc, field_path, 0)
}

fn calculate_fields_from_doc(obj: &Map<String, Value>) -> FlattenFields<'_> {
    fn recursive_object_inspection<'s>(
        obj: &'s Map<String, Value>,
        fields: &mut Vec<Vec<&'s str>>,
        stack: Vec<&'s str>,
    ) {
        for (key, value) in obj {
            match value {
                Value::Null => continue,
                Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
                    let mut path = stack.clone();
                    path.push(key);
                    fields.push(path);
                }
                Value::Object(obj) => {
                    let mut path = stack.clone();
                    path.push(key);
                    recursive_object_inspection(obj, fields, path);
                }
            }
        }
    }

    let mut fields = Vec::new();
    recursive_object_inspection(obj, &mut fields, vec![]);

    FlattenFields(fields)
}

#[derive(Debug, Clone)]
struct FlattenFields<'field>(Vec<Vec<&'field str>>);

#[derive(Debug, Serialize, Deserialize)]
enum IndexDump {
    #[serde(rename = "v1")]
    V1(IndexDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexDumpV1 {
    id: IndexId,
    locale: Locale,
    field_id_generator: u16,
    filter_fields: Vec<SerializedFilterFieldType>,
    score_fields: Vec<SerializedScoreFieldType>,

    created_at: DateTime<Utc>,
}
