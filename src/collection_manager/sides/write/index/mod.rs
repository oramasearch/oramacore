mod doc_id_storage;
mod fields;

use std::{
    borrow::Cow,
    collections::HashMap,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU16, Ordering},
        Arc,
    },
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
use tracing::{info, instrument, trace, warn};

use crate::{
    collection_manager::sides::{
        field_names_to_paths, write::context::WriteSideContext, CollectionWriteOperation,
        DocumentStorageWriteOperation, IndexWriteOperation, IndexWriteOperationFieldType,
        WriteOperation,
    },
    lock::{OramaAsyncLock, OramaAsyncLockReadGuard},
    python::embeddings::Model,
    types::{
        CollectionId, DescribeCollectionIndexResponse, Document, DocumentId, DocumentList, FieldId,
        IndexEmbeddingsCalculation, IndexFieldType, IndexId, OramaDate,
    },
};
use oramacore_lib::fs::BufferedFile;
use oramacore_lib::nlp::{locales::Locale, TextParser};

pub use fields::{EnumStrategy, FieldType, GeoPoint, IndexedValue};

#[derive(Clone)]
pub enum EmbeddingStringCalculation {
    AllProperties,
    Properties(Box<[Box<[String]>]>),
    Automatic,
}

pub struct Index {
    collection_id: CollectionId,
    index_id: IndexId,
    locale: Locale,
    text_parser: Arc<TextParser>,

    filter_fields: OramaAsyncLock<Vec<IndexFilterField>>,
    score_fields: OramaAsyncLock<Vec<IndexScoreField>>,

    doc_id_storage: OramaAsyncLock<DocIdStorage>,

    field_id_generator: AtomicU16,

    context: WriteSideContext,

    created_at: DateTime<Utc>,

    runtime_index_id: Option<IndexId>,

    enum_strategy: EnumStrategy,

    /// Track if the index has unsaved changes that need to be committed
    is_dirty: AtomicBool,
}

impl Index {
    pub async fn empty(
        index_id: IndexId,
        collection_id: CollectionId,
        runtime_index_id: Option<IndexId>,
        text_parser: Arc<TextParser>,
        context: WriteSideContext,
        enum_strategy: EnumStrategy,
    ) -> Result<Self> {
        let field_id = 0;
        let score_fields = vec![];

        let locale = text_parser.locale();

        Ok(Self {
            collection_id,
            index_id,
            locale,
            text_parser,

            doc_id_storage: OramaAsyncLock::new("doc_id_storage", DocIdStorage::empty()),

            filter_fields: OramaAsyncLock::new("filter_fields", Default::default()),
            score_fields: OramaAsyncLock::new("score_fields", score_fields),

            field_id_generator: AtomicU16::new(field_id),

            context,

            created_at: Utc::now(),

            runtime_index_id,

            enum_strategy,

            is_dirty: AtomicBool::new(false),
        })
    }

    pub fn try_load(
        collection_id: CollectionId,
        index_id: IndexId,
        data_dir: PathBuf,
        context: WriteSideContext,
    ) -> Result<Self> {
        let dump: IndexDump = BufferedFile::open(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .read_json_data()
            .context("Cannot serialize collection info")?;
        let dump = match dump {
            IndexDump::V1(dump) => IndexDumpV2 {
                id: dump.id,
                locale: dump.locale,
                field_id_generator: dump.field_id_generator,
                filter_fields: dump.filter_fields,
                score_fields: dump.score_fields,
                created_at: dump.created_at,
                runtime_index_id: None,
                enum_strategy: get_default_enum_strategy(),
            },
            IndexDump::V2(dump) => dump,
        };

        let filter_fields = dump
            .filter_fields
            .into_iter()
            .map(|field| IndexFilterField::load_from(field, dump.enum_strategy))
            .collect();
        let score_fields = dump
            .score_fields
            .into_iter()
            .map(|d| IndexScoreField::load_from(d, collection_id, index_id, context.clone()))
            .collect();

        let doc_id_storage = DocIdStorage::load(data_dir.clone())?;

        Ok(Self {
            collection_id,
            index_id,
            locale: Locale::EN, // TODO: load from file
            text_parser: Arc::new(TextParser::from_locale(Locale::EN)),

            doc_id_storage: OramaAsyncLock::new("doc_id_storage", doc_id_storage),

            filter_fields: OramaAsyncLock::new("filter_fields", filter_fields),
            score_fields: OramaAsyncLock::new("score_fields", score_fields),

            field_id_generator: AtomicU16::new(dump.field_id_generator),

            context,

            created_at: dump.created_at,

            runtime_index_id: dump.runtime_index_id,

            enum_strategy: dump.enum_strategy,

            is_dirty: AtomicBool::new(false),
        })
    }

    pub fn is_dirty(&self) -> bool {
        self.is_dirty.load(Ordering::SeqCst)
    }

    pub fn set_index_id(&mut self, index_id: IndexId) {
        self.index_id = index_id;
        self.is_dirty.store(true, Ordering::SeqCst);
    }

    pub fn get_locale(&self) -> Locale {
        self.locale
    }

    pub fn get_runtime_index_id(&self) -> Option<IndexId> {
        self.runtime_index_id
    }

    pub fn is_temp(&self) -> bool {
        self.runtime_index_id.is_some()
    }

    pub fn is_runtime(&self) -> bool {
        self.runtime_index_id.is_none()
    }

    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    pub fn get_enum_strategy(&self) -> EnumStrategy {
        self.enum_strategy
    }

    pub async fn get_embedding_field(
        &self,
        field_path: Box<[String]>,
    ) -> Result<IndexEmbeddingsCalculation> {
        let score_fields = self.score_fields.read("get_embedding_field").await;
        let field = match get_field_by_path(&field_path, &score_fields) {
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
        }
    }

    pub async fn add_embedding_field(
        &self,
        field_path: Box<[String]>,
        model: Model,
        embedding_calculation: IndexEmbeddingsCalculation,
    ) -> Result<()> {
        let field_id = self
            .field_id_generator
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let field_id = FieldId(field_id);

        let string_calculation = match embedding_calculation {
            IndexEmbeddingsCalculation::AllProperties => EmbeddingStringCalculation::AllProperties,
            IndexEmbeddingsCalculation::Automatic => EmbeddingStringCalculation::Automatic,
            IndexEmbeddingsCalculation::None => return Ok(()),
            IndexEmbeddingsCalculation::Properties(v) => {
                EmbeddingStringCalculation::Properties(field_names_to_paths(v))
            }
        };

        let field = IndexScoreField::new_embedding(
            self.collection_id,
            self.index_id,
            field_id,
            field_path.clone(),
            model,
            self.context.clone(),
            string_calculation,
        );

        let mut field_lock = self.score_fields.write("add_embedding_field").await;
        field_lock.push(field);
        drop(field_lock);

        self.context
            .op_sender
            .send(WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    self.index_id,
                    IndexWriteOperation::CreateField2 {
                        field_id,
                        field_path,
                        is_array: false,
                        field_type: IndexWriteOperationFieldType::Embedding(model),
                    },
                ),
            ))
            .await
            .context("Cannot send operation")?;

        self.is_dirty.store(true, Ordering::SeqCst);

        Ok(())
    }

    pub async fn commit(&self, data_dir: PathBuf) -> Result<()> {
        std::fs::create_dir_all(&data_dir).context("Cannot create data directory")?;

        let (filter_fields, score_fields) = tokio::join!(
            self.filter_fields.read("commit"),
            self.score_fields.read("commit")
        );

        if !self.is_dirty.load(Ordering::SeqCst) {
            info!("Index is not dirty, skipping commit");
            return Ok(());
        }

        let dump = IndexDump::V1(IndexDumpV1 {
            id: self.index_id,
            locale: self.locale,
            field_id_generator: self
                .field_id_generator
                .load(std::sync::atomic::Ordering::Relaxed),
            filter_fields: filter_fields.iter().map(|f| f.serialize()).collect(),
            score_fields: score_fields.iter().map(|f| f.serialize()).collect(),
            created_at: self.created_at,
            enum_strategy: self.enum_strategy,
        });

        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        let doc_id_storage = self.doc_id_storage.read("commit").await;
        doc_id_storage
            .commit(data_dir)
            .context("Cannot commit index")?;

        self.is_dirty.store(false, Ordering::SeqCst);

        drop(filter_fields);
        drop(score_fields);

        Ok(())
    }

    pub async fn get_document_ids(&self) -> Vec<DocumentId> {
        let doc_id_storage = self.doc_id_storage.read("get").await;
        doc_id_storage.get_document_ids().collect()
    }

    pub async fn reindex_document(
        &self,
        doc_id: DocumentId,
        doc: Document,
        index_operation_batch: &mut Vec<WriteOperation>,
    ) -> Result<()> {
        // The document is already:
        // - indexed (but in another index)
        // - added to the document storage
        // So, we just need to reindex it and stop.

        self.process_document(doc_id, doc, index_operation_batch)
            .await
            .context("Cannot process document")?;

        Ok(())
    }

    pub async fn get_document_id_storage<'index>(&'index self) -> DocIdStorageReadLock<'index> {
        let doc_id_storage = self.doc_id_storage.read("get").await;

        DocIdStorageReadLock { doc_id_storage }
    }

    pub async fn process_new_document(
        &self,
        doc_id: DocumentId,
        doc_id_str: String,
        doc: Document,
        index_operation_batch: &mut Vec<WriteOperation>,
    ) -> Result<Option<DocumentId>> {
        let mut old_document_id = None;

        // Check if the document is already indexed. If so, we will replace it
        let mut doc_id_storage = self.doc_id_storage.write("new_doc").await;
        if let Some(old_doc_id) = doc_id_storage.insert_document_id(doc_id_str, doc_id) {
            trace!("Document already indexed, replacing it.");
            old_document_id = Some(old_doc_id);

            // Remove the old document
            // Note: We use DeleteDocuments (not DeleteDocumentsWithDocIdStr) here because
            // insert_document_id has already updated the doc_id_str mapping to point to the new doc_id.
            // Only the old document data needs to be deleted from storage and index.
            index_operation_batch.push(WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::DeleteDocuments {
                        doc_ids: vec![old_doc_id],
                    },
                ),
            ));
            index_operation_batch.push(WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    self.index_id,
                    IndexWriteOperation::DeleteDocuments {
                        doc_ids: vec![old_doc_id],
                    },
                ),
            ));
        }
        drop(doc_id_storage);

        self.process_document(doc_id, doc, index_operation_batch)
            .await
            .context("Cannot process document")?;

        Ok(old_document_id)
    }

    async fn process_document(
        &self,
        doc_id: DocumentId,
        doc: Document,
        index_operation_batch: &mut Vec<WriteOperation>,
    ) -> Result<()> {
        let mut doc_indexed_values: Vec<IndexedValue> = vec![];

        let filter_fields = self.filter_fields.read("process").await;
        for field in &**filter_fields {
            let indexed_values = match field.index_value(doc_id, &doc.inner).await {
                Ok(indexed_values) => indexed_values,
                Err(_) => continue,
            };

            doc_indexed_values.extend(indexed_values);
        }
        drop(filter_fields);

        let score_fields = self.score_fields.read("process").await;
        for field in &**score_fields {
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
            index_operation_batch.push(WriteOperation::Collection(
                self.collection_id,
                CollectionWriteOperation::IndexWriteOperation(
                    self.index_id,
                    IndexWriteOperation::Index {
                        doc_id,
                        indexed_values: doc_indexed_values,
                    },
                ),
            ));
        }

        self.is_dirty.store(true, Ordering::SeqCst);

        Ok(())
    }

    #[instrument(skip(self, doc_ids), fields(collection_id = ?self.collection_id, index_id = ?self.index_id))]
    pub async fn delete_documents(
        &self,
        doc_ids: Vec<String>,
    ) -> Result<Vec<(DocumentId, String)>> {
        info!("Deleting documents: {:?}", doc_ids);

        let doc_ids: Vec<&str> = doc_ids
            .iter()
            .map(|id| {
                id.strip_prefix(self.index_id.as_str())
                    .and_then(|a| a.strip_prefix(":"))
                    .unwrap_or(id)
            })
            .collect();

        let mut doc_id_storage = self.doc_id_storage.write("delete").await;
        let doc_id_pairs = doc_id_storage.remove_document_ids(doc_ids);

        if !doc_id_pairs.is_empty() {
            // Send document storage operation FIRST with doc_id_str for proper cleanup
            self.context
                .op_sender
                .send(WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::DocumentStorage(
                        DocumentStorageWriteOperation::DeleteDocumentsWithDocIdStr(
                            doc_id_pairs.clone(),
                        ),
                    ),
                ))
                .await
                .context("Cannot send operation")?;

            // Send index deletion (only needs DocumentId)
            let doc_ids: Vec<DocumentId> = doc_id_pairs.iter().map(|(doc_id, _)| *doc_id).collect();

            self.context
                .op_sender
                .send(WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::IndexWriteOperation(
                        self.index_id,
                        IndexWriteOperation::DeleteDocuments { doc_ids },
                    ),
                ))
                .await
                .context("Cannot send operation")?;
        } else {
            warn!("No document to delete");
        }
        drop(doc_id_storage);

        self.is_dirty.store(true, Ordering::SeqCst);

        Ok(doc_id_pairs)
    }

    pub async fn describe(&self) -> DescribeCollectionIndexResponse {
        let filter_fields = self.filter_fields.read("describe").await;
        let score_fields = self.score_fields.read("describe").await;

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

        let document_count = self.get_document_count("describe").await;

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

    pub async fn get_document_count(&self, reason: &'static str) -> usize {
        let doc_id_storage = self.doc_id_storage.read(reason).await;
        doc_id_storage.len()
    }

    pub async fn add_fields_if_needed(&self, docs: &DocumentList) -> Result<()> {
        let mut index_operation_batch = Vec::with_capacity(10);

        #[derive(Debug)]
        enum F {
            AlreadyInserted,
            Bool(Value),
            Number(Value),
            String(Value),
            GeoPoint(Value),
            Array(Value),
        }

        fn recursive_object_inspection<'s>(
            obj: &'s Map<String, Value>,
            fields: &mut HashMap<Box<[Cow<'s, String>]>, F>,
            stack: Vec<Cow<'s, String>>,
        ) {
            for (key, value) in obj {
                let f = match value {
                    Value::Null => continue,
                    Value::Array(arr) => {
                        if arr.is_empty() {
                            continue;
                        }
                        let first_non_null_value = arr.iter().find(|v| !matches!(v, Value::Null));
                        let Some(first_non_null_value) = first_non_null_value else {
                            // If all values are null, we don't know which type to use
                            // So we just return
                            continue;
                        };

                        match first_non_null_value {
                            Value::Bool(_) => F::Array(Value::Array(arr.clone())),
                            Value::Number(_) => F::Array(Value::Array(arr.clone())),
                            Value::String(_) => F::Array(Value::Array(arr.clone())),
                            Value::Array(_) | Value::Object(_) | Value::Null => {
                                // We don't support nested arrays or objects
                                // So we just return
                                continue;
                            }
                        }
                    }
                    Value::Object(obj) => {
                        if obj.len() == 2 && obj.contains_key("lat") && obj.contains_key("lon") {
                            let mut path = stack.clone();
                            path.push(Cow::Borrowed(key));

                            F::GeoPoint(Value::Object(obj.clone()))
                        } else {
                            let mut path = stack.clone();
                            path.push(Cow::Borrowed(key));
                            recursive_object_inspection(obj, fields, path);
                            continue;
                        }
                    }
                    Value::Bool(v) => F::Bool(Value::Bool(*v)),
                    Value::Number(v) => F::Number(Value::Number(v.clone())),
                    Value::String(v) => F::String(Value::String(v.clone())),
                };

                let mut path = stack.clone();
                path.push(Cow::Borrowed(key));
                let key = path.into_boxed_slice();
                fields.entry(key).or_insert(f);
            }
        }

        let score_fields = self.score_fields.read("add_field").await;
        let filter_fields = self.filter_fields.read("add_field").await;

        let mut current_fields = score_fields
            .iter()
            .map(|f| {
                let a: Vec<_> = f
                    .field_path()
                    .iter()
                    .map(|s| Cow::Owned(s.to_string()))
                    .collect();
                (a.into_boxed_slice(), F::AlreadyInserted)
            })
            .chain(filter_fields.iter().map(|f| {
                let a: Vec<_> = f
                    .field_path()
                    .iter()
                    .map(|s| Cow::Owned(s.to_string()))
                    .collect();
                (a.into_boxed_slice(), F::AlreadyInserted)
            }))
            .collect::<HashMap<_, _>>();
        drop(score_fields);
        drop(filter_fields);

        // `id` field is always present in the documents
        current_fields.insert(Box::new([Cow::Owned("id".to_string())]), F::AlreadyInserted);

        for doc in &docs.0 {
            recursive_object_inspection(&doc.inner, &mut current_fields, vec![])
        }

        current_fields.retain(|_, f| !matches!(f, F::AlreadyInserted));

        if current_fields.is_empty() {
            // No new fields to add
            // Avoid locking the index fields
            return Ok(());
        }

        let mut score_fields = self.score_fields.write("add_field").await;
        let mut filter_fields = self.filter_fields.write("add_field").await;
        for (k, f) in current_fields {
            let (filter, score) = match f {
                F::AlreadyInserted => continue,
                F::Bool(v) | F::Number(v) | F::String(v) | F::Array(v) | F::GeoPoint(v) => {
                    calculate_fields_for(
                        &k,
                        &v,
                        &self.text_parser,
                        &self.field_id_generator,
                        self.enum_strategy,
                    )
                }
            };

            if let Some(filter) = filter {
                index_operation_batch.push(WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::IndexWriteOperation(
                        self.index_id,
                        IndexWriteOperation::CreateField2 {
                            field_id: filter.field_id(),
                            field_path: filter.field_path().to_vec().into_boxed_slice(),
                            is_array: filter.is_array(),
                            field_type: match &filter {
                                IndexFilterField::Bool(_) => IndexWriteOperationFieldType::Bool,
                                IndexFilterField::Number(_) => IndexWriteOperationFieldType::Number,
                                IndexFilterField::String(_) => {
                                    IndexWriteOperationFieldType::StringFilter
                                }
                                IndexFilterField::Date(_) => IndexWriteOperationFieldType::Date,
                                IndexFilterField::GeoPoint(_) => {
                                    IndexWriteOperationFieldType::GeoPoint
                                }
                            },
                        },
                    ),
                ));

                filter_fields.push(filter);
            }
            if let Some(score) = score {
                index_operation_batch.push(WriteOperation::Collection(
                    self.collection_id,
                    CollectionWriteOperation::IndexWriteOperation(
                        self.index_id,
                        IndexWriteOperation::CreateField2 {
                            field_id: score.field_id(),
                            field_path: score.field_path().to_vec().into_boxed_slice(),
                            is_array: score.is_array(),
                            field_type: match &score {
                                IndexScoreField::String(_) => IndexWriteOperationFieldType::String,
                                IndexScoreField::Embedding(f) => {
                                    IndexWriteOperationFieldType::Embedding(f.get_model())
                                }
                            },
                        },
                    ),
                ));
                score_fields.push(score);
            }
        }
        drop(score_fields);
        drop(filter_fields);

        self.context
            .op_sender
            .send_batch(index_operation_batch)
            .await
            .context("Cannot send add fields operation")?;

        self.is_dirty.store(true, Ordering::SeqCst);

        Ok(())
    }
}

fn calculate_fields_for(
    field_path: &[Cow<String>],
    value: &Value,
    text_parser: &Arc<TextParser>,
    field_id_generator: &AtomicU16,
    enum_strategy: EnumStrategy,
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
        Value::String(s) => {
            if OramaDate::try_from(s).is_ok() {
                let field = IndexFilterField::new_date(generate_id(), field_path.clone());
                filter_field = Some(field);
            } else {
                let field =
                    IndexFilterField::new_string(generate_id(), field_path.clone(), enum_strategy);
                filter_field = Some(field);
            }

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
                    let field = IndexFilterField::new_string_arr(
                        generate_id(),
                        field_path.clone(),
                        enum_strategy,
                    );
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
        Value::Object(obj) => {
            if obj.len() == 2 && obj.contains_key("lon") && obj.contains_key("lat") {
                filter_field = Some(IndexFilterField::new_geopoint(generate_id(), field_path));
            } else {
                use debug_panic::debug_panic;
                eprintln!("Path: {field_path:?}");
                debug_panic!("Something is wrong: we should never have an object here.");
            }
        }
    };

    (filter_field, score_field)
}

fn get_field_by_path<'index, X: AsRef<str>, T: GenericField>(
    field_path: &[X],
    fields: &'index [T],
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
            if index == field_path.len() - 1 {
                return Some(value);
            }

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

#[derive(Debug, Serialize, Deserialize)]
enum IndexDump {
    #[serde(rename = "v1")]
    V1(IndexDumpV1),
    #[serde(rename = "v2")]
    V2(IndexDumpV2),
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexDumpV1 {
    id: IndexId,
    locale: Locale,
    field_id_generator: u16,
    filter_fields: Vec<SerializedFilterFieldType>,
    score_fields: Vec<SerializedScoreFieldType>,

    created_at: DateTime<Utc>,

    #[serde(default = "get_default_enum_strategy")]
    enum_strategy: EnumStrategy,
}

#[derive(Debug, Serialize, Deserialize)]
struct IndexDumpV2 {
    id: IndexId,
    locale: Locale,
    field_id_generator: u16,
    filter_fields: Vec<SerializedFilterFieldType>,
    score_fields: Vec<SerializedScoreFieldType>,

    created_at: DateTime<Utc>,

    runtime_index_id: Option<IndexId>,

    #[serde(default = "get_default_enum_strategy")]
    enum_strategy: EnumStrategy,
}

pub struct DocIdStorageReadLock<'index> {
    doc_id_storage: OramaAsyncLockReadGuard<'index, DocIdStorage>,
}

impl Deref for DocIdStorageReadLock<'_> {
    type Target = DocIdStorage;

    fn deref(&self) -> &Self::Target {
        &self.doc_id_storage
    }
}

fn get_default_enum_strategy() -> EnumStrategy {
    EnumStrategy::StringLength(25)
}
