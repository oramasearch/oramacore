use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU16, AtomicU64},
        Arc,
    },
};

use anyhow::{anyhow, Context, Ok, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

use crate::{
    collection_manager::{
        dto::{CollectionDTO, FieldId},
        sides::hooks::{HookName, HooksRuntime},
    },
    file_utils::BufferedFile,
    metrics::{CommitLabels, COMMIT_METRIC},
    nlp::{locales::Locale, NLPService, TextParser},
    types::{CollectionId, ComplexType, Document, DocumentId, ScalarType, ValueType},
};

use crate::collection_manager::dto::{LanguageDTO, TypedField};

use super::{
    embedding::EmbeddingCalculationRequest, fields::FieldsToIndex, CollectionField,
    CollectionWriteOperation, OperationSender, SerializedFieldIndexer, WriteOperation,
};

pub struct CollectionWriter {
    id: CollectionId,
    description: Option<String>,
    default_language: LanguageDTO,
    fields: DashMap<String, (ValueType, CollectionField)>,

    collection_document_count: AtomicU64,

    field_id_generator: AtomicU16,
    field_id_by_name: DashMap<String, FieldId>,

    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
}

impl CollectionWriter {
    pub fn new(
        id: CollectionId,
        description: Option<String>,
        default_language: LanguageDTO,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    ) -> Self {
        Self {
            id: id.clone(),
            description,
            default_language,
            collection_document_count: Default::default(),
            fields: Default::default(),
            field_id_by_name: DashMap::new(),
            field_id_generator: AtomicU16::new(0),
            embedding_sender,
        }
    }

    pub fn as_dto(&self) -> CollectionDTO {
        CollectionDTO {
            id: self.id.clone(),
            description: self.description.clone(),
            document_count: self
                .collection_document_count
                .load(std::sync::atomic::Ordering::Relaxed),
            fields: self
                .fields
                .iter()
                .map(|e| (e.key().clone(), e.value().0.clone()))
                .collect(),
        }
    }

    pub fn set_embedding_hook(&self, hook_name: HookName) {
        let mut field = match self.fields.get_mut("___orama_auto_embedding") {
            None => return,
            Some(field) => field,
        };

        field.1.set_embedding_hook(hook_name);
    }

    pub async fn process_new_document(
        &self,
        doc_id: DocumentId,
        doc: Document,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<()> {
        self.collection_document_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        sender
            .send(WriteOperation::Collection(
                self.id.clone(),
                CollectionWriteOperation::InsertDocument {
                    doc_id,
                    doc: doc.into_raw()?,
                },
            ))
            .await
            .map_err(|e| anyhow!("Error sending document to index writer: {:?}", e))?;

        let fields_to_index = self
            .get_fields_to_index(doc.clone(), sender.clone(), hooks_runtime)
            .await
            .context("Cannot get fields to index")?;

        let flatten = doc.clone().into_flatten();

        for entry in fields_to_index {
            let (_, field) = entry.value();

            field
                .get_write_operations(doc_id, &flatten, sender.clone())
                .await?;
        }

        Ok(())
    }

    fn get_field_id_by_name(&self, name: &str) -> FieldId {
        use dashmap::Entry;

        let v = self.field_id_by_name.get(name);
        // Fast path
        if let Some(v) = v {
            return *v;
        }
        let entry = self.field_id_by_name.entry(name.to_string());
        match entry {
            // This is odd, but concurrently,
            // we can have the first `get` None and have the entry occupied
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                // Vacant entry locks the map, so we can safely insert the field_id
                let field_id = self
                    .field_id_generator
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let field_id = FieldId(field_id);
                e.insert(field_id);

                info!("Field created {} -> {:?}", name, field_id);

                field_id
            }
        }
    }

    fn value_to_typed_field(&self, value_type: ValueType) -> Option<TypedField> {
        match value_type {
            ValueType::Scalar(ScalarType::String) => Some(TypedField::Text(self.default_language)),
            ValueType::Scalar(ScalarType::Number) => Some(TypedField::Number),
            ValueType::Scalar(ScalarType::Boolean) => Some(TypedField::Bool),
            _ => None, // @todo: support other types
        }
    }

    fn get_text_parser(&self, language: LanguageDTO) -> Arc<TextParser> {
        let locale: Locale = language.into();
        let parser = TextParser::from_locale(locale);
        Arc::new(parser)
    }

    pub async fn register_fields(
        &self,
        typed_fields: HashMap<String, TypedField>,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<()> {
        for (field_name, field_type) in typed_fields {
            let field_id = self.get_field_id_by_name(&field_name);

            self.create_field(
                field_id,
                field_name,
                field_type,
                self.embedding_sender.clone(),
                sender.clone(),
                hooks_runtime.clone(),
            )
            .await
            .context("Cannot create field")?;
        }

        Ok(())
    }

    #[instrument(skip(self, sender, embedding_sender))]
    async fn create_field(
        &self,
        field_id: FieldId,
        field_name: String,
        typed_field: TypedField,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<()> {
        match &typed_field {
            TypedField::Embedding(embedding_field) => {
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Complex(ComplexType::Embedding),
                        CollectionField::new_embedding(
                            embedding_field.model,
                            embedding_field.document_fields.clone(),
                            embedding_sender,
                            hooks_runtime,
                            self.id.clone(),
                            field_id,
                        ),
                    ),
                );
            }
            TypedField::Text(language) => {
                let parser = self.get_text_parser(*language);
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Scalar(ScalarType::String),
                        CollectionField::new_string(
                            parser,
                            self.id.clone(),
                            field_id,
                            field_name.clone(),
                        ),
                    ),
                );
            }
            TypedField::Number => {
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Scalar(ScalarType::Number),
                        CollectionField::new_number(self.id.clone(), field_id, field_name.clone()),
                    ),
                );
            }
            TypedField::Bool => {
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Scalar(ScalarType::Boolean),
                        CollectionField::new_bool(self.id.clone(), field_id, field_name.clone()),
                    ),
                );
            }
        }

        sender
            .send(WriteOperation::Collection(
                self.id.clone(),
                CollectionWriteOperation::CreateField {
                    field_id,
                    field_name,
                    field: typed_field,
                },
            ))
            .await
            .context("Cannot sent creation field")?;
        info!("Field created");

        Ok(())
    }

    async fn get_fields_to_index(
        &self,
        doc: Document,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<&FieldsToIndex> {
        let flatten = doc.clone().into_flatten();
        let schema = flatten.get_field_schema();

        for (field_name, value_type) in schema {
            if self.fields.contains_key(&field_name) {
                continue;
            }

            let field_id = self.get_field_id_by_name(&field_name);

            // @todo: add support to other types
            if let Some(typed_field) = self.value_to_typed_field(value_type) {
                self.create_field(
                    field_id,
                    field_name,
                    typed_field,
                    self.embedding_sender.clone(),
                    sender.clone(),
                    hooks_runtime.clone(),
                )
                .await
                .context("Cannot create field")?;
            }
        }

        Ok(&self.fields)
    }

    pub fn commit(&self, path: PathBuf) -> Result<()> {
        info!("Committing collection {}", self.id.0);

        let m = COMMIT_METRIC.create(CommitLabels {
            side: "write",
            collection: self.id.0.clone(),
            index_type: "info",
        });
        std::fs::create_dir_all(&path).context("Cannot create collection directory")?;
        let dump = CollectionDump::V1(CollectionDumpV1 {
            id: self.id.clone(),
            description: self.description.clone(),
            default_language: self.default_language,
            fields: self
                .fields
                .iter()
                .map(|e| {
                    let k = e.key().clone();
                    let (_, indexer) = e.value();
                    (k, indexer.serialized())
                })
                .collect(),
            document_count: self
                .collection_document_count
                .load(std::sync::atomic::Ordering::Relaxed),
            field_id_generator: self
                .field_id_generator
                .load(std::sync::atomic::Ordering::Relaxed),
            field_id_by_name: self
                .field_id_by_name
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
        });

        BufferedFile::create_or_overwrite(path.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        drop(m);

        Ok(())
    }

    pub async fn load(
        &mut self,
        path: PathBuf,
        hooks_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
    ) -> Result<()> {
        let dump: CollectionDump = BufferedFile::open(path.join("info.json"))
            .context("Cannot open info.json file")?
            .read_json_data()
            .context("Cannot deserialize collection info")?;

        let CollectionDump::V1(dump) = dump;

        self.id = dump.id;
        self.description = dump.description;
        self.default_language = dump.default_language;
        self.field_id_by_name = dump.field_id_by_name.into_iter().collect();

        for (field_name, serialized) in dump.fields {
            let field_id = match self.field_id_by_name.get(&field_name) {
                None => {
                    return Err(anyhow!(
                        "Field {} not found in field_id_by_name",
                        field_name
                    ));
                }
                Some(field_id) => *field_id,
            };
            let (value_type, collection_field): (ValueType, CollectionField) = match serialized {
                SerializedFieldIndexer::String(locale) => (
                    ValueType::Scalar(ScalarType::String),
                    CollectionField::new_string(
                        nlp_service.get(locale),
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                ),
                SerializedFieldIndexer::Number => (
                    ValueType::Scalar(ScalarType::Number),
                    CollectionField::new_number(self.id.clone(), field_id, field_name.clone()),
                ),
                SerializedFieldIndexer::Bool => (
                    ValueType::Scalar(ScalarType::Boolean),
                    CollectionField::new_bool(self.id.clone(), field_id, field_name.clone()),
                ),
                SerializedFieldIndexer::Embedding(model, fields) => (
                    ValueType::Complex(ComplexType::Embedding),
                    CollectionField::new_embedding(
                        model.0,
                        fields,
                        self.embedding_sender.clone(),
                        hooks_runtime.clone(),
                        self.id.clone(),
                        field_id,
                    ),
                ),
            };
            self.fields
                .insert(field_name, (value_type, collection_field));
        }
        self.collection_document_count
            .store(dump.document_count, std::sync::atomic::Ordering::Relaxed);
        self.field_id_generator = AtomicU16::new(dump.field_id_generator);

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
enum CollectionDump {
    #[serde(rename = "1")]
    V1(CollectionDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionDumpV1 {
    id: CollectionId,
    description: Option<String>,
    default_language: LanguageDTO,
    fields: Vec<(String, SerializedFieldIndexer)>,
    document_count: u64,
    field_id_generator: u16,
    field_id_by_name: Vec<(String, FieldId)>,
}
