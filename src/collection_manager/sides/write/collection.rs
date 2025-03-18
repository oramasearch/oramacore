use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU16, AtomicU64},
        Arc,
    },
};

use anyhow::{anyhow, bail, Context, Result};
use doc_id_storage::DocIdStorage;
use redact::Secret;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, trace, warn};

use crate::{
    collection_manager::{
        dto::{ApiKey, CollectionDTO, FieldId},
        sides::{
            hooks::{HookName, HooksRuntime},
            CollectionWriteOperation, DocumentFieldsWrapper, EmbeddingTypedFieldWrapper,
            OperationSender, TypedFieldWrapper, WriteOperation,
        },
    },
    file_utils::BufferedFile,
    metrics::{document_insertion::FIELD_CALCULATION_TIME, FieldCalculationLabels},
    nlp::{locales::Locale, NLPService, TextParser},
    types::{CollectionId, ComplexType, Document, DocumentId, ScalarType, ValueType},
};

use crate::collection_manager::dto::{LanguageDTO, TypedField};

use super::{
    embedding::EmbeddingCalculationRequest, CollectionFilterField, CollectionScoreField,
    SerializedFieldIndexer,
};

mod doc_id_storage;

pub const DEFAULT_EMBEDDING_FIELD_NAME: &str = "___orama_auto_embedding";

pub struct CollectionWriter {
    pub(super) id: CollectionId,
    description: Option<String>,
    default_language: LanguageDTO,
    filter_fields: RwLock<HashMap<FieldId, CollectionFilterField>>,
    score_fields: RwLock<HashMap<FieldId, CollectionScoreField>>,
    // fields: RwLock<HashMap<FieldId, (String, ValueType, CollectionField)>>,
    write_api_key: ApiKey,
    collection_document_count: AtomicU64,

    field_id_generator: AtomicU16,
    // field_id_by_name: RwLock<HashMap<String, FieldId>>,
    filter_field_id_by_name: RwLock<HashMap<String, FieldId>>,
    score_field_id_by_name: RwLock<HashMap<String, FieldId>>,

    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,

    doc_id_storage: RwLock<DocIdStorage>,
}

impl CollectionWriter {
    pub fn new(
        id: CollectionId,
        description: Option<String>,
        write_api_key: ApiKey,
        default_language: LanguageDTO,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    ) -> Self {
        Self {
            id: id.clone(),
            description,
            write_api_key,
            default_language,
            collection_document_count: Default::default(),
            filter_field_id_by_name: Default::default(),
            filter_fields: Default::default(),
            score_field_id_by_name: Default::default(),
            score_fields: Default::default(),
            field_id_generator: Default::default(),
            embedding_sender,
            doc_id_storage: Default::default(),
        }
    }

    pub fn check_write_api_key(&self, api_key: ApiKey) -> Result<()> {
        if self.write_api_key == api_key {
            Ok(())
        } else {
            Err(anyhow!("Invalid write api key"))
        }
    }

    pub async fn as_dto(&self) -> CollectionDTO {
        let filter_fields = self.filter_fields.read().await;
        let score_fields = self.score_fields.read().await;
        let fields = filter_fields
            .iter()
            .map(|(_, v)| (v.field_name(), v.field_type()))
            .chain(
                score_fields
                    .iter()
                    .map(|(_, v)| (v.field_name(), v.field_type())),
            )
            .collect();

        CollectionDTO {
            id: self.id.clone(),
            description: self.description.clone(),
            document_count: self
                .collection_document_count
                .load(std::sync::atomic::Ordering::Relaxed),
            fields,
        }
    }

    pub async fn get_document_ids(&self) -> Vec<DocumentId> {
        self.doc_id_storage
            .read()
            .await
            .get_document_ids()
            .collect()
    }

    pub async fn set_embedding_hook(&self, hook_name: HookName) -> Result<()> {
        let field_id_by_name = self.score_field_id_by_name.read().await;
        let field_id = field_id_by_name
            .get(DEFAULT_EMBEDDING_FIELD_NAME)
            .cloned()
            .context("Field for embedding not found")?;
        drop(field_id_by_name);

        let mut w = self.score_fields.write().await;
        let field = match w.get_mut(&field_id) {
            None => bail!("Field for embedding not found"),
            Some(field) => field,
        };
        field.set_embedding_hook(hook_name);

        Ok(())
    }

    pub async fn process_new_document(
        &self,
        doc_id: DocumentId,
        doc: Document,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<()> {
        // Those `?` is never triggered, but it's here to make the compiler happy:
        // The "id" property is always present in the document.
        // TODO: do this better
        let doc_id_str = doc
            .inner
            .get("id")
            .context("Document does not have an id")?
            .as_str()
            .context("Document id is not a string")?;
        let mut doc_id_storage = self.doc_id_storage.write().await;
        if !doc_id_storage.insert_document_id(doc_id_str.to_string(), doc_id) {
            // The document is already indexed.
            // If the document id is there, it will be difficul to remove it.
            // So, we decided to just ignore it.
            // We could at least return a warning to the user.
            // TODO: return a warning
            warn!("Document '{}' already indexed", doc_id_str);
            return Ok(());
        }
        drop(doc_id_storage);

        let fields_to_index = self
            .get_fields_to_index(doc.clone(), sender.clone(), hooks_runtime)
            .await
            .context("Cannot get fields to index")?;
        trace!("Fields to index: {:?}", fields_to_index);

        let flatten = doc.clone().into_flatten();

        let filter_fields = self.filter_fields.read().await;
        for field_id in &fields_to_index {
            let field = match filter_fields.get(field_id) {
                None => continue,
                Some(v) => v,
            };
            let field_name = field.field_name();

            let metric = FIELD_CALCULATION_TIME.create(FieldCalculationLabels {
                collection: self.id.0.clone().into(),
                field: field_name.clone().into(),
                field_type: field.field_type_str().into(),
            });
            field
                .get_write_operations(doc_id, &flatten, sender.clone())
                .await
                .with_context(|| format!("Cannot index field {}", field_name))?;
            drop(metric);
        }

        let score_fields = self.score_fields.read().await;
        for field_id in &fields_to_index {
            let field = match score_fields.get(field_id) {
                None => continue,
                Some(v) => v,
            };
            let field_name = field.field_name();

            let metric = FIELD_CALCULATION_TIME.create(FieldCalculationLabels {
                collection: self.id.0.clone().into(),
                field: field_name.clone().into(),
                field_type: field.field_type_str().into(),
            });
            field
                .get_write_operations(doc_id, &flatten, sender.clone())
                .await
                .with_context(|| format!("Cannot index field {}", field_name))?;
            drop(metric);
        }

        self.collection_document_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        trace!("Document field indexed");

        Ok(())
    }

    fn value_to_typed_field(&self, value_type: ValueType) -> Option<TypedField> {
        match value_type {
            ValueType::Scalar(ScalarType::String) => {
                Some(TypedField::Text(self.default_language.into()))
            }
            ValueType::Scalar(ScalarType::Number) => Some(TypedField::Number),
            ValueType::Scalar(ScalarType::Boolean) => Some(TypedField::Bool),
            ValueType::Complex(ComplexType::Array(ScalarType::String)) => {
                Some(TypedField::ArrayText(self.default_language.into()))
            }
            ValueType::Complex(ComplexType::Array(ScalarType::Number)) => {
                Some(TypedField::ArrayNumber)
            }
            ValueType::Complex(ComplexType::Array(ScalarType::Boolean)) => {
                Some(TypedField::ArrayBoolean)
            }
            _ => None, // @todo: support other types
        }
    }

    fn get_text_parser(&self, locale: Locale) -> Arc<TextParser> {
        // TextParser is expensive to create, so we cache it
        // TODO: add a cache
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
            debug!(
                "Registering field {} with type {:?}",
                field_name, field_type
            );

            // Avoid creating fields that already exists
            if self
                .filter_field_id_by_name
                .read()
                .await
                .contains_key(&field_name)
            {
                continue;
            }
            if self
                .score_field_id_by_name
                .read()
                .await
                .contains_key(&field_name)
            {
                continue;
            }

            self.create_field(
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

    #[instrument(skip(self, sender, embedding_sender, hooks_runtime))]
    async fn create_field(
        &self,
        field_name: String,
        typed_field: TypedField,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<Vec<FieldId>> {
        // We don't index the "id" field at all.
        if field_name == "id" {
            return Ok(vec![]);
        }

        let mut added_fields = vec![];
        let mut create_new_id = async |field_id_generator: &AtomicU16,
                                       map: &RwLock<HashMap<String, FieldId>>,
                                       field_name: String|
               -> FieldId {
            let field_id = field_id_generator.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let field_id = FieldId(field_id);
            let mut field_id_by_name = map.write().await;
            field_id_by_name.insert(field_name, field_id);
            drop(field_id_by_name);

            added_fields.push(field_id);

            field_id
        };

        async fn send(
            sender: OperationSender,
            collection_id: CollectionId,
            field_id: FieldId,
            field_name: String,
            field: TypedFieldWrapper,
        ) -> Result<()> {
            sender
                .send(WriteOperation::Collection(
                    collection_id,
                    CollectionWriteOperation::CreateField {
                        field_id,
                        field_name,
                        field,
                    },
                ))
                .await
                .context("Cannot sent creation field")?;

            Ok(())
        }

        match typed_field {
            TypedField::Embedding(embedding_field) => {
                let mut lock = self.score_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.score_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionScoreField::new_embedding(
                        embedding_field.model.0,
                        embedding_field.document_fields.clone(),
                        embedding_sender,
                        hooks_runtime,
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::Embedding(EmbeddingTypedFieldWrapper {
                        model: embedding_field.model,
                        document_fields: DocumentFieldsWrapper(embedding_field.document_fields),
                    }),
                )
                .await?;
                drop(lock);
            }
            TypedField::Text(locale) => {
                let parser = self.get_text_parser(locale);
                let mut lock = self.score_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.score_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionScoreField::new_string(
                        parser,
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender.clone(),
                    self.id.clone(),
                    field_id,
                    field_name.clone(),
                    TypedFieldWrapper::Text(locale),
                )
                .await?;
                drop(lock);

                let mut lock = self.filter_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.filter_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionFilterField::new_string(
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::String,
                )
                .await?;
                drop(lock);
            }
            TypedField::ArrayText(locale) => {
                let parser = self.get_text_parser(locale);
                let mut lock = self.score_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.score_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionScoreField::new_arr_string(
                        parser,
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender.clone(),
                    self.id.clone(),
                    field_id,
                    field_name.clone(),
                    TypedFieldWrapper::ArrayText(locale),
                )
                .await?;
                drop(lock);

                let mut lock = self.filter_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.filter_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionFilterField::new_string(
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::String,
                )
                .await?;
                drop(lock);
            }
            TypedField::Number => {
                let mut lock = self.filter_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.filter_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionFilterField::new_number(
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::Number,
                )
                .await?;
                drop(lock);
            }
            TypedField::ArrayNumber => {
                let mut lock = self.filter_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.filter_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionFilterField::new_arr_number(
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::ArrayNumber,
                )
                .await?;
                drop(lock);
            }
            TypedField::Bool => {
                let mut lock = self.filter_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.filter_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionFilterField::new_bool(self.id.clone(), field_id, field_name.clone()),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::Bool,
                )
                .await?;
                drop(lock);
            }
            TypedField::ArrayBoolean => {
                let mut lock = self.filter_fields.write().await;
                let field_id = create_new_id(
                    &self.field_id_generator,
                    &self.filter_field_id_by_name,
                    field_name.clone(),
                )
                .await;
                lock.insert(
                    field_id,
                    CollectionFilterField::new_arr_bool(
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    ),
                );
                send(
                    sender,
                    self.id.clone(),
                    field_id,
                    field_name,
                    TypedFieldWrapper::ArrayBoolean,
                )
                .await?;
                drop(lock);
            }
        }

        info!("Field created");

        Ok(added_fields)
    }

    #[instrument(skip(self, doc, sender, hooks_runtime))]
    async fn get_fields_to_index(
        &self,
        doc: Document,
        sender: OperationSender,
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<Vec<FieldId>> {
        let flatten = doc.clone().into_flatten();
        let schema = flatten.get_field_schema();

        let mut field_ids = vec![];

        // `DEFAULT_EMBEDDING_FIELD_NAME` doesn't appear in the document, but it's always indexed
        let field_id_by_name = self.score_field_id_by_name.read().await;
        if let Some(field_id) = field_id_by_name.get(DEFAULT_EMBEDDING_FIELD_NAME) {
            field_ids.push(*field_id);
        }
        drop(field_id_by_name);

        for (field_name, value_type) in schema {
            let field_id_by_name = self.score_field_id_by_name.read().await;
            let score_field_id = field_id_by_name.get(&field_name).cloned();
            drop(field_id_by_name);

            let field_id_by_name = self.filter_field_id_by_name.read().await;
            let filter_field_id = field_id_by_name.get(&field_name).cloned();
            drop(field_id_by_name);

            let mut is_new = true;
            if let Some(score_field_id) = score_field_id {
                field_ids.push(score_field_id);
                is_new = false;
            }
            if let Some(filter_field_id) = filter_field_id {
                field_ids.push(filter_field_id);
                is_new = false;
            }

            if !is_new {
                continue;
            }

            // @todo: add support to other types
            if let Some(typed_field) = self.value_to_typed_field(value_type.clone()) {
                let fields = self
                    .create_field(
                        field_name,
                        typed_field,
                        self.embedding_sender.clone(),
                        sender.clone(),
                        hooks_runtime.clone(),
                    )
                    .await
                    .context("Cannot create field")?;
                field_ids.extend(fields);
            } else {
                warn!("Field type not supported: {:?}", value_type);
            }
        }

        Ok(field_ids)
    }

    pub async fn delete_documents(
        &self,
        doc_ids: Vec<String>,
        sender: OperationSender,
    ) -> Result<()> {
        let doc_ids = self
            .doc_id_storage
            .write()
            .await
            .remove_document_id(doc_ids);
        info!(coll_id= ?self.id, ?doc_ids, "Deleting documents");

        let doc_ids_len = doc_ids.len();

        sender
            .send(WriteOperation::Collection(
                self.id.clone(),
                CollectionWriteOperation::DeleteDocuments { doc_ids },
            ))
            .await?;

        self.collection_document_count
            .fetch_sub(doc_ids_len as u64, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    pub async fn remove_from_fs(self, path: PathBuf) {
        match std::fs::remove_dir_all(&path) {
            Ok(_) => {}
            Err(e) => {
                warn!(coll_id= ?self.id, "Cannot remove collection directory. Ignored: {:?}", e);
            }
        };
    }

    pub async fn commit(&self, path: PathBuf) -> Result<()> {
        info!(coll_id= ?self.id, "Committing collection");

        std::fs::create_dir_all(&path).context("Cannot create collection directory")?;

        let doc_id_storage_path = path.join("doc_id_storage");
        self.doc_id_storage
            .read()
            .await
            .commit(doc_id_storage_path.clone())
            .context("Cannot commit doc_id_storage")?;

        let dump = CollectionDump::V2(CollectionDumpV2 {
            id: self.id.clone(),
            description: self.description.clone(),
            write_api_key: self.write_api_key.0.expose_secret().clone(),
            default_language: self.default_language,
            document_count: self
                .collection_document_count
                .load(std::sync::atomic::Ordering::Relaxed),
            field_id_generator: self
                .field_id_generator
                .load(std::sync::atomic::Ordering::Relaxed),
            filter_field_id_by_name: self
                .filter_field_id_by_name
                .read()
                .await
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            filter_fields: self
                .filter_fields
                .read()
                .await
                .iter()
                .map(|(k, v)| (*k, v.serialized()))
                .collect(),
            score_field_id_by_name: self
                .score_field_id_by_name
                .read()
                .await
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            score_fields: self
                .score_fields
                .read()
                .await
                .iter()
                .map(|(k, v)| (*k, v.serialized()))
                .collect(),
            doc_id_storage_path,
        });

        BufferedFile::create_or_overwrite(path.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

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

        let dump: CollectionDumpV2 = match dump {
            CollectionDump::V1(dump) => {
                let mut score_field_id_by_name = vec![];
                let mut score_fields = vec![];
                let mut filter_field_id_by_name = vec![];
                let mut filter_fields = vec![];
                for (field_name, indexer) in dump.fields {
                    let field_id = dump
                        .field_id_by_name
                        .iter()
                        .find(|(n, _)| n == &field_name)
                        .unwrap()
                        .1;
                    match indexer {
                        SerializedFieldIndexer::Number => {
                            filter_field_id_by_name.push((field_name.clone(), field_id));
                            filter_fields.push((field_id, indexer))
                        }
                        SerializedFieldIndexer::Bool => {
                            filter_field_id_by_name.push((field_name.clone(), field_id));
                            filter_fields.push((field_id, indexer))
                        }
                        // This never happens, but it's here to make the compiler happy
                        SerializedFieldIndexer::StringFilter => {
                            filter_field_id_by_name.push((field_name.clone(), field_id));
                            filter_fields.push((field_id, indexer))
                        }
                        SerializedFieldIndexer::Embedding(_, _) => {
                            score_field_id_by_name.push((field_name.clone(), field_id));
                            score_fields.push((field_id, indexer))
                        }
                        SerializedFieldIndexer::String(_) => {
                            score_field_id_by_name.push((field_name.clone(), field_id));
                            score_fields.push((field_id, indexer))
                        }
                    }
                }
                CollectionDumpV2 {
                    id: dump.id,
                    description: dump.description,
                    write_api_key: dump.write_api_key,
                    default_language: dump.default_language,
                    document_count: dump.document_count,
                    field_id_generator: dump.field_id_generator,
                    score_fields,
                    filter_fields,
                    score_field_id_by_name,
                    filter_field_id_by_name: dump.field_id_by_name,
                    doc_id_storage_path: dump.doc_id_storage_path,
                }
            }
            CollectionDump::V2(dump) => dump,
        };

        self.id = dump.id;
        self.description = dump.description;
        self.write_api_key = ApiKey(Secret::new(dump.write_api_key));
        self.default_language = dump.default_language;
        self.doc_id_storage = RwLock::new(DocIdStorage::load(dump.doc_id_storage_path)?);

        for (field_id, serialized) in dump.filter_fields {
            let field_name = dump
                .filter_field_id_by_name
                .iter()
                .find(|(_, id)| id == &field_id)
                .with_context(|| format!("Field id not found: {:?}", field_id))?
                .0
                .clone();

            let collection_field: CollectionFilterField = match serialized {
                SerializedFieldIndexer::Number => {
                    CollectionFilterField::new_number(self.id.clone(), field_id, field_name.clone())
                }
                SerializedFieldIndexer::Bool => {
                    CollectionFilterField::new_bool(self.id.clone(), field_id, field_name.clone())
                }
                SerializedFieldIndexer::StringFilter => {
                    CollectionFilterField::new_string(self.id.clone(), field_id, field_name.clone())
                }
                SerializedFieldIndexer::Embedding(_, _) => {
                    return Err(anyhow!("Embedding field not supported"))
                }
                SerializedFieldIndexer::String(_) => {
                    return Err(anyhow!("String field not supported"))
                }
            };
            let mut w = self.filter_fields.write().await;
            w.insert(field_id, collection_field);
        }

        for (field_id, serialized) in dump.score_fields {
            let field_name = dump
                .score_field_id_by_name
                .iter()
                .find(|(_, id)| id == &field_id)
                .with_context(|| format!("Field id not found: {:?}", field_id))?
                .0
                .clone();

            let collection_field: CollectionScoreField = match serialized {
                SerializedFieldIndexer::String(locale) => CollectionScoreField::new_string(
                    nlp_service.get(locale),
                    self.id.clone(),
                    field_id,
                    field_name.clone(),
                ),
                SerializedFieldIndexer::Embedding(model, fields) => {
                    CollectionScoreField::new_embedding(
                        model.0,
                        fields,
                        self.embedding_sender.clone(),
                        hooks_runtime.clone(),
                        self.id.clone(),
                        field_id,
                        field_name.clone(),
                    )
                }
                SerializedFieldIndexer::Number => {
                    return Err(anyhow!("Number field not supported"))
                }
                SerializedFieldIndexer::Bool => return Err(anyhow!("Bool field not supported")),
                SerializedFieldIndexer::StringFilter => {
                    return Err(anyhow!("String filter field not supported"))
                }
            };
            let mut w = self.score_fields.write().await;
            w.insert(field_id, collection_field);
        }

        self.filter_field_id_by_name =
            RwLock::new(dump.filter_field_id_by_name.into_iter().collect());
        self.score_field_id_by_name =
            RwLock::new(dump.score_field_id_by_name.into_iter().collect());

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
    #[serde(rename = "2")]
    V2(CollectionDumpV2),
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionDumpV1 {
    id: CollectionId,
    description: Option<String>,
    write_api_key: String,
    default_language: LanguageDTO,
    fields: Vec<(String, SerializedFieldIndexer)>,
    document_count: u64,
    field_id_generator: u16,
    field_id_by_name: Vec<(String, FieldId)>,
    doc_id_storage_path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionDumpV2 {
    id: CollectionId,
    description: Option<String>,
    write_api_key: String,
    default_language: LanguageDTO,
    document_count: u64,
    field_id_generator: u16,
    score_fields: Vec<(FieldId, SerializedFieldIndexer)>,
    filter_fields: Vec<(FieldId, SerializedFieldIndexer)>,
    score_field_id_by_name: Vec<(String, FieldId)>,
    filter_field_id_by_name: Vec<(String, FieldId)>,
    doc_id_storage_path: PathBuf,
}
