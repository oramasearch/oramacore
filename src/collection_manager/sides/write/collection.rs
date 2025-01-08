use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicU16, AtomicU64},
        Arc,
    },
};

use anyhow::{anyhow, Context, Ok, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast::Sender;
use tracing::info;

use crate::{
    collection_manager::dto::{CollectionDTO, FieldId},
    embeddings::EmbeddingService,
    file_utils::BufferedFile,
    nlp::{locales::Locale, TextParser},
    types::{CollectionId, ComplexType, Document, DocumentId, ScalarType, ValueType},
};

use crate::collection_manager::dto::{LanguageDTO, TypedField};

use super::{
    fields::{BoolField, EmbeddingField, FieldIndexer, FieldsToIndex, NumberField, StringField},
    CollectionWriteOperation, WriteOperation,
};

pub struct CollectionWriter {
    id: CollectionId,
    description: Option<String>,
    default_language: LanguageDTO,
    fields: DashMap<String, (ValueType, Arc<Box<dyn FieldIndexer>>)>,

    document_count: AtomicU64,

    field_id_generator: AtomicU16,
    field_id_by_name: DashMap<String, FieldId>,
}

impl CollectionWriter {
    pub fn new(
        id: CollectionId,
        description: Option<String>,
        default_language: LanguageDTO,
    ) -> Self {
        Self {
            id: id.clone(),
            description,
            default_language,
            document_count: Default::default(),
            fields: Default::default(),
            field_id_by_name: DashMap::new(),
            field_id_generator: AtomicU16::new(0),
        }
    }

    pub fn as_dto(&self) -> CollectionDTO {
        CollectionDTO {
            id: self.id.clone(),
            description: self.description.clone(),
            document_count: self
                .document_count
                .load(std::sync::atomic::Ordering::Relaxed),
            fields: self
                .fields
                .iter()
                .map(|e| (e.key().clone(), e.value().0.clone()))
                .collect(),
        }
    }

    pub(super) async fn process_new_document(
        &self,
        doc_id: DocumentId,
        doc: Document,
        embedding_service: Arc<EmbeddingService>,
        sender: &Sender<WriteOperation>,
    ) -> Result<()> {
        self.document_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        sender
            .send(WriteOperation::Collection(
                self.id.clone(),
                CollectionWriteOperation::InsertDocument {
                    doc_id,
                    doc: doc.clone(),
                },
            ))
            .map_err(|e| anyhow!("Error sending document to index writer: {:?}", e))?;

        let fields_to_index = self
            .get_fields_to_index(doc.clone(), embedding_service.clone(), sender)
            .await
            .context("Cannot get fields to index")?;

        let flatten = doc.clone().into_flatten();
        for entry in fields_to_index.iter() {
            let (field_name, (_, field)) = entry.pair();

            let field_id = self.get_field_id_by_name(field_name);

            let write_operations = field.get_write_operations(
                self.id.clone(),
                doc_id,
                field_name,
                field_id,
                &flatten,
            )?;
            for write_operation in write_operations {
                sender
                    .send(write_operation)
                    .map_err(|e| anyhow!("Error sending document to index writer: {:?}", e))?;
            }
        }

        Ok(())
    }

    pub(super) fn get_field_id_by_name(&self, name: &str) -> FieldId {
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

    pub(super) fn value_to_typed_field(&self, value_type: ValueType) -> TypedField {
        match value_type {
            ValueType::Scalar(ScalarType::String) => TypedField::Text(self.default_language),
            ValueType::Scalar(ScalarType::Number) => TypedField::Number,
            ValueType::Scalar(ScalarType::Boolean) => TypedField::Bool,
            x => unimplemented!("Field type not implemented yet {:?}", x),
        }
    }

    fn get_text_parser(&self, language: LanguageDTO) -> Arc<TextParser> {
        let locale: Locale = language.into();
        let parser = TextParser::from_language(locale);
        Arc::new(parser)
    }

    pub(super) async fn create_field(
        &self,
        field_id: FieldId,
        field_name: String,
        typed_field: TypedField,
        embedding_service: Arc<EmbeddingService>,
        sender: &Sender<WriteOperation>,
    ) -> Result<()> {
        match &typed_field {
            TypedField::Embedding(embedding_field) => {
                let model = embedding_service
                    .get_model(embedding_field.model_name.clone())
                    .await?;
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Complex(ComplexType::Embedding),
                        Arc::new(Box::new(EmbeddingField::new(
                            model,
                            embedding_field.document_fields.clone(),
                        ))),
                    ),
                );
            }
            TypedField::Text(language) => {
                let parser = self.get_text_parser(*language);
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Scalar(ScalarType::String),
                        Arc::new(Box::new(StringField::new(parser))),
                    ),
                );
            }
            TypedField::Number => {
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Scalar(ScalarType::Number),
                        Arc::new(Box::new(NumberField::new())),
                    ),
                );
            }
            TypedField::Bool => {
                self.fields.insert(
                    field_name.clone(),
                    (
                        ValueType::Scalar(ScalarType::Boolean),
                        Arc::new(Box::new(BoolField::new())),
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
            .context("Cannot sent creation field")?;

        Ok(())
    }

    pub(super) async fn get_fields_to_index(
        &self,
        doc: Document,
        embedding_service: Arc<EmbeddingService>,
        sender: &Sender<WriteOperation>,
    ) -> Result<FieldsToIndex> {
        let flatten = doc.clone().into_flatten();
        let schema = flatten.get_field_schema();

        for (field_name, value_type) in schema {
            if self.fields.contains_key(&field_name) {
                continue;
            }

            let field_id = self.get_field_id_by_name(&field_name);

            let typed_field = self.value_to_typed_field(value_type);

            self.create_field(
                field_id,
                field_name,
                typed_field,
                embedding_service.clone(),
                sender,
            )
            .await
            .context("Cannot create field")?;
        }

        Ok(self.fields.clone())
    }

    pub(super) fn commit(&mut self, path: PathBuf) -> Result<()> {
        // `&mut self` is not used, but it needed to ensure no other thread is using the collection

        let dump = CollectionDump {
            id: self.id.clone(),
            description: self.description.clone(),
            default_language: self.default_language,
            fields: self
                .fields
                .iter()
                .map(|e| (e.key().clone(), e.value().0.clone()))
                .collect(),
            document_count: self
                .document_count
                .load(std::sync::atomic::Ordering::Relaxed),
            field_id_generator: self
                .field_id_generator
                .load(std::sync::atomic::Ordering::Relaxed),
            field_id_by_name: self
                .field_id_by_name
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
        };

        BufferedFile::create(path.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        Ok(())
    }

    pub(super) fn load(&mut self, path: PathBuf) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionDump {
    id: CollectionId,
    description: Option<String>,
    default_language: LanguageDTO,
    fields: Vec<(String, ValueType)>,
    document_count: u64,
    field_id_generator: u16,
    field_id_by_name: Vec<(String, FieldId)>,
}
