use std::{path::PathBuf, sync::Arc};

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::{
        dto::{FieldId, TypedField},
        sides::document_storage::DocumentStorage,
    },
    embeddings::{EmbeddingService, LoadedModel},
    file_utils::BufferedFile,
    indexes::{
        bool::BoolIndex,
        number::{NumberIndex, NumberIndexConfig},
        string::{StringIndex, StringIndexConfig},
        vector::{VectorIndex, VectorIndexConfig},
    },
    nlp::{locales::Locale, NLPService, TextParser},
    types::CollectionId,
};

use super::IndexesConfig;

#[derive(Debug)]
pub struct CollectionReader {
    pub(super) id: CollectionId,
    pub(super) embedding_service: Arc<EmbeddingService>,
    pub(super) nlp_service: Arc<NLPService>,

    pub(super) document_storage: Arc<dyn DocumentStorage>,

    pub(super) fields: DashMap<String, (FieldId, TypedField)>,

    // indexes
    pub(super) vector_index: VectorIndex,
    pub(super) fields_per_model: DashMap<Arc<LoadedModel>, Vec<FieldId>>,

    pub(super) string_index: StringIndex,
    pub(super) text_parser_per_field: DashMap<FieldId, (Locale, Arc<TextParser>)>,

    pub(super) number_index: NumberIndex,
    pub(super) bool_index: BoolIndex,
    // TODO: textparser -> vec<field_id>
}

impl CollectionReader {
    pub fn try_new(
        id: CollectionId,
        embedding_service: Arc<EmbeddingService>,
        nlp_service: Arc<NLPService>,
        document_storage: Arc<dyn DocumentStorage>,
        _: IndexesConfig,
    ) -> Result<Self> {
        let vector_index = VectorIndex::try_new(VectorIndexConfig {})
            .context("Cannot create vector index during collection creation")?;

        let string_index = StringIndex::new(StringIndexConfig {});

        let number_index = NumberIndex::try_new(NumberIndexConfig {})
            .context("Cannot create number index during collection creation")?;

        let bool_index = BoolIndex::new();

        Ok(Self {
            id,
            embedding_service,
            nlp_service,

            document_storage,

            vector_index,
            fields_per_model: Default::default(),

            string_index,
            text_parser_per_field: Default::default(),

            number_index,

            bool_index,

            fields: Default::default(),
        })
    }

    pub async fn get_total_documents(&self) -> Result<usize> {
        self.document_storage.get_total_documents().await
    }

    pub(super) fn get_field_id(&self, field_name: String) -> Result<FieldId> {
        let field_id = self.fields.get(&field_name);

        match field_id {
            Some(field_id) => Ok(field_id.0),
            None => Err(anyhow!("Field not found")),
        }
    }

    pub(super) fn get_field_id_with_type(&self, field_name: &str) -> Result<(FieldId, TypedField)> {
        self.fields
            .get(field_name)
            .map(|v| v.clone())
            .ok_or_else(|| anyhow!("Field not found"))
    }

    pub async fn load(&mut self, collection_data_dir: PathBuf) -> Result<()> {
        self.string_index
            .load(collection_data_dir.join("strings"))
            .context("Cannot load string index")?;
        self.number_index
            .load(collection_data_dir.join("numbers"))
            .context("Cannot load number index")?;
        self.vector_index
            .load(collection_data_dir.join("vectors"))
            .context("Cannot load vectors index")?;

        let coll_desc_file_path = collection_data_dir.join("desc.json");
        let dump: CollectionDescriptorDump = BufferedFile::open(coll_desc_file_path)
            .context("Cannot open collection file")?
            .read_json_data()
            .with_context(|| {
                format!("Cannot deserialize collection descriptor for {:?}", self.id)
            })?;
        for (field_name, (field_id, field_type)) in dump.fields {
            self.fields.insert(field_name, (field_id, field_type));
        }

        for (orama_model, fields) in dump.used_models {
            let model = self
                .embedding_service
                .get_model(orama_model.clone())
                .await
                .context("Model not found")?;
            self.fields_per_model.insert(model, fields);
        }

        self.text_parser_per_field = self.fields
            .iter()
            .filter_map(|e| {
                if let TypedField::Text(l) = e.1 {
                    let locale = l.into();
                    Some(
                        (
                            e.0,
                            (
                                locale,
                                self.nlp_service.get(locale)
                            )
                        )
                    )
                } else {
                    None
                }
            })
            .collect();

        Ok(())
    }

    pub fn commit(&self, commit_config: CommitConfig) -> Result<()> {
        self.string_index
            .commit(commit_config.folder_to_commit.join("strings"))
            .context("Cannot commit string index")?;
        self.number_index
            .commit(commit_config.folder_to_commit.join("numbers"))
            .context("Cannot commit number index")?;
        self.vector_index
            .commit(commit_config.folder_to_commit.join("vectors"))
            .context("Cannot commit vectors index")?;

        let dump = CollectionDescriptorDump {
            id: self.id.clone(),
            fields: self
                .fields
                .iter()
                .map(|v| {
                    let (field_name, (field_id, typed_field)) = v.pair();
                    (field_name.clone(), (*field_id, typed_field.clone()))
                })
                .collect(),
            used_models: self
                .fields_per_model
                .iter()
                .map(|v| {
                    let (model, field_ids) = v.pair();
                    (model.model_name(), field_ids.clone())
                })
                .collect(),
        };

        let coll_desc_file_path = commit_config.folder_to_commit.join("desc.json");
        BufferedFile::create(coll_desc_file_path)
            .context("Cannot create desc.json file")?
            .write_json_data(&dump)
            .with_context(|| format!("Cannot serialize collection descriptor for {:?}", self.id))?;

        Ok(())
    }
}

pub struct CommitConfig {
    pub folder_to_commit: PathBuf,
    pub epoch: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Committed {
    pub epoch: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionDescriptorDump {
    pub id: CollectionId,
    pub fields: Vec<(String, (FieldId, TypedField))>,
    pub used_models: Vec<(String, Vec<FieldId>)>,
}
