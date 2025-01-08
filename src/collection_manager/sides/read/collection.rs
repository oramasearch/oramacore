use std::{io::Write, path::PathBuf, sync::Arc};

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::{
        dto::{FieldId, TypedField},
        sides::document_storage::DocumentStorage,
    },
    embeddings::{EmbeddingService, LoadedModel},
    indexes::{
        bool::BoolIndex,
        number::{NumberIndex, NumberIndexConfig},
        string::{StringIndex, StringIndexConfig},
        vector::{VectorIndex, VectorIndexConfig},
    },
    types::CollectionId,
};

use super::IndexesConfig;

#[derive(Debug)]
pub struct CollectionReader {
    pub(super) id: CollectionId,
    pub(super) embedding_service: Arc<EmbeddingService>,

    pub(super) document_storage: Arc<dyn DocumentStorage>,

    pub(super) fields: DashMap<String, (FieldId, TypedField)>,

    // indexes
    pub(super) vector_index: VectorIndex,
    pub(super) fields_per_model: DashMap<Arc<LoadedModel>, Vec<FieldId>>,

    pub(super) string_index: StringIndex,
    pub(super) number_index: NumberIndex,
    pub(super) bool_index: BoolIndex,
    // TODO: textparser -> vec<field_id>
}

impl CollectionReader {
    pub fn try_new(
        id: CollectionId,
        embedding_service: Arc<EmbeddingService>,
        document_storage: Arc<dyn DocumentStorage>,
        _: IndexesConfig,
    ) -> Result<Self> {
        // let collection_data_dir = indexes_config.data_dir.join(&id.0);

        let vector_index = VectorIndex::try_new(VectorIndexConfig {
            // FIX ME!!
            base_path: PathBuf::new(),
        })
        .context("Cannot create vector index during collection creation")?;

        let string_index = StringIndex::new(StringIndexConfig {});

        let number_index = NumberIndex::try_new(NumberIndexConfig {})?;

        let bool_index = BoolIndex::new();

        Ok(Self {
            id,
            embedding_service,
            document_storage,

            vector_index,
            fields_per_model: Default::default(),

            string_index,

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

    pub fn load(&mut self, collection_data_dir: PathBuf) -> Result<()> {
        self.string_index
            .load(collection_data_dir.join("strings"))
            .context("Cannot load string index")?;

        let coll_desc_file_path = collection_data_dir.join("desc.json");
        let coll_desc_file = std::fs::File::open(&coll_desc_file_path).with_context(|| {
            format!(
                "Cannot open file for collection {:?} at {:?}",
                self.id, coll_desc_file_path
            )
        })?;
        let dump: CollectionDescriptorDump =
            serde_json::from_reader(coll_desc_file).with_context(|| {
                format!(
                    "Cannot deserialize collection descriptor for {:?} to file {:?}",
                    self.id, coll_desc_file_path
                )
            })?;
        for (field_name, (field_id, field_type)) in dump.fields {
            self.fields.insert(field_name, (field_id, field_type));
        }

        Ok(())
    }

    pub fn commit(&self, commit_config: CommitConfig) -> Result<()> {
        self.string_index
            .commit(commit_config.folder_to_commit.join("strings"))?;

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
                    (model.model_name().to_string(), field_ids.clone())
                })
                .collect(),
        };

        let coll_desc_file_path = commit_config.folder_to_commit.join("desc.json");
        let mut coll_desc_file =
            std::fs::File::create(&coll_desc_file_path).with_context(|| {
                format!(
                    "Cannot create file for collection {:?} at {:?}",
                    self.id, coll_desc_file_path
                )
            })?;
        serde_json::to_writer(&mut coll_desc_file, &dump).with_context(|| {
            format!(
                "Cannot serialize collection descriptor for {:?} to file {:?}",
                self.id, coll_desc_file_path
            )
        })?;
        coll_desc_file.flush().with_context(|| {
            format!(
                "Cannot flush collection descriptor for {:?} to file {:?}",
                self.id, coll_desc_file_path
            )
        })?;
        coll_desc_file.sync_data().with_context(|| {
            format!(
                "Cannot sync collection descriptor for {:?} to file {:?}",
                self.id, coll_desc_file_path
            )
        })?;

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
