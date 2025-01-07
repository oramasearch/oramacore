use std::sync::{atomic::AtomicU64, Arc};

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;

use crate::{
    collection_manager::{
        dto::{FieldId, TypedField},
        sides::document_storage::DocumentStorage,
        CollectionId,
    },
    embeddings::{EmbeddingService, LoadedModel},
    indexes::{
        bool::BoolIndex,
        number::{NumberIndex, NumberIndexConfig},
        string::{StringIndex, StringIndexConfig},
        vector::{VectorIndex, VectorIndexConfig},
    },
};

use super::{commit::CollectionDescriptorDump, IndexesConfig};

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
        indexes_config: IndexesConfig,
    ) -> Result<Self> {
        let collection_data_dir = indexes_config.data_dir.join(&id.0);

        let vector_index = VectorIndex::try_new(VectorIndexConfig {
            base_path: collection_data_dir.join("vectors"),
        })
        .context("Cannot create vector index during collection creation")?;

        let mut string_index = StringIndex::new(StringIndexConfig {
            // posting_id_generator,
            // base_path: collection_data_dir.join("strings"),
        });
        string_index
            .load(collection_data_dir.join("strings"))
            .context("Cannot load string index")?;

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

    pub(super) fn get_collection_descriptor_dump(&self) -> Result<CollectionDescriptorDump> {
        Ok(CollectionDescriptorDump {
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
        })
    }
}

impl CollectionReader {}
