use anyhow::Result;
use tracing::instrument;

use crate::{
    collection_manager::{
        dto::{FieldId, TypedField},
        sides::write::InsertStringTerms,
    },
    indexes::number::Number,
    types::{Document, DocumentId},
};

use super::CollectionReader;

impl CollectionReader {
    pub async fn create_field(
        &self,
        field_id: FieldId,
        field_name: String,
        field: TypedField,
    ) -> Result<()> {
        self.fields
            .insert(field_name.clone(), (field_id, field.clone()));

        if let TypedField::Embedding(embedding) = field {
            let orama_model = self
                .embedding_service
                .get_model(embedding.model_name)
                .await?;

            self.vector_index
                .add_field(field_id, orama_model.dimensions())?;

            self.fields_per_model
                .entry(orama_model)
                .or_default()
                .push(field_id);
        };

        Ok(())
    }

    #[instrument(skip(self, value), level="debug", fields(self.id = ?self.id))]
    pub fn index_embedding(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        value: Vec<f32>,
    ) -> Result<()> {
        // `insert_batch` is designed to process multiple values at once
        // We are inserting only one value, and this is not good for performance
        // We should add an API to accept a single value and avoid the rebuild step
        // Instead, we could move the "rebuild" logic to the `VectorIndex`
        // TODO: do it.
        self.vector_index
            .insert_batch(vec![(doc_id, field_id, vec![value])])
    }

    #[instrument(skip(self, terms), level="debug", fields(self.id = ?self.id))]
    pub fn index_string(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        field_length: u16,
        terms: InsertStringTerms,
    ) -> Result<()> {
        self.string_index
            .insert(doc_id, field_id, field_length, terms)?;
        Ok(())
    }

    #[instrument(skip(self, value), level="debug", fields(self.id = ?self.id))]
    pub fn index_number(&self, doc_id: DocumentId, field_id: FieldId, value: Number) -> Result<()> {
        self.number_index.add(doc_id, field_id, value)?;
        Ok(())
    }

    #[instrument(skip(self, value), level="debug", fields(self.id = ?self.id))]
    pub fn index_boolean(&self, doc_id: DocumentId, field_id: FieldId, value: bool) -> Result<()> {
        self.bool_index.add(doc_id, field_id, value);
        Ok(())
    }

    #[instrument(skip(self), level="debug", fields(self.id = ?self.id))]
    pub async fn insert_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        self.document_storage.add_document(doc_id, doc).await
    }
}
