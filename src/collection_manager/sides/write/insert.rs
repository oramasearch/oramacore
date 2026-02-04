//! Insert documents operation.
//!
//! This module provides the `InsertDocuments` struct which handles document insertion
//! following the same pattern as the read side's `Search` struct.

use std::borrow::Cow;
use std::collections::HashSet;

use anyhow::{Context, Result};
use tracing::{debug, info, trace};

use crate::collection_manager::sides::write::collection_document_storage::CollectionDocumentStorage;
use crate::collection_manager::sides::write::document_storage::ZeboDocument;
use crate::collection_manager::sides::{
    CollectionWriteOperation, DocumentStorageWriteOperation, DocumentToInsert, WriteOperation,
};
use crate::types::{
    CollectionId, DocumentId, DocumentList, IndexId, InsertDocumentsResult, WriteApiKey,
};

use super::helpers::{extract_or_generate_document_id, WriteOperationContext, WriteYieldGuard};
use super::WriteError;

/// Request to insert documents into an index.
pub struct InsertDocumentsRequest {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub documents: DocumentList,
    pub write_api_key: WriteApiKey,
}

/// Operation for inserting documents into an index.
///
/// This struct follows the same pattern as the read side's `Search` struct:
/// - Created with `new()` for validation and constraint checks
/// - Executed with `execute()` to perform the two-phase insertion
///
/// The insertion happens in two phases:
/// 1. `store_documents()` - Stores documents in the document storage and sends to read side
/// 2. `index_documents()` - Processes documents through the index (creates embeddings, updates indexes)
pub struct InsertDocuments<'ctx> {
    context: WriteOperationContext<'ctx>,
    request: InsertDocumentsRequest,
    /// The target index ID for document storage keys.
    /// For temp indexes, this is the linked runtime index ID.
    target_index_id: IndexId,
}

impl<'ctx> InsertDocuments<'ctx> {
    /// Creates a new insert operation.
    ///
    /// This performs validation and constraint checks before the operation can be executed:
    /// - Verifies the collection and index exist
    /// - Checks document count limits against JWT claims
    ///
    /// # Arguments
    /// * `context` - The context with shared resources
    /// * `request` - The insert request with documents
    ///
    /// # Returns
    /// A new `InsertDocuments` ready for execution, or an error if validation fails.
    pub async fn new(
        context: WriteOperationContext<'ctx>,
        request: InsertDocumentsRequest,
    ) -> Result<Self, WriteError> {
        let collection = context
            .collections
            .get_collection(request.collection_id)
            .await
            .ok_or(WriteError::CollectionNotFound(request.collection_id))?;

        let index = collection
            .get_index(request.index_id)
            .await
            .ok_or(WriteError::IndexNotFound(
                request.collection_id,
                request.index_id,
            ))?;

        // Constraint checks
        if index.is_runtime() {
            collection
                .check_claim_limitations(request.write_api_key, request.documents.len(), 0)
                .await?;
        } else {
            // Temp index: check against linked runtime index
            let linked_runtime_id = index
                .get_runtime_index_id()
                .expect("Temp index must have a linked runtime index");
            collection
                .check_claim_limitations_for_temp_index(
                    request.write_api_key,
                    request.index_id,
                    linked_runtime_id,
                    request.documents.len(),
                )
                .await?;
        }

        // For temp indexes, use the linked runtime index ID for document storage keys
        let target_index_id = index.get_runtime_index_id().unwrap_or(request.index_id);

        drop(index);
        drop(collection);

        Ok(Self {
            context,
            request,
            target_index_id,
        })
    }

    /// Executes the insert operation.
    ///
    /// This performs the two-phase insertion:
    /// 1. Store documents in the document storage
    /// 2. Process documents through the index
    ///
    /// # Returns
    /// The result containing counts of inserted, replaced, and failed documents.
    pub async fn execute(mut self) -> Result<InsertDocumentsResult, WriteError> {
        let document_count = self.request.documents.len();
        info!(?document_count, "Inserting batch of documents");

        // Get collection and index for phase 1
        let collection = self
            .context
            .collections
            .get_collection(self.request.collection_id)
            .await
            .ok_or(WriteError::CollectionNotFound(self.request.collection_id))?;

        let index = collection
            .get_index(self.request.index_id)
            .await
            .ok_or(WriteError::IndexNotFound(
                self.request.collection_id,
                self.request.index_id,
            ))?;

        // Phase 1: Store documents
        debug!("Inserting documents {}", document_count);
        let collection_document_storage = collection.get_document_storage();
        let doc_ids = self
            .store_documents(collection_document_storage)
            .await
            .context("Cannot insert documents into document storage")?;
        debug!("Documents inserted");

        // Add fields if needed (schema detection)
        debug!("Looking for new fields...");
        index
            .add_fields_if_needed(&self.request.documents)
            .await
            .context("Cannot add fields if needed")?;
        debug!("Done");

        // Release locks before phase 2
        drop(index);
        drop(collection);

        // Phase 2: Index documents
        debug!("Processing documents {}", document_count);
        let result = self.index_documents(doc_ids).await?;
        info!("All documents are inserted: {}", document_count);

        Ok(result)
    }

    /// Phase 1: Stores documents in the document storage and sends operations to read side.
    ///
    /// This method:
    /// - Generates document IDs for documents without them
    /// - Batches documents for efficient storage
    /// - Sends operations to the read side via the operation sender
    async fn store_documents(
        &mut self,
        collection_document_storage: &CollectionDocumentStorage,
    ) -> Result<Vec<DocumentId>> {
        let document_count = self.request.documents.len();

        let batch_size = document_count.min(200);
        let mut batch = Vec::with_capacity(batch_size);

        let mut insert_document_batch = Vec::with_capacity(document_count);
        let mut doc_ids = Vec::with_capacity(document_count);
        let mut docs = Vec::with_capacity(batch_size);

        for (index, doc) in self.request.documents.0.iter_mut().enumerate() {
            if index % 100 == 0 {
                trace!("Processing document {}/{}", index, document_count);
            }

            // Flush batch when full
            if index % batch_size == 0 && !batch.is_empty() {
                collection_document_storage
                    .insert_many(&docs)
                    .await
                    .context("Cannot insert document into document storage")?;
                docs.clear();

                insert_document_batch.push(WriteOperation::Collection(
                    self.request.collection_id,
                    CollectionWriteOperation::DocumentStorage(
                        DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr(batch),
                    ),
                ));
                batch = Vec::with_capacity(batch_size);
            }

            // Extract or generate document ID
            let doc_id_str = extract_or_generate_document_id(doc);

            let doc_id = collection_document_storage.get_next_document_id();
            doc_ids.push(doc_id);

            batch.push((
                doc_id,
                doc_id_str.clone(),
                DocumentToInsert(
                    doc.clone()
                        .into_raw(format!("{}:{}", self.target_index_id, doc_id_str))
                        .expect("Cannot get raw document"),
                ),
            ));

            let doc_str = serde_json::to_string(&doc.inner).context("Cannot serialize document")?;
            docs.push((
                doc_id,
                ZeboDocument::new(Cow::Owned(doc_id_str), Cow::Owned(doc_str)),
            ));
        }

        // Flush remaining batch
        if !batch.is_empty() {
            insert_document_batch.push(WriteOperation::Collection(
                self.request.collection_id,
                CollectionWriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr(batch),
                ),
            ));

            collection_document_storage
                .insert_many(&docs)
                .await
                .context("Cannot insert document into document storage")?;
        }

        trace!("Sending documents");
        self.context
            .op_sender
            .send_batch(insert_document_batch)
            .await
            .context("Cannot send document storage operation")?;
        trace!("Documents sent");

        Ok(doc_ids)
    }

    /// Phase 2: Processes documents through the index.
    ///
    /// This method:
    /// - Processes each document (creates embeddings, updates indexes)
    /// - Yields to pending write operations when needed
    /// - Tracks affected pin rules and shelves
    /// - Cleans up replaced documents
    async fn index_documents(
        self,
        doc_ids: Vec<DocumentId>,
    ) -> Result<InsertDocumentsResult, WriteError> {
        let mut result = InsertDocumentsResult {
            inserted: 0,
            replaced: 0,
            failed: 0,
        };

        let collection_id = self.request.collection_id;
        let index_id = self.request.index_id;

        let mut collection = self
            .context
            .collections
            .get_collection(collection_id)
            .await
            .ok_or(WriteError::CollectionNotFound(collection_id))?;

        let mut index = collection
            .get_index(index_id)
            .await
            .ok_or(WriteError::IndexNotFound(collection_id, index_id))?;

        let mut pin_rules_writer = collection.get_pin_rule_writer("process_documents").await;
        let mut pin_rules_touched = HashSet::new();

        let mut shelves_writer = collection.get_shelves_writer("process_documents").await;
        let mut shelves_touched = HashSet::new();

        let document_count = self.request.documents.len();

        let mut index_operation_batch = Vec::with_capacity(document_count * 10);
        let mut docs_to_remove = Vec::with_capacity(document_count);

        for (i, doc) in self.request.documents.0.into_iter().enumerate() {
            if i % 100 == 0 {
                info!("Processing document {}/{}", i, document_count);
            }

            let doc_id = doc_ids[i];

            // Flush batch when 80% full
            if index_operation_batch.capacity() * 4 / 5 < index_operation_batch.len() {
                trace!("Sending operations");
                self.context
                    .op_sender
                    .send_batch(index_operation_batch)
                    .await
                    .context("Cannot send index operation")?;
                trace!("Operations sent");
                index_operation_batch = Vec::with_capacity(document_count * 10);
            }

            // Check for pending write operations and yield if needed
            let yield_guard = WriteYieldGuard::new(
                self.context.write_operation_counter,
                self.context.collections,
            );
            if yield_guard.should_yield() {
                // Drop locks before waiting
                drop(shelves_writer);
                drop(pin_rules_writer);
                drop(index);
                drop(collection);

                yield_guard.wait_for_pending_operations().await?;

                // Reacquire locks
                collection = self
                    .context
                    .collections
                    .get_collection(collection_id)
                    .await
                    .ok_or(WriteError::CollectionNotFound(collection_id))?;
                index = collection
                    .get_index(index_id)
                    .await
                    .ok_or(WriteError::IndexNotFound(collection_id, index_id))?;
                pin_rules_writer = collection.get_pin_rule_writer("process_documents").await;
                shelves_writer = collection.get_shelves_writer("process_documents").await;
            }

            let doc_id_str = doc
                .inner
                .get("id")
                .context("Document does not have an id")?
                .as_str()
                .context("Document id is not a string")?
                .to_string();

            match index
                .process_new_document(doc_id, doc_id_str.clone(), doc, &mut index_operation_batch)
                .await
                .context("Cannot process document")
            {
                Ok(Some(old_doc_id)) => {
                    docs_to_remove.push(old_doc_id);
                    result.inserted += 1;
                }
                Ok(None) => {
                    result.inserted += 1;
                }
                Err(e) => {
                    // If the document cannot be processed, remove it from storage
                    if let Err(remove_err) =
                        collection.get_document_storage().remove(vec![doc_id]).await
                    {
                        tracing::error!(error = ?remove_err, "Cannot remove failed document");
                    }

                    tracing::error!(error = ?e, "Cannot process document");
                    result.failed += 1;
                    continue;
                }
            };

            pin_rules_touched.extend(pin_rules_writer.get_matching_rules_ids(&doc_id_str));
            shelves_touched.extend(shelves_writer.get_matching_shelves_ids(&doc_id_str));
        }

        debug!("All documents processed {}", document_count);

        // Update affected pin rules and shelves
        collection
            .update_pin_rules(pin_rules_touched, &mut index_operation_batch)
            .await;

        collection
            .update_shelves(shelves_touched, &mut index_operation_batch)
            .await;

        // Send remaining operations
        if !index_operation_batch.is_empty() {
            trace!("Sending operations");
            self.context
                .op_sender
                .send_batch(index_operation_batch)
                .await
                .context("Cannot send index operation")?;
            trace!("Operations sent");
        }

        // Remove replaced documents
        collection
            .get_document_storage()
            .remove(docs_to_remove)
            .await
            .context("Cannot remove replaced documents")?;

        Ok(result)
    }
}
