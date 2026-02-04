//! Update documents operation.
//!
//! This module provides the `UpdateDocuments` struct which handles document updates
//! following the same pattern as the read side's `Search` struct.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use tokio_stream::StreamExt;
use tracing::{debug, trace};

use crate::collection_manager::sides::write::document_storage::ZeboDocument;
use crate::collection_manager::sides::{
    CollectionWriteOperation, DocumentStorageWriteOperation, DocumentToInsert, WriteOperation,
};
use crate::types::{
    CollectionId, Document, IndexId, UpdateDocumentRequest, UpdateDocumentsResult, WriteApiKey,
};

use super::helpers::{extract_or_generate_document_id, merge, WriteOperationContext, WriteYieldGuard};
use super::WriteError;

/// Request to update documents in an index.
pub struct UpdateDocumentsRequest {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub documents: UpdateDocumentRequest,
    pub write_api_key: WriteApiKey,
}

/// Operation for updating documents in an index.
///
/// This struct follows the same pattern as the read side's `Search` struct:
/// - Created with `new()` for validation and constraint checks
/// - Executed with `execute()` to perform the update
///
/// The update operation:
/// 1. Streams existing documents from storage
/// 2. Merges delta changes into each existing document
/// 3. Stores the updated documents
/// 4. Processes them through the index
pub struct UpdateDocuments<'ctx> {
    context: WriteOperationContext<'ctx>,
    request: UpdateDocumentsRequest,
}

impl<'ctx> UpdateDocuments<'ctx> {
    /// Creates a new update operation.
    ///
    /// # Arguments
    /// * `context` - The context with shared resources
    /// * `request` - The update request with document deltas
    ///
    /// # Returns
    /// A new `UpdateDocuments` ready for execution.
    pub fn new(context: WriteOperationContext<'ctx>, request: UpdateDocumentsRequest) -> Self {
        Self { context, request }
    }

    /// Executes the update operation.
    ///
    /// # Returns
    /// The result containing counts of inserted, updated, and failed documents.
    pub async fn execute(self) -> Result<UpdateDocumentsResult, WriteError> {
        let Self { context, request } = self;

        let collection_id = request.collection_id;
        let index_id = request.index_id;
        let document_count = request.documents.documents.len();

        let mut collection = context
            .collections
            .get_collection(collection_id)
            .await
            .ok_or(WriteError::CollectionNotFound(collection_id))?;

        let mut index = collection
            .get_index(index_id)
            .await
            .ok_or(WriteError::IndexNotFound(collection_id, index_id))?;

        // Constraint checks for runtime indexes
        if index.is_runtime() {
            collection
                .check_claim_limitations(request.write_api_key, document_count, 0)
                .await?;
        }

        let mut collection_document_storage = collection.get_document_storage();
        let target_index_id = index.get_runtime_index_id().unwrap_or(index_id);

        // Prepare document ID mapping
        let document_id_storage = index.get_document_id_storage().await;
        let document_ids_map: HashMap<_, _> = request
            .documents
            .documents
            .0
            .iter()
            .filter_map(|d| {
                d.get("id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| document_id_storage.get(s).map(|id| (id, s.to_string())))
            })
            .collect();
        let document_ids: Vec<_> = document_ids_map.keys().copied().collect();
        drop(document_id_storage);

        // Prepare documents with IDs
        let mut documents: HashMap<String, Document> = request
            .documents
            .documents
            .into_iter()
            .map(|mut doc| {
                let doc_id_str = extract_or_generate_document_id(&mut doc);
                (doc_id_str, doc)
            })
            .collect();

        let mut pin_rules_writer = collection.get_pin_rule_writer("update_documents").await;
        let mut pin_rules_touched = HashSet::new();

        let mut shelves_writer = collection.get_shelves_writer("update_documents").await;
        let mut shelves_touched = HashSet::new();

        let mut result = UpdateDocumentsResult {
            inserted: 0,
            updated: 0,
            failed: 0,
        };

        let mut index_operation_batch = Vec::with_capacity(document_count * 10);
        let mut docs_to_remove = Vec::with_capacity(document_count);

        let mut doc_stream = collection_document_storage
            .stream_documents(document_ids)
            .await;
        let mut processed_count = -1i32;

        while let Some((doc_id, doc)) = doc_stream.next().await {
            processed_count += 1;
            if processed_count % 100 == 0 {
                trace!(
                    "Processing document {}/{}",
                    processed_count,
                    document_count
                );
            }

            // Flush batch when it gets too large
            if index_operation_batch.len() > 200 {
                trace!("Sending operations");
                context
                    .op_sender
                    .send_batch(index_operation_batch)
                    .await
                    .context("Cannot send index operation")?;
                trace!("Operations sent");
                index_operation_batch = Vec::with_capacity(document_count * 10);
            }

            // Check for pending write operations and yield if needed
            let yield_guard =
                WriteYieldGuard::new(context.write_operation_counter, context.collections);
            if yield_guard.should_yield() {
                drop(shelves_writer);
                drop(pin_rules_writer);
                // Force drop of collection_document_storage by reassigning
                let _ = collection_document_storage;
                drop(index);
                drop(collection);

                yield_guard.wait_for_pending_operations().await?;

                collection = context
                    .collections
                    .get_collection(collection_id)
                    .await
                    .ok_or(WriteError::CollectionNotFound(collection_id))?;
                index = collection
                    .get_index(index_id)
                    .await
                    .ok_or(WriteError::IndexNotFound(collection_id, index_id))?;
                collection_document_storage = collection.get_document_storage();
                pin_rules_writer = collection.get_pin_rule_writer("update_documents").await;
                shelves_writer = collection.get_shelves_writer("update_documents").await;
            }

            if let Some(doc_id_str) = document_ids_map.get(&doc_id) {
                if let Ok(v) = serde_json::from_str(doc.inner.get()) {
                    if let Some(delta) = documents.remove(doc_id_str) {
                        let new_document = merge(v, delta);

                        let new_doc_id = collection_document_storage.get_next_document_id();

                        let doc_str = serde_json::to_string(&doc.inner)
                            .context("Cannot serialize document")?;
                        collection_document_storage
                            .insert_many(&[(
                                new_doc_id,
                                ZeboDocument::new(Cow::Borrowed(doc_id_str), Cow::Owned(doc_str)),
                            )])
                            .await
                            .context("Cannot insert document into document storage")?;

                        context
                            .op_sender
                            .send(WriteOperation::Collection(
                                collection_id,
                                CollectionWriteOperation::DocumentStorage(
                                    DocumentStorageWriteOperation::InsertDocumentWithDocIdStr {
                                        doc_id: new_doc_id,
                                        doc_id_str: doc_id_str.clone(),
                                        doc: DocumentToInsert(
                                            new_document
                                                .clone()
                                                .into_raw(format!("{target_index_id}:{doc_id_str}"))
                                                .expect("Cannot get raw document"),
                                        ),
                                    },
                                ),
                            ))
                            .await
                            .context("Cannot send document storage operation")?;

                        let new_doc_id_str = new_document
                            .inner
                            .get("id")
                            .context("Document does not have an id")?
                            .as_str()
                            .context("Document id is not a string")?
                            .to_string();

                        match index
                            .process_new_document(
                                new_doc_id,
                                new_doc_id_str,
                                new_document,
                                &mut index_operation_batch,
                            )
                            .await
                            .context("Cannot process document")
                        {
                            Ok(Some(old_doc_id)) => {
                                docs_to_remove.push(old_doc_id);
                                result.updated += 1;
                            }
                            Ok(None) => {
                                result.inserted += 1;
                            }
                            Err(e) => {
                                // If the document cannot be processed, remove it from storage
                                collection_document_storage
                                    .remove(vec![new_doc_id])
                                    .await
                                    .context("Cannot remove document after failed processing")?;

                                tracing::error!(error = ?e, "Cannot process document");
                                result.failed += 1;
                            }
                        };
                    }
                }

                pin_rules_touched.extend(pin_rules_writer.get_matching_rules_ids(doc_id_str));
                shelves_touched.extend(shelves_writer.get_matching_shelves_ids(doc_id_str));
            }
        }

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
            context
                .op_sender
                .send_batch(index_operation_batch)
                .await
                .context("Cannot send index operation")?;
            trace!("Operations sent");
        }

        // Remove replaced documents
        collection_document_storage
            .remove(docs_to_remove)
            .await
            .context("Cannot remove replaced documents")?;

        debug!("All documents");

        Ok(result)
    }
}
