//! Delete documents operation.
//!
//! This module provides the `DeleteDocumentsOp` struct which handles document deletion
//! following the same pattern as the read side's `Search` struct.

use anyhow::Context;

use crate::types::{CollectionId, DeleteDocuments, DocumentId, IndexId};

use super::collections::CollectionReadLock;
use super::WriteError;

/// Request to delete documents from an index.
pub struct DeleteDocumentsRequest {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub document_ids: DeleteDocuments,
}

/// Operation for deleting documents from an index.
///
/// This struct follows the same pattern as the read side's `Search` struct:
/// - Created with `new()` for validation
/// - Executed with `execute()` to perform the operation
pub struct DeleteDocumentsOp<'collection> {
    collection: CollectionReadLock<'collection>,
    request: DeleteDocumentsRequest,
}

impl<'collection> DeleteDocumentsOp<'collection> {
    /// Creates a new delete operation.
    ///
    /// # Arguments
    /// * `collection` - The collection containing the index
    /// * `request` - The delete request with document IDs to delete
    ///
    /// # Returns
    /// A new `DeleteDocumentsOp` ready for execution.
    pub fn new(collection: CollectionReadLock<'collection>, request: DeleteDocumentsRequest) -> Self {
        Self {
            collection,
            request,
        }
    }

    /// Executes the delete operation.
    ///
    /// This deletes the specified documents from both the index and the document storage.
    ///
    /// # Returns
    /// `Ok(())` on success, or an error if the deletion fails.
    pub async fn execute(self) -> Result<(), WriteError> {
        let Self {
            collection,
            request,
        } = self;

        // Get the index - validation happens here
        let index = collection
            .get_index(request.index_id)
            .await
            .ok_or_else(|| WriteError::IndexNotFound(request.collection_id, request.index_id))?;

        // Delete documents from the index, getting back the (DocumentId, doc_id_str) pairs
        let doc_id_pairs = index.delete_documents(request.document_ids).await?;

        // Extract DocumentIds for removal from document storage
        let doc_ids: Vec<DocumentId> = doc_id_pairs.iter().map(|(doc_id, _)| *doc_id).collect();

        // Drop the index lock before accessing document storage
        drop(index);

        // Remove from collection's document storage
        collection
            .get_document_storage()
            .remove(doc_ids)
            .await
            .context("Cannot remove deleted documents")?;

        Ok(())
    }
}
