//! Common utilities for write operations.
//!
//! This module provides shared helper components used across insert, update, and delete operations
//! to reduce code duplication and ensure consistent behavior.

// Allow unused code for utilities that may be used in future refactoring
#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use anyhow::Result;
use oramacore_lib::shelves::ShelfId;
use tokio::time::{sleep, Instant};
use tracing::warn;

use crate::collection_manager::sides::{OperationSender, WriteOperation};
use crate::types::{CollectionId, Document, IndexId};

use super::collections::{CollectionReadLock, CollectionsWriter};
use super::WriteError;

/// Context needed to execute write operations (insert, update, delete).
///
/// This provides access to shared resources from WriteSide.
pub struct WriteOperationContext<'a> {
    pub op_sender: &'a OperationSender,
    pub collections: &'a CollectionsWriter,
    pub write_operation_counter: &'a AtomicU32,
}

/// Guard that handles yielding to pending write operations.
///
/// When long-running operations (like document insertion) are processing,
/// they need to periodically check if there are pending write operations
/// (like collection creation/deletion) that need to acquire write locks.
/// This guard encapsulates that pattern.
pub struct WriteYieldGuard<'a> {
    write_operation_counter: &'a AtomicU32,
    collections: &'a CollectionsWriter,
}

impl<'a> WriteYieldGuard<'a> {
    /// Creates a new WriteYieldGuard.
    pub fn new(write_operation_counter: &'a AtomicU32, collections: &'a CollectionsWriter) -> Self {
        Self {
            write_operation_counter,
            collections,
        }
    }

    /// Checks if there are pending write operations that need the lock.
    pub fn should_yield(&self) -> bool {
        self.write_operation_counter.load(Ordering::Relaxed) > 0
    }

    /// Waits for pending write operations to complete.
    ///
    /// Uses exponential backoff starting at 10ms, capped at 500ms.
    /// Times out after 5 seconds with a warning.
    pub async fn wait_for_pending_operations(&self) -> Result<()> {
        let timeout = Duration::from_secs(5);
        let start = Instant::now();

        // Initialize backoff parameters
        let mut backoff = Duration::from_millis(10); // Start at 10ms
        let max_backoff = Duration::from_millis(500); // Cap at 500ms to allow multiple retries within 5s

        while self.write_operation_counter.load(Ordering::Relaxed) > 0 {
            if start.elapsed() > timeout {
                warn!("Timeout waiting for write operations to complete");
                break;
            }

            // If there's some write pending operation we yield to let the write operation be processed
            tokio::task::yield_now().await;

            // Anyway, `yield_now` doesn't guarantee that the other task will be executed
            // If the scheduler will process this task without waiting the other task,
            // we propose a sleep of 10ms to be sure that the other task will be executed.
            // Anyway, this is not guaranteed again, it is just an hope.
            if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                sleep(backoff).await;
                // Triple the backoff time, but cap it at max_backoff
                backoff = (backoff * 3).min(max_backoff);
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Reacquires a collection lock after yielding.
    ///
    /// Returns an error if the collection no longer exists (was deleted during the yield).
    pub async fn reacquire_collection(
        &self,
        collection_id: CollectionId,
    ) -> Result<CollectionReadLock<'_>, WriteError> {
        self.collections
            .get_collection(collection_id)
            .await
            .ok_or(WriteError::CollectionNotFound(collection_id))
    }
}

/// Batches write operations and flushes them when thresholds are reached.
///
/// This helps reduce overhead from sending many small batches by grouping
/// operations together until a threshold is reached.
pub struct OperationBatcher {
    batch: Vec<WriteOperation>,
    threshold: BatchThreshold,
}

/// Defines when to flush the operation batch.
pub enum BatchThreshold {
    /// Flush when the batch reaches this percentage of its capacity.
    /// For example, 80 means flush when 80% full.
    CapacityPercentage(usize),
    /// Flush when the batch contains this many items.
    FixedCount(usize),
}

impl OperationBatcher {
    /// Creates a new OperationBatcher with the given capacity and threshold.
    pub fn new(capacity: usize, threshold: BatchThreshold) -> Self {
        Self {
            batch: Vec::with_capacity(capacity),
            threshold,
        }
    }

    /// Adds an operation to the batch.
    pub fn push(&mut self, op: WriteOperation) {
        self.batch.push(op);
    }

    /// Extends the batch with multiple operations.
    pub fn extend(&mut self, ops: impl IntoIterator<Item = WriteOperation>) {
        self.batch.extend(ops);
    }

    /// Returns true if the batch should be flushed based on the threshold.
    pub fn should_flush(&self) -> bool {
        match self.threshold {
            BatchThreshold::CapacityPercentage(percent) => {
                let threshold = self.batch.capacity() * percent / 100;
                self.batch.len() >= threshold
            }
            BatchThreshold::FixedCount(count) => self.batch.len() >= count,
        }
    }

    /// Takes and returns the current batch, leaving an empty batch in its place.
    /// Returns None if the batch is empty.
    pub fn flush(&mut self) -> Option<Vec<WriteOperation>> {
        if self.batch.is_empty() {
            return None;
        }
        let capacity = self.batch.capacity();
        Some(std::mem::replace(&mut self.batch, Vec::with_capacity(capacity)))
    }

    /// Takes the remaining operations regardless of threshold.
    /// Returns None if the batch is empty.
    pub fn flush_remaining(&mut self) -> Option<Vec<WriteOperation>> {
        if self.batch.is_empty() {
            return None;
        }
        let capacity = self.batch.capacity();
        Some(std::mem::replace(&mut self.batch, Vec::with_capacity(capacity)))
    }

    /// Returns the current batch mutably for direct manipulation.
    pub fn batch_mut(&mut self) -> &mut Vec<WriteOperation> {
        &mut self.batch
    }

    /// Returns true if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    /// Returns the number of operations in the batch.
    pub fn len(&self) -> usize {
        self.batch.len()
    }
}

/// Extracts a document ID from a document, or generates a new one.
///
/// If the document has an "id" field with a non-empty string value, that is used.
/// Otherwise, a new unique ID is generated using cuid2.
/// The document is mutated to ensure it always has a valid string "id" field.
pub fn extract_or_generate_document_id(doc: &mut Document) -> String {
    let doc_id_str = doc.get("id").and_then(|v| v.as_str());
    let doc_id_str = if let Some(doc_id_str) = doc_id_str {
        if doc_id_str.is_empty() {
            cuid2::create_id()
        } else {
            doc_id_str.to_string()
        }
    } else {
        cuid2::create_id()
    };

    // Ensure the document contains a valid id
    // NB: this overwrites the previous id if it is not a string
    doc.inner.insert(
        "id".to_string(),
        serde_json::Value::String(doc_id_str.clone()),
    );

    doc_id_str
}

/// Tracks which pin rules and shelves have been affected by document operations.
///
/// When documents are inserted, updated, or deleted, we need to track which
/// pin rules and shelves reference those documents so we can update them.
#[derive(Default)]
pub struct AffectedReferencesTracker {
    pin_rules_touched: HashSet<String>,
    shelves_touched: HashSet<ShelfId>,
}

impl AffectedReferencesTracker {
    /// Creates a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records that a pin rule was affected.
    pub fn record_pin_rule(&mut self, rule_id: String) {
        self.pin_rules_touched.insert(rule_id);
    }

    /// Records multiple pin rules as affected.
    pub fn record_pin_rules(&mut self, rule_ids: impl IntoIterator<Item = String>) {
        self.pin_rules_touched.extend(rule_ids);
    }

    /// Records that a shelf was affected.
    pub fn record_shelf(&mut self, shelf_id: ShelfId) {
        self.shelves_touched.insert(shelf_id);
    }

    /// Records multiple shelves as affected.
    pub fn record_shelves(&mut self, shelf_ids: impl IntoIterator<Item = ShelfId>) {
        self.shelves_touched.extend(shelf_ids);
    }

    /// Returns the set of affected pin rules.
    pub fn pin_rules_touched(self) -> HashSet<String> {
        self.pin_rules_touched
    }

    /// Returns the set of affected shelves.
    pub fn shelves_touched(self) -> HashSet<ShelfId> {
        self.shelves_touched
    }

    /// Consumes the tracker and returns both sets.
    pub fn into_parts(self) -> (HashSet<String>, HashSet<ShelfId>) {
        (self.pin_rules_touched, self.shelves_touched)
    }
}

/// Merges a delta document into an existing document.
///
/// This function supports:
/// - Simple key-value updates (replaces the value)
/// - Nested property updates using dot notation (e.g., "person.name")
/// - Removing keys by setting them to null
///
/// # Arguments
/// * `old` - The existing document as a JSON object
/// * `delta` - The document containing the changes to apply
///
/// # Returns
/// A new Document with the merged values
pub fn merge(mut old: serde_json::value::Map<String, serde_json::Value>, delta: Document) -> Document {
    let delta = delta.inner;
    for (k, v) in delta {
        if k.contains('.') {
            // Handle nested property updates using dot notation
            let mut path = k.split('.').peekable();
            let mut nested_doc = Some(&mut old);
            let k = loop {
                let k = path.next();
                let f = path.peek();
                nested_doc = match (k, f) {
                    (None, _) => break None,
                    (Some(k), None) => break Some(k),
                    (Some(k), _) => {
                        nested_doc.and_then(|v| v.get_mut(k).and_then(|v| v.as_object_mut()))
                    }
                }
            };
            if let Some(nested_doc) = nested_doc {
                if let Some(k) = k {
                    if v.is_null() {
                        // Null removes the key from an object
                        nested_doc.remove(k);
                    } else {
                        nested_doc.insert(k.to_string(), v);
                    }
                }
            }
        } else {
            // Fast path for simple key-value updates
            if v.is_null() {
                // Null removes the key from an object
                old.remove(&k);
            } else {
                old.insert(k, v);
            }
        }
    }
    Document { inner: old }
}

/// Context needed for document operations that require yielding.
///
/// This is used to reacquire resources after yielding to pending write operations.
pub struct DocumentOperationContext {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
}

impl DocumentOperationContext {
    pub fn new(collection_id: CollectionId, index_id: IndexId) -> Self {
        Self {
            collection_id,
            index_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_or_generate_document_id_with_existing_id() {
        let serde_json::Value::Object(inner) = json!({
            "id": "existing-id",
            "name": "test"
        }) else {
            panic!("Expected object");
        };
        let mut doc = Document { inner };

        let id = extract_or_generate_document_id(&mut doc);
        assert_eq!(id, "existing-id");
        assert_eq!(doc.get("id").unwrap().as_str().unwrap(), "existing-id");
    }

    #[test]
    fn test_extract_or_generate_document_id_with_empty_id() {
        let serde_json::Value::Object(inner) = json!({
            "id": "",
            "name": "test"
        }) else {
            panic!("Expected object");
        };
        let mut doc = Document { inner };

        let id = extract_or_generate_document_id(&mut doc);
        assert!(!id.is_empty());
        assert_ne!(id, "");
        assert_eq!(doc.get("id").unwrap().as_str().unwrap(), id);
    }

    #[test]
    fn test_extract_or_generate_document_id_without_id() {
        let serde_json::Value::Object(inner) = json!({
            "name": "test"
        }) else {
            panic!("Expected object");
        };
        let mut doc = Document { inner };

        let id = extract_or_generate_document_id(&mut doc);
        assert!(!id.is_empty());
        assert_eq!(doc.get("id").unwrap().as_str().unwrap(), id);
    }

    #[test]
    fn test_operation_batcher_capacity_threshold() {
        let mut batcher = OperationBatcher::new(10, BatchThreshold::CapacityPercentage(80));

        // Add 7 items (70% of 10) - should not trigger flush
        for _ in 0..7 {
            batcher.push(WriteOperation::DeleteCollection(CollectionId::try_new("test").unwrap()));
        }
        assert!(!batcher.should_flush());

        // Add 1 more item (80% of 10) - should trigger flush
        batcher.push(WriteOperation::DeleteCollection(CollectionId::try_new("test").unwrap()));
        assert!(batcher.should_flush());
    }

    #[test]
    fn test_operation_batcher_fixed_count_threshold() {
        let mut batcher = OperationBatcher::new(100, BatchThreshold::FixedCount(5));

        for _ in 0..4 {
            batcher.push(WriteOperation::DeleteCollection(CollectionId::try_new("test").unwrap()));
        }
        assert!(!batcher.should_flush());

        batcher.push(WriteOperation::DeleteCollection(CollectionId::try_new("test").unwrap()));
        assert!(batcher.should_flush());
    }

    #[test]
    fn test_operation_batcher_flush() {
        let mut batcher = OperationBatcher::new(10, BatchThreshold::FixedCount(5));

        for _ in 0..3 {
            batcher.push(WriteOperation::DeleteCollection(CollectionId::try_new("test").unwrap()));
        }

        let flushed = batcher.flush_remaining();
        assert!(flushed.is_some());
        assert_eq!(flushed.unwrap().len(), 3);
        assert!(batcher.is_empty());
    }

    #[test]
    fn test_affected_references_tracker() {
        let mut tracker = AffectedReferencesTracker::new();

        tracker.record_pin_rule("rule1".to_string());
        tracker.record_pin_rules(vec!["rule2".to_string(), "rule3".to_string()]);
        tracker.record_shelf(ShelfId::try_new("shelf1").unwrap());
        tracker.record_shelves(vec![ShelfId::try_new("shelf2").unwrap()]);

        let (pin_rules, shelves) = tracker.into_parts();
        assert_eq!(pin_rules.len(), 3);
        assert!(pin_rules.contains("rule1"));
        assert!(pin_rules.contains("rule2"));
        assert!(pin_rules.contains("rule3"));
        assert_eq!(shelves.len(), 2);
    }

    #[test]
    fn test_merge_simple() {
        let old = json!({
            "id": "1",
            "text": "foo",
            "name": "Tommaso",
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "text": "bar",
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "text": "bar",
                "name": "Tommaso",
            })
        );
    }

    #[test]
    fn test_merge_null_removes_key() {
        let old = json!({
            "id": "1",
            "name": "Tommaso",
            "age": 34,
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "age": null,
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "name": "Tommaso",
            })
        );
    }

    #[test]
    fn test_merge_nested_property() {
        let old = json!({
            "id": "1",
            "person": {
                "name": "Tommaso",
                "age": 34,
            }
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "person.name": "Michele",
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "person": {
                    "name": "Michele",
                    "age": 34,
                }
            })
        );
    }
}
