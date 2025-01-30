use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

use serde::{Deserialize, Serialize};

use crate::collection_manager::dto::{FieldId, Number};
use crate::metrics::{Empty, OPERATION_GAUGE};
use crate::types::{CollectionId, DocumentId, RawJSONDocument};

use crate::collection_manager::dto::{ApiKey, TypedField};

#[derive(Debug, Clone)]
pub enum GenericWriteOperation {
    CreateCollection,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Term(pub String);

#[derive(Debug, Clone)]
pub struct TermStringField {
    pub positions: Vec<usize>,
}

pub type InsertStringTerms = HashMap<Term, TermStringField>;

#[derive(Debug, Clone)]
pub enum DocumentFieldIndexOperation {
    IndexString {
        field_length: u16,
        terms: InsertStringTerms,
    },
    IndexEmbedding {
        value: Vec<f32>,
    },
    IndexNumber {
        value: Number,
    },
    IndexBoolean {
        value: bool,
    },
}

#[derive(Debug, Clone)]
pub enum CollectionWriteOperation {
    InsertDocument {
        doc_id: DocumentId,
        doc: RawJSONDocument,
    },
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
    CreateField {
        field_id: FieldId,
        field_name: String,
        field: TypedField,
    },
    Index(DocumentId, FieldId, DocumentFieldIndexOperation),
}

#[derive(Debug, Clone)]
pub enum WriteOperation {
    CreateCollection {
        id: CollectionId,
        read_api_key: ApiKey,
        // Params here... but which ones?
        // TODO: add params
    },
    Collection(CollectionId, CollectionWriteOperation),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Offset(pub u64);

#[derive(Clone)]
pub struct OperationSender {
    offset_counter: Arc<AtomicU64>,
    sender: tokio::sync::mpsc::Sender<(Offset, WriteOperation)>,
}

impl OperationSender {
    pub fn offset(&self) -> Offset {
        Offset(
            self.offset_counter
                .load(std::sync::atomic::Ordering::SeqCst),
        )
    }
    pub fn set_offset(&self, offset: Offset) {
        self.offset_counter
            .store(offset.0, std::sync::atomic::Ordering::SeqCst);
    }

    pub async fn send(
        &self,
        operation: WriteOperation,
    ) -> Result<(), tokio::sync::mpsc::error::SendError<(Offset, WriteOperation)>> {
        OPERATION_GAUGE.create(Empty {}).increment_by(1);
        let offset = self
            .offset_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.sender.send((Offset(offset), operation)).await?;
        Ok(())
    }
}

pub struct OperationReceiver {
    receiver: tokio::sync::mpsc::Receiver<(Offset, WriteOperation)>,
}

impl OperationReceiver {
    pub async fn recv(&mut self) -> Option<(Offset, WriteOperation)> {
        let r = self.receiver.recv().await;
        OPERATION_GAUGE.create(Empty {}).decrement_by(1);
        r
    }
}

pub fn channel(capacity: usize) -> (OperationSender, OperationReceiver) {
    let (sender, receiver) = tokio::sync::mpsc::channel(capacity);

    (
        OperationSender {
            //  We internally use `0` to represent "no offset"
            // So, we start at `1` to avoid confusion
            // This is a bit of a hack, we should model this better
            // TODO: model this better
            offset_counter: Arc::new(AtomicU64::new(1)),
            sender,
        },
        OperationReceiver { receiver },
    )
}
