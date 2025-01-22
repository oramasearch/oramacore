use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

use serde::{Deserialize, Serialize};

use crate::types::{CollectionId, DocumentId, RawJSONDocument};
use crate::{collection_manager::dto::FieldId, indexes::number::Number};

use crate::collection_manager::dto::TypedField;

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
    sender: tokio::sync::broadcast::Sender<(Offset, WriteOperation)>,
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

    pub fn send(
        &self,
        operation: WriteOperation,
    ) -> Result<(), tokio::sync::broadcast::error::SendError<(Offset, WriteOperation)>> {
        let offset = self
            .offset_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.sender.send((Offset(offset), operation))?;
        Ok(())
    }
}

pub struct OperationReceiver {
    receiver: tokio::sync::broadcast::Receiver<(Offset, WriteOperation)>,
}

impl OperationReceiver {
    pub async fn recv(
        &mut self,
    ) -> Result<(Offset, WriteOperation), tokio::sync::broadcast::error::RecvError> {
        self.receiver.recv().await
    }
}

pub fn channel(capacity: usize) -> (OperationSender, OperationReceiver) {
    let (sender, receiver) = tokio::sync::broadcast::channel(capacity);

    (
        OperationSender {
            offset_counter: Arc::new(AtomicU64::new(0)),
            sender,
        },
        OperationReceiver { receiver },
    )
}
