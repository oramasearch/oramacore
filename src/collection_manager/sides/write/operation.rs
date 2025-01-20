use std::{collections::HashMap, fmt::Debug};

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

#[derive(Clone)]
pub struct OperationSender {
    sender: tokio::sync::broadcast::Sender<WriteOperation>,
}

impl OperationSender {
    pub fn send(
        &self,
        operation: WriteOperation,
    ) -> Result<(), tokio::sync::broadcast::error::SendError<WriteOperation>> {
        self.sender.send(operation)?;
        Ok(())
    }
}

pub struct OperationReceiver {
    receiver: tokio::sync::broadcast::Receiver<WriteOperation>,
}

impl OperationReceiver {
    pub async fn recv(
        &mut self,
    ) -> Result<WriteOperation, tokio::sync::broadcast::error::RecvError> {
        self.receiver.recv().await
    }
}

pub fn channel(capacity: usize) -> (OperationSender, OperationReceiver) {
    let (sender, receiver) = tokio::sync::broadcast::channel(capacity);

    (OperationSender { sender }, OperationReceiver { receiver })
}
