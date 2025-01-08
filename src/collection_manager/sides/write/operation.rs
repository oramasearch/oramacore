use std::{collections::HashMap, fmt::Debug};

use crate::types::{CollectionId, DocumentId};
use crate::{collection_manager::dto::FieldId, indexes::number::Number, types::Document};

use crate::collection_manager::dto::TypedField;

#[derive(Debug, Clone)]
pub enum GenericWriteOperation {
    CreateCollection {
        id: CollectionId,
        // Params here... but which ones?
        // TODO: add params
    },
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
        doc: Document,
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
    Generic(GenericWriteOperation),
    Collection(CollectionId, CollectionWriteOperation),
}
