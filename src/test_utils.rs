use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc},
};

use anyhow::{Context, Result};
use tempdir::TempDir;

use crate::{
    collection_manager::{
        dto::FieldId,
        sides::write::{
            CollectionWriteOperation, DocumentFieldIndexOperation, FieldIndexer, StringField,
            WriteOperation,
        },
        CollectionId,
    },
    document_storage::DocumentId,
    indexes::string::{
        merge::merge, CommittedStringFieldIndex, StringIndex, StringIndexConfig,
        UncommittedStringFieldIndex,
    },
    nlp::TextParser,
    types::Document,
};

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").unwrap();
    let dir = tmp_dir.path().to_path_buf();
    fs::create_dir_all(dir.clone()).unwrap();
    dir
}

pub fn create_string_index(
    fields: Vec<(FieldId, String)>,
    documents: Vec<Document>,
) -> Result<StringIndex> {
    let index = StringIndex::new(StringIndexConfig {
        posting_id_generator: Arc::new(AtomicU64::new(0)),
        base_path: generate_new_path(),
    });

    let string_fields: Vec<_> = fields
        .into_iter()
        .map(|(field_id, field_name)| {
            (
                field_id,
                field_name,
                StringField::new(Arc::new(TextParser::from_language(
                    crate::nlp::locales::Locale::EN,
                ))),
            )
        })
        .collect();
    for (id, doc) in documents.into_iter().enumerate() {
        let document_id = DocumentId(id as u32);
        let flatten = doc.into_flatten();

        let operations: Vec<_> = string_fields
            .iter()
            .flat_map(|(field_id, field_name, string_field)| {
                string_field
                    .get_write_operations(
                        CollectionId("collection".to_string()),
                        document_id,
                        field_name,
                        *field_id,
                        &flatten,
                    )
                    .unwrap()
            })
            .collect();

        for operation in operations {
            match operation {
                WriteOperation::Collection(
                    _,
                    CollectionWriteOperation::Index(
                        doc_id,
                        field_id,
                        DocumentFieldIndexOperation::IndexString {
                            field_length,
                            terms,
                        },
                    ),
                ) => {
                    index.insert(doc_id, field_id, field_length, terms)?;
                }
                _ => unreachable!(),
            };
        }
    }

    Ok(index)
}

pub fn create_uncommitted_string_field_index(
    documents: Vec<Document>,
) -> Result<UncommittedStringFieldIndex> {
    create_uncommitted_string_field_index_from(documents, 0)
}

pub fn create_uncommitted_string_field_index_from(
    documents: Vec<Document>,
    starting_doc_id: u32,
) -> Result<UncommittedStringFieldIndex> {
    let index = UncommittedStringFieldIndex::new();

    let string_field = StringField::new(Arc::new(TextParser::from_language(
        crate::nlp::locales::Locale::EN,
    )));

    for (id, doc) in documents.into_iter().enumerate() {
        let document_id = DocumentId(starting_doc_id + id as u32);
        let flatten = doc.into_flatten();
        let operations = string_field
            .get_write_operations(
                CollectionId("collection".to_string()),
                document_id,
                "field",
                FieldId(1),
                &flatten,
            )
            .with_context(|| {
                format!("Test get_write_operations {:?} {:?}", document_id, flatten)
            })?;

        for operation in operations {
            match operation {
                WriteOperation::Collection(
                    _,
                    CollectionWriteOperation::Index(
                        _,
                        _,
                        DocumentFieldIndexOperation::IndexString {
                            field_length,
                            terms,
                        },
                    ),
                ) => {
                    index
                        .insert(document_id, field_length, terms)
                        .with_context(|| {
                            format!("test cannot insert index_string {:?}", document_id)
                        })?;
                }
                _ => unreachable!(),
            };
        }
    }

    Ok(index)
}

pub fn create_committed_string_field_index(
    documents: Vec<Document>,
) -> Result<(CommittedStringFieldIndex, Arc<AtomicU64>)> {
    let uncommitted = create_uncommitted_string_field_index(documents)?;

    let posting_id_generator: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
    let committed_index = merge(
        posting_id_generator.clone(),
        uncommitted,
        CommittedStringFieldIndex::new(None, HashMap::new(), 0, HashMap::new(), 0),
        generate_new_path(),
    )?;

    Ok((committed_index, posting_id_generator))
}
