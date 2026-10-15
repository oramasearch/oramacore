use std::{borrow::Cow, sync::Arc};

use anyhow::Result;
use serde_json::{json, Value};
use tokio_stream::StreamExt;

use crate::{
    collection_manager::sides::{
        read::{
            collection_document_storage::CollectionDocumentStorage as ReadCollectionStorage,
            document_storage::{DocumentStorage as ReadStorage, DocumentStorageConfig},
        },
        write::{
            collection_document_storage::CollectionDocumentStorage as WriteCollectionStorage,
            document_storage::{DocumentStorage as WriteStorage, ZeboDocument},
        },
        DocumentStorageWriteOperation, DocumentToInsert, Offset,
    },
    types::{CollectionId, DocumentId, RawJSONDocument},
};

fn doc(id: u64, version: &str) -> (DocumentId, DocumentToInsert) {
    let raw = json!({ "id": id.to_string(), "version": version })
        .try_into()
        .expect("valid document");
    (DocumentId(id), DocumentToInsert(raw))
}

fn versions(mut docs: Vec<(DocumentId, Arc<RawJSONDocument>)>) -> Vec<(u64, String)> {
    docs.sort_by_key(|(id, _)| *id);
    docs.into_iter()
        .map(|(id, doc)| {
            let value: Value = serde_json::from_str(doc.inner.get()).expect("stored JSON");
            (id.0, value["version"].as_str().expect("version").to_owned())
        })
        .collect()
}

fn latest_versions() -> Vec<(u64, String)> {
    vec![(1, "v2".into()), (2, "v1".into()), (3, "v1".into())]
}

#[tokio::test]
#[allow(deprecated)]
async fn global_commit_sorts_accumulated_batches_and_keeps_last_write() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let storage = ReadStorage::try_new(DocumentStorageConfig {
        data_dir: dir.path().join("global"),
    })
    .await?;
    // Each incoming batch is sorted; their accumulation is not, and id 1 repeats.
    for docs in [
        vec![doc(1, "v1"), doc(3, "v1")],
        vec![doc(1, "v2"), doc(2, "v1")],
    ] {
        storage
            .update(DocumentStorageWriteOperation::InsertDocuments(docs))
            .await?;
    }
    storage.commit().await?;
    drop(storage);

    let storage = ReadStorage::try_new(DocumentStorageConfig {
        data_dir: dir.path().join("global"),
    })
    .await?;
    let mut found = Vec::new();
    storage
        .get_documents_by_ids(
            vec![DocumentId(1), DocumentId(2), DocumentId(3)],
            &mut found,
        )
        .await?;
    assert_eq!(versions(found), latest_versions());
    Ok(())
}

#[tokio::test]
async fn collection_commit_sorts_both_insert_variants_and_keeps_last_write() -> Result<()> {
    // Both update variants feed the same uncommitted buffer.
    for with_string_ids in [false, true] {
        let dir = tempfile::tempdir()?;
        let global = Arc::new(
            ReadStorage::try_new(DocumentStorageConfig {
                data_dir: dir.path().join("global"),
            })
            .await?,
        );
        let collection_id = CollectionId::try_new("test")?;
        let storage = ReadCollectionStorage::new(
            global.clone(),
            dir.path().join("collection"),
            collection_id,
        )?;
        for docs in [
            vec![doc(3, "v1"), doc(1, "v1")],
            vec![doc(2, "v1"), doc(1, "v2")],
        ] {
            let op = if with_string_ids {
                DocumentStorageWriteOperation::InsertDocumentsWithDocIdStr(
                    docs.into_iter()
                        .map(|(id, doc)| (id, id.0.to_string(), doc))
                        .collect(),
                )
            } else {
                DocumentStorageWriteOperation::InsertDocuments(docs)
            };
            storage.update(op).await?;
        }
        storage.commit(Offset(0)).await?;
        drop(storage);

        let storage = ReadCollectionStorage::new(
            global,
            dir.path().join("collection"),
            CollectionId::try_new("test")?,
        )?;
        let found = storage
            .get_documents_by_ids(vec![DocumentId(1), DocumentId(2), DocumentId(3)])
            .await?;
        assert_eq!(versions(found), latest_versions());
    }
    Ok(())
}

#[tokio::test]
async fn write_insert_many_sorts_and_keeps_last_write() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let global = Arc::new(WriteStorage::try_new(dir.path().join("global")).await?);
    let storage = WriteCollectionStorage::new(global, dir.path().join("collection")).await?;
    let doc = |id: u64, version: &str| {
        (
            DocumentId(id),
            ZeboDocument::new(
                Cow::Owned(id.to_string()),
                Cow::Owned(format!(r#"{{"version":"{version}"}}"#)),
            ),
        )
    };
    storage
        .insert_many(&[doc(3, "v1"), doc(1, "v1"), doc(2, "v1"), doc(1, "v2")])
        .await?;
    let found = storage
        .stream_documents(vec![DocumentId(1), DocumentId(2), DocumentId(3)])
        .await
        .map(|(id, doc)| (id, Arc::new(doc)))
        .collect()
        .await;
    assert_eq!(versions(found), latest_versions());
    Ok(())
}
