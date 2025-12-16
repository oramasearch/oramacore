use std::{
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc},
};

use anyhow::{Context, Result};
use futures::Stream;
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tracing::error;
use zebo::Zebo;

use crate::{
    collection_manager::sides::write::document_storage::{DocumentStorage, ZeboDocument},
    lock::OramaAsyncLock,
    types::{DocumentId, RawJSONDocument},
};

// 1GB
const PAGE_SIZE: u64 = 1024 * 1024 * 1024;

pub struct CollectionDocumentStorage {
    base_dir: PathBuf,
    zebo: Arc<OramaAsyncLock<Zebo<1_000_000, PAGE_SIZE, DocumentId>>>,
    global_document_storage: Arc<DocumentStorage>,
    collection_document_id: AtomicU64,
    last_global_document_id: u64,
}

impl CollectionDocumentStorage {
    pub async fn new(
        global_document_storage: Arc<DocumentStorage>,
        base_dir: PathBuf,
    ) -> Result<Self> {
        create_if_not_exists(&base_dir).context("Cannot create base dir for collection storage")?;

        let dump: Dump = if std::fs::exists(base_dir.join("doc_storage.json")).unwrap_or(false) {
            BufferedFile::open(base_dir.join("doc_storage.json"))
                .context("Cannot read doc_storage.json")?
                .read_json_data()
                .context("Cannot read doc_storage.json")?
        } else {
            let doc_id = global_document_storage
                .get_last_inserted_document_id()
                .await
                .context("Cannot get last document id from global storage")?;
            let document_id = doc_id.map(|id| id.0 + 1).unwrap_or(1);
            Dump::V1(DumpV1 {
                document_id,
                last_global_document_id: doc_id.map(|id| id.0).unwrap_or(0),
            })
        };

        let (collection_document_id, last_global_document_id) = match dump {
            Dump::V1(d) => (AtomicU64::new(d.document_id), d.last_global_document_id),
        };

        Ok(Self {
            base_dir: base_dir.clone(),
            zebo: Arc::new(OramaAsyncLock::new(
                "collection_zebo",
                Zebo::try_new(base_dir).context("Cannot create collection zebo")?,
            )),
            global_document_storage,
            collection_document_id,
            last_global_document_id,
        })
    }

    pub fn get_next_document_id(&self) -> DocumentId {
        let next_id = self
            .collection_document_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DocumentId(next_id)
    }

    pub async fn insert_many(&self, docs: &[(DocumentId, ZeboDocument<'_>)]) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }

        let mut zebo = self.zebo.write("insert_many").await;
        let space = zebo
            .reserve_space_for(docs)
            .context("Cannot reserve space in zebo")?;
        drop(zebo);

        space.write_all().context("Cannot write documents")?;

        Ok(())
    }

    fn split_into_local_and_global(
        &self,
        ids: Vec<DocumentId>,
    ) -> (Vec<DocumentId>, Vec<DocumentId>) {
        let (local_ids, global_ids): (Vec<DocumentId>, Vec<DocumentId>) = ids
            .into_iter()
            .partition(|id| *id > DocumentId(self.last_global_document_id));
        (local_ids, global_ids)
    }

    pub async fn remove(&self, ids: Vec<DocumentId>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let (local_ids, global_ids) = self.split_into_local_and_global(ids);

        if !local_ids.is_empty() {
            let mut zebo = self.zebo.write("remove").await;
            zebo.remove_documents(local_ids, false)
                .context("cannot remove documents from collection document id")?;
        }

        if !global_ids.is_empty() {
            #[allow(deprecated)]
            self.global_document_storage.remove(global_ids).await;
        }

        Ok(())
    }

    pub async fn stream_documents(
        &self,
        ids: Vec<DocumentId>,
    ) -> impl Stream<Item = (DocumentId, RawJSONDocument)> {
        let (tx, rx) = mpsc::channel(100);

        let (local_ids, global_ids) = self.split_into_local_and_global(ids);

        let local_tx = tx.clone();
        let zebo = self.zebo.clone();
        tokio::spawn(async move {
            let zebo = zebo.read("stream_documents").await;

            let docs_iter = match zebo.get_documents(local_ids) {
                Ok(a) => a,
                Err(e) => {
                    error!(error = ?e, "Cannot get documents");
                    return;
                }
            };

            for doc in docs_iter {
                let (doc_id, data) = match doc {
                    Ok(doc) => doc,
                    Err(e) => {
                        error!(error = ?e, "Cannot get document");
                        continue;
                    }
                };

                let doc = match ZeboDocument::from_bytes(data)
                    .context("Cannot convert bytes to document")
                {
                    Ok(doc) => doc,
                    Err(e) => {
                        error!(error = ?e, "Cannot convert bytes to document");
                        continue;
                    }
                };
                let inner = RawValue::from_string(doc.1.to_string())
                    .context("Cannot convert bytes to document")
                    .unwrap();
                let d = RawJSONDocument {
                    id: Some(doc.0.to_string()),
                    inner,
                };

                if let Err(e) = local_tx.send((doc_id, d)).await {
                    error!(error = ?e, "Cannot send document data. Stopped stream_documents");
                    break;
                }
            }
        });

        let global_tx = tx.clone();
        let global_storage = self.global_document_storage.clone();
        tokio::spawn(async move {
            #[allow(deprecated)]
            let mut stream = global_storage.stream_documents(global_ids).await;
            while let Some(doc) = stream.next().await {
                if let Err(e) = global_tx.send(doc).await {
                    error!(error = ?e, "Cannot send document data. Stopped stream_documents");
                    break;
                }
            }
        });

        ReceiverStream::new(rx)
    }

    pub fn commit(&self) -> Result<()> {
        let dump = Dump::V1(DumpV1 {
            document_id: self
                .collection_document_id
                .load(std::sync::atomic::Ordering::SeqCst),
            last_global_document_id: self.last_global_document_id,
        });
        BufferedFile::create_or_overwrite(self.base_dir.join("doc_storage.json"))
            .context("Cannot create or overwrite doc_storage.json")?
            .write_json_data(&dump)
            .context("Cannot write doc_storage.json")?;
        Ok(())
    }
}

#[derive(Deserialize, Serialize, Debug)]
enum Dump {
    V1(DumpV1),
}

#[derive(Deserialize, Serialize, Debug)]
struct DumpV1 {
    document_id: u64,
    last_global_document_id: u64,
}
