pub mod s3;
use crate::lock::OramaAsyncLock;
use crate::types::{CollectionId, DeleteDocuments, DocumentList, IndexId};
use anyhow::{Context, Result};
use oramacore_lib::fs::create_if_not_exists;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{sync::Arc, time};
use thiserror::Error;
use tokio::sync::Mutex;
use tokio::time::MissedTickBehavior;
use tracing::{error, info};

#[derive(Error, Debug)]
pub enum DatasourceError {
    #[error("Sync failed: {0}")]
    SyncFailed(#[from] anyhow::Error),
}

impl From<tokio::sync::mpsc::error::SendError<SyncUpdate>> for DatasourceError {
    fn from(e: tokio::sync::mpsc::error::SendError<SyncUpdate>) -> Self {
        DatasourceError::SyncFailed(anyhow::anyhow!(e))
    }
}

pub enum Operation {
    Insert(DocumentList),
    Delete(DeleteDocuments),
}

pub struct SyncUpdate {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub operation: Operation,
}

struct DatasourceWrapper {
    in_use: AtomicBool,
    fetcher: Fetcher,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum Fetcher {
    S3(s3::S3Fetcher),
}

impl Fetcher {
    pub async fn validate(&self) -> Result<()> {
        match self {
            Fetcher::S3(s3_fetcher) => s3_fetcher.validate_credentials().await,
        }
    }

    pub async fn delete(
        &self,
        base_dir: &PathBuf,
        collection_id: CollectionId,
        index_id: IndexId,
    ) -> Result<()> {
        match self {
            Fetcher::S3(s3_fetcher) => {
                s3_fetcher
                    .remove_resy_db(base_dir, collection_id, index_id)
                    .await
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableDatasourceEntry {
    collection_id: CollectionId,
    index_id: IndexId,
    fetcher: Fetcher,
}

type CollectionDatasources = HashMap<(CollectionId, IndexId), Arc<DatasourceWrapper>>;

pub struct DatasourceStorage {
    base_dir: PathBuf,
    file_path: PathBuf,
    fetcher_trigg_sender: tokio::sync::mpsc::Sender<(CollectionId, IndexId)>,
    fetcher_trigg_receiver: Mutex<Option<tokio::sync::mpsc::Receiver<(CollectionId, IndexId)>>>,
    map: Arc<OramaAsyncLock<CollectionDatasources>>,
}

impl DatasourceStorage {
    pub fn try_new(base_dir: &PathBuf) -> Result<Self> {
        create_if_not_exists(base_dir).context("Cannot create datasource directory")?;

        let file_path = base_dir.join("datasources.json");
        let map = Self::load_from_disk(&file_path)?;

        let (fetcher_trigg_sender, fetcher_trigg_receiver) = tokio::sync::mpsc::channel(100);

        Ok(Self {
            base_dir: base_dir.clone(),
            file_path,
            map: Arc::new(OramaAsyncLock::new("datasource_storage", map)),
            fetcher_trigg_sender,
            fetcher_trigg_receiver: Mutex::new(Some(fetcher_trigg_receiver)),
        })
    }

    pub async fn insert(
        &self,
        collection_id: CollectionId,
        index_id: IndexId,
        fetcher: Fetcher,
    ) -> Result<()> {
        fetcher.validate().await?;

        let mut m = self.map.write("insert").await;
        match m.entry((collection_id, index_id)) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Arc::new(DatasourceWrapper {
                    in_use: AtomicBool::new(false),
                    fetcher,
                }));
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                unreachable!("nop")
            }
        }

        self.save_to_disk_sync(&m)?;

        drop(m);

        if let Err(e) = self
            .fetcher_trigg_sender
            .try_send((collection_id, index_id))
        {
            error!("Failed to send datasource sync request: {}", e);
        }

        Ok(())
    }

    pub async fn remove_collection(&self, collection_id: CollectionId) -> Result<()> {
        let mut m = self.map.write("remove_collection").await;
        let keys_to_remove: Vec<_> = m.keys().filter(|k| k.0 == collection_id).cloned().collect();

        if keys_to_remove.is_empty() {
            drop(m);
            return Ok(());
        }

        let fetchers_to_delete: Vec<_> = keys_to_remove
            .iter()
            .filter_map(|key| m.remove(key).map(|f| (key.clone(), f)))
            .collect();

        self.save_to_disk_sync(&m)?;

        for (key, f) in fetchers_to_delete {
            if let Err(e) = f.fetcher.delete(&self.base_dir, key.0, key.1).await {
                error!("Failed to delete datasource for {:?}: {}", key, e);
            }
        }

        drop(m);

        Ok(())
    }

    pub async fn remove_index(&self, collection_id: CollectionId, index_id: IndexId) -> Result<()> {
        let mut m = self.map.write("remove_index").await;
        let f = m.remove(&(collection_id, index_id));

        if let Some(fetcher) = f {
            self.save_to_disk_sync(&m)?;

            if let Err(e) = fetcher
                .fetcher
                .delete(&self.base_dir, collection_id, index_id)
                .await
            {
                error!(
                    "Failed to delete datasource for ({}, {}): {}",
                    collection_id, index_id, e
                );
            }
        }

        drop(m);

        Ok(())
    }

    pub async fn exists(&self, collection_id: CollectionId, index_id: IndexId) -> bool {
        let m = self.map.read("exists").await;
        m.contains_key(&(collection_id, index_id))
    }

    fn load_from_disk(file_path: &PathBuf) -> Result<CollectionDatasources> {
        if file_path.exists() {
            let file = std::fs::File::open(file_path)
                .with_context(|| format!("Cannot open datasources.json at {:?}", file_path))?;
            let serializable_vec: Vec<SerializableDatasourceEntry> = serde_json::from_reader(file)
                .with_context(|| {
                    format!("Cannot deserialize datasources.json at {:?}", file_path)
                })?;

            Ok(serializable_vec
                .into_iter()
                .map(|entry| {
                    (
                        (entry.collection_id, entry.index_id),
                        Arc::new(DatasourceWrapper {
                            in_use: AtomicBool::new(false),
                            fetcher: entry.fetcher,
                        }),
                    )
                })
                .collect())
        } else {
            Ok(HashMap::new())
        }
    }

    fn save_to_disk_sync(&self, map: &CollectionDatasources) -> Result<()> {
        let file_path_tmp = self.file_path.with_extension("tmp.json");

        let serializable_vec: Vec<SerializableDatasourceEntry> = map
            .iter()
            .map(
                |((collection_id, index_id), value)| SerializableDatasourceEntry {
                    collection_id: *collection_id,
                    index_id: *index_id,
                    fetcher: value.fetcher.clone(),
                },
            )
            .collect();

        let file = std::fs::File::create(&file_path_tmp)
            .with_context(|| format!("Cannot create datasources.json at {:?}", file_path_tmp))?;

        serde_json::to_writer_pretty(file, &serializable_vec)
            .with_context(|| format!("Cannot serialize datasources to {:?}", file_path_tmp))?;

        std::fs::rename(&file_path_tmp, &self.file_path).with_context(|| {
            format!("Cannot rename {:?} to {:?}", file_path_tmp, self.file_path)
        })?;

        Ok(())
    }

    pub async fn start_datasource_loop(
        &self,
        interval: time::Duration,
        datasource_update_sender: tokio::sync::mpsc::Sender<SyncUpdate>,
        mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
        stop_done_sender: tokio::sync::mpsc::Sender<()>,
    ) -> Result<()> {
        let map = self.map.clone();
        let base_dir = self.base_dir.clone();
        let mut fetcher_trigg_receiver = self
            .fetcher_trigg_receiver
            .lock()
            .await
            .take()
            .context("start_datasource_loop can only be called once")?;

        tokio::task::spawn(async move {
            let start = tokio::time::Instant::now() + interval;
            let mut interval = tokio::time::interval_at(start, interval);
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

            'outer: loop {
                tokio::select! {
                    _ = stop_receiver.recv() => {
                        info!("Stopping datasource loop");
                        break 'outer;
                    }
                    _ = interval.tick() => {
                        info!("Running periodic datasource sync");
                        let mut v = Vec::new();
                        let lock = map.read("run_interval").await;
                        for ((collection_id, index_id), kind) in lock.iter() {
                            if kind.in_use.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                                v.push((DatasourceStorage::run_fetcher_task(
                                    *collection_id,
                                    *index_id,
                                    base_dir.clone(),
                                    kind.fetcher.clone(),
                                    datasource_update_sender.clone(),
                                ), kind.clone()));
                            }
                        }
                        drop(lock);
                        tokio::spawn(async move {
                            for (handle, kind) in v {
                                if let Err(e) = handle.await {
                                    error!("Datasource sync task failed: {:?}", e);
                                }
                                kind.in_use.store(false, Ordering::Release);
                            }
                        });
                    }
                Some((collection_id, index_id)) = fetcher_trigg_receiver.recv() => {
                        let mut v = Vec::new();
                        let lock = map.read("run_explicit").await;
                        if let Some(kind) = lock.get(&(collection_id, index_id)) {
                            if kind.in_use.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                                v.push((DatasourceStorage::run_fetcher_task(
                                    collection_id,
                                    index_id,
                                    base_dir.clone(),
                                    kind.fetcher.clone(),
                                    datasource_update_sender.clone(),
                                    ), kind.clone()));
                            }

                        }
                        drop(lock);

                        tokio::spawn(async move {
                            for (handle , kind )in v {
                                if let Err(e) = handle.await {
                                    error!("Datasource sync task failed: {:?}", e);
                                }
                                kind.in_use.store(false, Ordering::Release);
                            }
                        });
                    }
                };
            }

            drop(datasource_update_sender);

            if let Err(e) = stop_done_sender.send(()).await {
                error!(error = ?e, "Cannot send stop signal to writer side");
            }
        });

        Ok(())
    }

    fn run_fetcher_task(
        collection_id: CollectionId,
        index_id: IndexId,
        datasource_dir_clone: PathBuf,
        datasource: Fetcher,
        sync_sender: tokio::sync::mpsc::Sender<SyncUpdate>,
    ) -> tokio::task::JoinHandle<()> {
        let task_sender_clone = sync_sender.clone();

        match &datasource {
            Fetcher::S3(s3_datasource) => {
                let s3_datasource = s3_datasource.clone();
                // TODO: update resy to implement async correctly
                tokio::task::spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();

                    rt.block_on(async {
                        match s3_datasource
                            .sync(
                                datasource_dir_clone,
                                collection_id,
                                index_id,
                                task_sender_clone,
                            )
                            .await
                        {
                            Ok(_) => {}
                            Err(DatasourceError::SyncFailed(e)) => {
                                error!(error = ?e, "Failed to sync S3 datasource");
                            }
                        }
                    })
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::utils::setup_s3_container;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_insert_fail_invalid_credentials() {
        let temp_dir = tempdir().unwrap();
        let storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();

        let fetcher = Fetcher::S3(s3::S3Fetcher {
            bucket: "bucket1".to_string(),
            region: "us-east-1".to_string(),
            access_key_id: "invalid".to_string(),
            secret_access_key: "invalid".to_string(),
            endpoint_url: None,
        });

        // invalid validation since s3 is not running
        let result = storage.insert(coll_id, index_id, fetcher.clone()).await;
        assert!(result.is_err());

        assert!(!storage.exists(coll_id, index_id).await);
    }

    #[tokio::test]
    async fn test_datasource_persistence() {
        let (_s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;
        let temp_dir = tempdir().unwrap();
        let temp_path = &temp_dir.path().to_path_buf();
        let storage = DatasourceStorage::try_new(temp_path).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();

        let fetcher = Fetcher::S3(s3::S3Fetcher {
            bucket: bucket_name,
            region: "us-east-1".to_string(),
            access_key_id: "test".to_string(),
            secret_access_key: "test".to_string(),
            endpoint_url: Some(endpoint_url),
        });

        // commit on insert
        storage
            .insert(coll_id, index_id, fetcher.clone())
            .await
            .unwrap();

        // load from disk on create
        let new_storage = DatasourceStorage::try_new(temp_path).unwrap();
        assert!(new_storage.exists(coll_id, index_id).await);
    }

    #[tokio::test]
    async fn test_insert_sends_trigger() {
        let (_s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;
        let temp_dir = tempdir().unwrap();
        let storage = DatasourceStorage::try_new(&temp_dir.path().to_path_buf()).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();

        let fetcher = Fetcher::S3(s3::S3Fetcher {
            bucket: bucket_name,
            region: "us-east-1".to_string(),
            access_key_id: "test".to_string(),
            secret_access_key: "test".to_string(),
            endpoint_url: Some(endpoint_url),
        });

        storage
            .insert(coll_id, index_id, fetcher.clone())
            .await
            .unwrap();

        let mut receiver = storage.fetcher_trigg_receiver.lock().await.take().unwrap();
        let (received_coll_id, received_index_id) = receiver.recv().await.take().unwrap();
        assert_eq!(received_coll_id, coll_id);
        assert_eq!(received_index_id, index_id);
    }

    #[tokio::test]
    async fn test_remove_deletes_resy_file() {
        let (_s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;
        let temp_dir = tempdir().unwrap();
        let temp_path = &temp_dir.path().to_path_buf();
        let storage = DatasourceStorage::try_new(temp_path).unwrap();

        let coll_id = CollectionId::try_new("coll1".to_string()).unwrap();
        let index_id = IndexId::try_new("idx1".to_string()).unwrap();

        let fetcher = Fetcher::S3(s3::S3Fetcher {
            bucket: bucket_name,
            region: "us-east-1".to_string(),
            access_key_id: "test".to_string(),
            secret_access_key: "test".to_string(),
            endpoint_url: Some(endpoint_url),
        });

        let db_name = format!("{}_{}.db", coll_id, index_id);
        let db_path = temp_path.join(db_name);
        std::fs::File::create(&db_path).unwrap();
        assert!(db_path.exists());

        storage
            .insert(coll_id, index_id, fetcher.clone())
            .await
            .unwrap();

        storage.remove_index(coll_id, index_id).await.unwrap();

        assert!(!db_path.exists());
    }
}
