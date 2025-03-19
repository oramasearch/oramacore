use std::{
    fmt::Debug,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    types::CollectionId,
};
use anyhow::{Context, Result};
use ptrie::Trie;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::sync::RwLock;

use super::{KVWriteOperation, Offset, OperationSender, WriteOperation};

#[derive(Clone)]
pub struct KVConfig {
    pub data_dir: PathBuf,
    pub sender: Option<OperationSender>,
}

pub struct KV {
    data: RwLock<Trie<u8, String>>,
    data_dir: PathBuf,

    sender: Option<OperationSender>,
    last_offset: AtomicU64,
    committed_offset: u64,
}

impl KV {
    pub fn try_load(config: KVConfig) -> Result<Self> {
        create_if_not_exists(&config.data_dir).context("Cannot create data directory")?;

        let info = config.data_dir.join("info.json");
        let info = BufferedFile::open(info).and_then(|file| file.read_json_data::<KVInfo>());
        let (tree, offset) = match info {
            Ok(info) => {
                let KVInfo::V1(info) = info;
                let tree = BufferedFile::open(info.path_to_kv)
                    .context("Cannot open previous kv info")?
                    .read_bincode_data::<Trie<u8, String>>()
                    .context("Cannot read previous kv info")?;
                (tree, info.current_offset)
            }
            Err(_) => (Trie::new(), 0),
        };

        Ok(Self {
            data: RwLock::new(tree),
            data_dir: config.data_dir,
            committed_offset: offset,
            last_offset: AtomicU64::new(offset),
            sender: config.sender,
        })
    }

    pub async fn insert<V: Clone + Serialize + DeserializeOwned>(
        &self,
        key: String,
        value: V,
    ) -> Result<()> {
        let value = serde_json::to_string(&value).context("Cannot serialize value")?;
        self.data
            .write()
            .await
            .insert(key.as_bytes().iter().cloned(), value.clone());

        if let Some(sender) = self.sender.as_ref() {
            let op = WriteOperation::KV(KVWriteOperation::Create(key, value));
            sender.send(op).await?;
        }

        self.last_offset.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    pub async fn get<V: Clone + Serialize + DeserializeOwned>(
        &self,
        key: &str,
    ) -> Option<Result<V>> {
        let read_ref = self.data.read().await;

        match read_ref.get(key.as_bytes().iter().cloned()) {
            Some(value) => {
                let value = serde_json::from_str(value).context("Cannot deserialize value");
                Some(value)
            }
            None => None,
        }
    }

    pub async fn remove(&self, key: &str) -> Result<Option<()>> {
        let data = self
            .data
            .write()
            .await
            .remove(key.as_bytes().iter().cloned());

        if let Some(sender) = self.sender.as_ref() {
            let op = WriteOperation::KV(KVWriteOperation::Delete(key.to_string()));
            sender.send(op).await?;
        }

        Ok(data.map(|_| ()))
    }

    pub async fn delete_with_prefix(&self, prefix: &str) -> Result<()> {
        // This is not atomic, but it's fine for now
        let read_ref = self.data.write().await;
        let keys: Vec<_> = read_ref
            .scan_postfix_keys(prefix.as_bytes().iter().cloned())
            .collect();
        drop(read_ref);

        for key in keys {
            let key = String::from_utf8_lossy(&key).to_string();
            self.remove(&key).await?;
        }

        Ok(())
    }

    #[must_use = "Use `remove` if you want to get the value back"]
    pub async fn remove_and_get<V: Clone + Serialize + DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<Result<V>>> {
        let data = self
            .data
            .write()
            .await
            .remove(key.as_bytes().iter().cloned());
        match data {
            Some(data) => {
                let value = serde_json::from_str(&data)
                    .map_err(|e| anyhow::anyhow!("Cannot deserialize value: {}", e));
                self.last_offset.fetch_add(1, Ordering::Relaxed);

                if let Some(sender) = self.sender.as_ref() {
                    let op = WriteOperation::KV(KVWriteOperation::Delete(key.to_string()));
                    sender.send(op).await?;
                }

                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    pub async fn prefix_scan<V: Clone + Serialize + DeserializeOwned>(
        &self,
        prefix: &str,
    ) -> Result<Vec<V>> {
        let read_ref = self.data.read().await;

        let data = read_ref.find_postfixes(prefix.as_bytes().iter().cloned());

        data.into_iter()
            .map(|value| {
                serde_json::from_str(value)
                    .context("Cannot deserialize prefix value in segments scan")
            })
            .collect()
    }

    pub async fn update(&self, offset: Offset, op: KVWriteOperation) -> Result<()> {
        self.last_offset.store(offset.0, Ordering::Relaxed);

        match op {
            KVWriteOperation::Create(key, value) => {
                self.data
                    .write()
                    .await
                    .insert(key.as_bytes().iter().cloned(), value.clone());
            }
            KVWriteOperation::Delete(key) => {
                self.data
                    .write()
                    .await
                    .remove(key.as_bytes().iter().cloned());
            }
        };

        Ok(())
    }

    pub async fn commit(&self) -> Result<()> {
        create_if_not_exists(&self.data_dir).context("Cannot create data directory")?;

        // We don't allow any new write operation during the commit
        let data = self.data.write().await;

        let current_offset = self.last_offset.load(Ordering::Relaxed);

        if current_offset == self.committed_offset {
            // Nothing to commit
            return Ok(());
        }

        let new_path = self.data_dir.join(format!("kv-{}.bin", current_offset));
        BufferedFile::create_or_overwrite(new_path.clone())
            .context("Cannot create previous kv info")?
            .write_bincode_data(&*data)
            .context("Cannot write previous kv info")?;

        let info = self.data_dir.join("info.json");

        BufferedFile::create_or_overwrite(info)
            .context("Cannot create previous kv info")?
            .write_json_data(&KVInfo::V1(KVInfoV1 {
                path_to_kv: new_path,
                current_offset,
            }))
            .context("Cannot write previous kv info")?;

        Ok(())
    }
}

pub fn format_key(collection_id: CollectionId, key: &str) -> String {
    format!("{}:{}", collection_id.0, key)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum KVInfo {
    V1(KVInfoV1),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KVInfoV1 {
    path_to_kv: PathBuf,
    current_offset: u64,
}

impl Debug for KV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elements: Vec<_> = match self.data.try_read() {
            Ok(inner) => inner
                .iter()
                .map(|(k, v)| (String::from_utf8_lossy(&k).to_string(), v))
                .collect(),
            Err(_) => vec![("<locked>".to_string(), "<locked>".to_string())],
        };

        f.debug_struct("KV")
            .field("data", &elements)
            .field("data_dir", &self.data_dir)
            .field("sender", &"..")
            .field("last_offset", &self.last_offset)
            .field("committed_offset", &self.committed_offset)
            .finish()
    }
}
