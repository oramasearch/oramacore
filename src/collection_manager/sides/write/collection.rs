use std::{collections::HashMap, ops::Deref, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Sender, RwLock, RwLockReadGuard};
use tracing::{info, warn};

use crate::{
    ai::OramaModel,
    collection_manager::sides::{
        hooks::HooksRuntime, index::CreateIndexRequest, CollectionWriteOperation, OperationSender,
        OramaModelSerializable, WriteOperation,
    },
    file_utils::BufferedFile,
    nlp::{locales::Locale, NLPService, TextParser},
    types::{ApiKey, CollectionId, DescribeCollectionResponse, DocumentId, IndexId},
};

use super::{embedding::MultiEmbeddingCalculationRequest, index::{CreateIndexEmbeddingFieldDefintionRequest, EmbeddingStringCalculation, Index}};

pub struct CollectionWriter {
    pub(super) id: CollectionId,
    description: Option<String>,
    default_locale: Locale,
    write_api_key: ApiKey,
    read_api_key: ApiKey,
    embeddings_model: OramaModel,
    embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
    indexes: RwLock<HashMap<IndexId, Index>>,
    temp_indexes: RwLock<HashMap<IndexId, Index>>,
    op_sender: OperationSender,
}

impl CollectionWriter {
    pub fn empty(
        id: CollectionId,
        description: Option<String>,
        write_api_key: ApiKey,
        read_api_key: ApiKey,
        default_locale: Locale,
        embeddings_model: OramaModel,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        op_sender: OperationSender,
    ) -> Self {
        Self {
            id,
            description,
            write_api_key,
            read_api_key,
            default_locale,
            embedding_sender,
            embeddings_model,
            indexes: Default::default(),
            temp_indexes: Default::default(),
            op_sender,
        }
    }

    pub async fn try_load(
        path: PathBuf,
        hooks_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        op_sender: OperationSender,
    ) -> Result<Self> {
        let dump: CollectionDump = BufferedFile::open(path.join("info.json"))
            .context("Cannot open info.json file")?
            .read_json_data()
            .context("Cannot deserialize collection info")?;

        let CollectionDump::V1(dump) = dump;

        let id = dump.id;
        let description = dump.description;
        let write_api_key =
            ApiKey::try_new(dump.write_api_key).context("Cannot create write api key")?;
        let read_api_key =
            ApiKey::try_new(dump.read_api_key).context("Cannot create read api key")?;
        let default_locale = dump.default_locale;

        let mut indexes = HashMap::new();
        for index_id in dump.indexes {
            let index = Index::try_load(
                id,
                index_id,
                path.join("indexes").join(index_id.as_str()),
                op_sender.clone(),
                hooks_runtime.clone(),
                embedding_sender.clone(),
            )
            .context("Cannot load index")?;
            indexes.insert(index_id, index);
        }

        Ok(Self {
            id,
            description,
            write_api_key,
            read_api_key,
            default_locale,
            embedding_sender,
            embeddings_model: dump.embeddings_model.0,
            indexes: RwLock::new(HashMap::new()),
            temp_indexes: RwLock::new(HashMap::new()),
            op_sender,
        })
    }

    pub async fn commit(&mut self, data_dir: PathBuf) -> Result<()> {
        info!(coll_id= ?self.id, "Committing collection");

        std::fs::create_dir_all(&data_dir).context("Cannot create collection directory")?;

        let indexes_lock = self.indexes.get_mut();
        let temp_indexes_lock = self.temp_indexes.get_mut();

        let indexes = indexes_lock.keys().copied().collect::<Vec<_>>();
        let indexes_path = data_dir.join("indexes");
        for (id, index) in indexes_lock.iter_mut() {
            index.commit(indexes_path.join(id.as_str())).await?;
        }

        let temporary_indexes = temp_indexes_lock.keys().copied().collect::<Vec<_>>();
        let indexes_path = data_dir.join("temp_indexes");
        for (id, index) in temp_indexes_lock.iter_mut() {
            index.commit(indexes_path.join(id.as_str())).await?;
        }

        let dump = CollectionDump::V1(CollectionDumpV1 {
            id: self.id,
            description: self.description.clone(),
            write_api_key: self.write_api_key.expose().to_string(),
            read_api_key: self.read_api_key.expose().to_string(),
            default_locale: self.default_locale,
            embeddings_model: OramaModelSerializable(self.embeddings_model),
            indexes,
            temporary_indexes,
        });

        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        Ok(())
    }

    pub async fn create_index(&self, request: CreateIndexRequest) -> Result<()> {
        let index_id = request.id;

        let mut indexes = self.indexes.write().await;
        if indexes.contains_key(&index_id) {
            bail!("Index with id {} already exists", index_id);
        }

        self.op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::CreateIndex2 {
                    index_id,
                    locale: self.default_locale,
                },
            ))
            .await
            .context("Cannot send create index operation")?;

        let index = Index::empty(
            self.id,
            request,
            self.embedding_sender.clone(),
            self.get_text_parser(self.default_locale),
            self.embeddings_model,
            self.op_sender.clone(),
        )
        .await
        .context("Cannot create index")?;

        indexes.insert(index_id, index);
        drop(indexes);

        Ok(())
    }

    pub async fn create_temp_index(
        &self,
        copy_from: IndexId,
        new_index_id: IndexId,
        index_embeddings: Vec<CreateIndexEmbeddingFieldDefintionRequest>,
    ) -> Result<()> {
        let lock = self.indexes.write().await;
        let Some(copy_from) = lock.get(&copy_from) else {
            bail!("Index with id {} not found", copy_from);
        };

        let index = Index::empty(
            self.id,
            CreateIndexRequest {
                id: new_index_id,
                embedding_field_definition: index_embeddings,
            },
            self.embedding_sender.clone(),
            self.get_text_parser(self.default_locale),
            self.embeddings_model,
            self.op_sender.clone(),
        )
        .await
        .context("Cannot create index")?;

        Ok(())
    }

    pub async fn delete_index(&self, index_id: IndexId) -> Result<()> {
        let mut indexes = self.indexes.write().await;
        let index = match indexes.remove(&index_id) {
            Some(index) => index,
            None => {
                warn!(collection_id= ?self.id, "Index not found. Ignored");
                return Ok(());
            }
        };
        drop(indexes);

        self.op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::DeleteIndex2 { index_id },
            ))
            .await
            .context("Cannot send delete index operation")?;

        // index.remove_from_fs().await;

        Ok(())
    }
    /*
        pub async fn create_temporary_index_from(
            &self,
            index_id: IndexId,
            temporary_index_id: IndexId,
            reindex_config: ReindexConfig,
        ) -> Result<()> {
            let index = self.get_index(index_id).await.context("Cannot get index")?;

            let locale: Locale = reindex_config
                .language
                .map(|l| l.into())
                .unwrap_or(index.get_locale());

            let embedding_field_definition = index
                .get_embedding_field_definition()
                .await
                .context("Cannot get embedding field definition")?;

            let req = CreateIndexRequest {
                id: temporary_index_id,
                embedding_field_definition,
            };

            let temp_index = Index::empty(
                self.id,
                req,
                self.embedding_sender.clone(),
                self.get_text_parser(locale),
                self.op_sender.clone(),
            )
            .await
            .context("Cannot create index")?;

            let mut temp_indexes = self.temp_indexes.write().await;
            temp_indexes.insert(temporary_index_id, temp_index);
            drop(temp_indexes);

            Ok(())
        }
    */
    pub async fn get_temporary_index<'s, 'index>(
        &'s self,
        id: IndexId,
    ) -> Option<IndexReadLock<'index>>
    where
        's: 'index,
    {
        let lock = self.temp_indexes.read().await;
        IndexReadLock::try_new(lock, id)
    }

    pub async fn promote_temp_index(&self, index_id: IndexId, temp_index: IndexId) -> Result<()> {
        let mut temp_index_lock = self.indexes.write().await;
        let mut index_lock = self.indexes.write().await;

        let temp_index = match temp_index_lock.remove(&temp_index) {
            Some(index) => index,
            None => bail!("Temporary index not found"),
        };
        match index_lock.insert(index_id, temp_index) {
            Some(_) => {
                info!(coll_id= ?self.id, "Index replaced");
            }
            None => {
                warn!(coll_id= ?self.id, "Index replaced but before it was not found. This is just a warning");
            }
        }
        drop(temp_index_lock);
        drop(index_lock);

        panic!();
    }

    pub async fn get_document_ids(&self) -> Vec<DocumentId> {
        let mut doc_id = vec![];
        let indexes = self.indexes.read().await;
        for (_, index) in indexes.iter() {
            let index_doc_ids = index.get_document_ids().await;
            doc_id.extend(index_doc_ids);
        }

        doc_id
    }

    pub fn check_write_api_key(&self, api_key: ApiKey) -> Result<()> {
        if self.write_api_key == api_key {
            Ok(())
        } else {
            Err(anyhow!("Invalid write api key"))
        }
    }

    pub async fn as_dto(&self) -> DescribeCollectionResponse {
        let mut indexes_desc = vec![];
        let mut document_count = 0_usize;
        let indexes = self.indexes.read().await;
        for index in indexes.values() {
            let index_desc = index.describe().await;
            document_count += index_desc.document_count;
            indexes_desc.push(index_desc);
        }
        drop(indexes);

        DescribeCollectionResponse {
            id: self.id,
            description: self.description.clone(),
            document_count,
            indexes: indexes_desc,
        }
    }

    pub async fn get_index<'s, 'index>(&'s self, id: IndexId) -> Option<IndexReadLock<'index>>
    where
        's: 'index,
    {
        let lock = self.indexes.read().await;
        IndexReadLock::try_new(lock, id)
    }

    fn get_text_parser(&self, locale: Locale) -> Arc<TextParser> {
        // TextParser is expensive to create, so we cache it
        // TODO: add a cache
        let parser = TextParser::from_locale(locale);
        Arc::new(parser)
    }

    pub async fn remove_from_fs(self, path: PathBuf) {
        match std::fs::remove_dir_all(&path) {
            Ok(_) => {}
            Err(e) => {
                warn!(coll_id= ?self.id, "Cannot remove collection directory. Ignored: {:?}", e);
            }
        };
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
enum CollectionDump {
    #[serde(rename = "1")]
    V1(CollectionDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionDumpV1 {
    id: CollectionId,
    description: Option<String>,
    write_api_key: String,
    read_api_key: String,
    default_locale: Locale,
    embeddings_model: OramaModelSerializable,
    indexes: Vec<IndexId>,
    temporary_indexes: Vec<IndexId>,
}

pub struct IndexReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<IndexId, Index>>,
    id: IndexId,
}

impl<'guard> IndexReadLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<IndexId, Index>>,
        id: IndexId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(Self { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for IndexReadLock<'_> {
    type Target = Index;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}
