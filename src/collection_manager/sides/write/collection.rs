use std::{collections::HashMap, ops::Deref, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Sender, RwLock, RwLockReadGuard};
use tracing::{info, warn};

use crate::{
    collection_manager::sides::{
        hooks::HooksRuntime, index::CreateIndexRequest, CollectionWriteOperation, OperationSender,
        WriteOperation,
    },
    file_utils::BufferedFile,
    nlp::{locales::Locale, NLPService, TextParser},
    types::{ApiKey, CollectionId, DescribeCollectionResponse, DocumentId, IndexId, ReindexConfig},
};

use super::{embedding::MultiEmbeddingCalculationRequest, index::Index};

pub mod doc_id_storage;

pub const DEFAULT_EMBEDDING_FIELD_NAME: &str = "___orama_auto_embedding";

pub struct CollectionWriter {
    pub(super) id: CollectionId,
    description: Option<String>,
    default_locale: Locale,
    // filter_fields: RwLock<HashMap<FieldId, CollectionFilterField>>,
    // score_fields: RwLock<HashMap<FieldId, CollectionScoreField>>,
    write_api_key: ApiKey,
    // collection_document_count: AtomicU64,

    // field_id_generator: AtomicU16,
    // filter_field_id_by_name: RwLock<HashMap<String, FieldId>>,
    // score_field_id_by_name: RwLock<HashMap<String, FieldId>>,
    embedding_sender: Sender<MultiEmbeddingCalculationRequest>,

    // doc_id_storage: RwLock<DocIdStorage>,
    indexes: RwLock<HashMap<IndexId, Index>>,

    temp_indexes: RwLock<HashMap<IndexId, Index>>,

    op_sender: OperationSender,
}

impl CollectionWriter {
    pub fn empty(
        id: CollectionId,
        description: Option<String>,
        write_api_key: ApiKey,
        default_locale: Locale,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        op_sender: OperationSender,
    ) -> Self {
        Self {
            id,
            description,
            write_api_key,
            default_locale,
            embedding_sender,
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
            default_locale,
            embedding_sender,
            indexes: RwLock::new(HashMap::new()),
            temp_indexes: RwLock::new(HashMap::new()),
            op_sender,
        })
    }

    pub async fn create_index(&self, request: CreateIndexRequest) -> Result<()> {
        let index_id = request.id;

        let index = Index::empty(
            self.id,
            request,
            self.embedding_sender.clone(),
            self.get_text_parser(self.default_locale),
            self.op_sender.clone(),
        )
        .await
        .context("Cannot create index")?;

        let mut indexes = self.indexes.write().await;
        indexes.insert(index_id, index);
        drop(indexes);

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

        Ok(())
    }

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

    /*
    pub async fn get_document_ids(&self) -> Vec<DocumentId> {
        self.doc_id_storage
            .read()
            .await
            .get_document_ids()
            .collect()
    }


    pub async fn set_embedding_hook(&self, hook_name: HookName) -> Result<()> {
        let field_id_by_name = self.score_field_id_by_name.read().await;
        let field_id = field_id_by_name
            .get(DEFAULT_EMBEDDING_FIELD_NAME)
            .cloned()
            .context("Field for embedding not found")?;
        drop(field_id_by_name);

        let mut w = self.score_fields.write().await;
        let field = match w.get_mut(&field_id) {
            None => bail!("Field for embedding not found"),
            Some(field) => field,
        };
        field.set_embedding_hook(hook_name);

        Ok(())
    }

    pub async fn contains(&self, doc_id_str: &str) -> bool {
        let doc_id_storage = self.doc_id_storage.write().await;
        doc_id_storage.contains(doc_id_str)
    }
    */

    pub async fn get_index<'s, 'index>(&'s self, id: IndexId) -> Option<IndexReadLock<'index>>
    where
        's: 'index,
    {
        let lock = self.indexes.read().await;
        IndexReadLock::try_new(lock, id)
    }

    /*
    fn value_to_typed_field(&self, value_type: ValueType) -> Option<TypedField> {
        match value_type {
            ValueType::Scalar(ScalarType::String) => {
                Some(TypedField::Text(self.default_language.into()))
            }
            ValueType::Scalar(ScalarType::Number) => Some(TypedField::Number),
            ValueType::Scalar(ScalarType::Boolean) => Some(TypedField::Bool),
            ValueType::Complex(ComplexType::Array(ScalarType::String)) => {
                Some(TypedField::ArrayText(self.default_language.into()))
            }
            ValueType::Complex(ComplexType::Array(ScalarType::Number)) => {
                Some(TypedField::ArrayNumber)
            }
            ValueType::Complex(ComplexType::Array(ScalarType::Boolean)) => {
                Some(TypedField::ArrayBoolean)
            }
            _ => None, // @todo: support other types
        }
    }
    */

    fn get_text_parser(&self, locale: Locale) -> Arc<TextParser> {
        // TextParser is expensive to create, so we cache it
        // TODO: add a cache
        let parser = TextParser::from_locale(locale);
        Arc::new(parser)
    }

    /*
    pub async fn delete_documents(
        &self,
        doc_ids: Vec<String>,
        sender: OperationSender,
    ) -> Result<()> {
        let doc_ids = self
            .doc_id_storage
            .write()
            .await
            .remove_document_id(doc_ids);
        info!(coll_id= ?self.id, ?doc_ids, "Deleting documents");

        let doc_ids_len = doc_ids.len();

        sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::DeleteDocuments { doc_ids },
            ))
            .await?;

        self.collection_document_count
            .fetch_sub(doc_ids_len as u64, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }
    */

    pub async fn remove_from_fs(self, path: PathBuf) {
        match std::fs::remove_dir_all(&path) {
            Ok(_) => {}
            Err(e) => {
                warn!(coll_id= ?self.id, "Cannot remove collection directory. Ignored: {:?}", e);
            }
        };
    }

    pub async fn commit(&self, path: PathBuf) -> Result<()> {
        info!(coll_id= ?self.id, "Committing collection");

        std::fs::create_dir_all(&path).context("Cannot create collection directory")?;

        let indexes_lock = self.indexes.read().await;
        let indexes = indexes_lock.keys().copied().collect::<Vec<_>>();
        let temp_indexes_lock = self.indexes.read().await;
        let temporary_indexes = temp_indexes_lock.keys().copied().collect::<Vec<_>>();

        let dump = CollectionDump::V1(CollectionDumpV1 {
            id: self.id,
            description: self.description.clone(),
            write_api_key: self.write_api_key.expose().to_string(),
            default_locale: self.default_locale,
            indexes,
            temporary_indexes,
        });

        BufferedFile::create_or_overwrite(path.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        Ok(())
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
    default_locale: Locale,
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
