mod collection;
mod collections;
mod document_storage;
mod embedding;
mod fields;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use super::{
    generic_kv::{KVConfig, KV},
    hooks::{HookName, HooksRuntime, HooksRuntimeConfig},
    segments::{Segment, SegmentInterface},
    triggers::{get_trigger_key, Trigger, TriggerInterface},
    Offset, OperationSender, OperationSenderCreator, OutputSideChannelType,
};

use anyhow::{anyhow, bail, Context, Result};
use document_storage::DocumentStorage;
use duration_str::deserialize_duration;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::{
    sync::RwLock,
    time::{Instant, MissedTickBehavior},
};
use tokio_stream::StreamExt;
use tracing::{debug, info, instrument, trace, warn};

use collections::CollectionsWriter;
use embedding::{start_calculate_embedding_loop, EmbeddingCalculationRequest};

pub use fields::*;

use crate::{
    ai::AIService,
    collection_manager::{
        dto::{
            ApiKey, CollectionDTO, CreateCollection, CreateCollectionFrom, DeleteDocuments,
            InsertTriggerParams, ReindexConfig, SwapCollections,
        },
        sides::{CollectionWriteOperation, DocumentToInsert, WriteOperation},
    },
    file_utils::BufferedFile,
    metrics::{document_insertion::DOCUMENT_CALCULATION_TIME, CollectionLabels},
    nlp::NLPService,
    types::{CollectionId, Document, DocumentId, DocumentList},
};

#[derive(Debug, Deserialize, Clone)]
pub struct CollectionsWriterConfig {
    pub data_dir: PathBuf,
    #[serde(default = "embedding_queue_limit_default")]
    pub embedding_queue_limit: u32,
    #[serde(default = "embedding_model_default")]
    pub default_embedding_model: OramaModelSerializable,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
    #[serde(default = "javascript_queue_limit_default")]
    pub javascript_queue_limit: usize,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
}

#[derive(Deserialize, Clone)]
pub struct WriteSideConfig {
    pub master_api_key: ApiKey,
    pub hooks: HooksRuntimeConfig,
    pub output: OutputSideChannelType,
    pub config: CollectionsWriterConfig,
}

pub struct WriteSide {
    sender: OperationSender,
    collections: CollectionsWriter,
    document_count: AtomicU64,
    data_dir: PathBuf,
    hook_runtime: Arc<HooksRuntime>,
    operation_counter: RwLock<u64>,
    insert_batch_commit_size: u64,

    document_storage: DocumentStorage,
    segments: SegmentInterface,
    triggers: TriggerInterface,
    kv: Arc<KV>,
    master_api_key: ApiKey,
}

impl WriteSide {
    pub async fn try_load(
        sender_creator: OperationSenderCreator,
        config: WriteSideConfig,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
    ) -> Result<Arc<Self>> {
        let master_api_key = config.master_api_key;
        let collections_writer_config = config.config;
        let data_dir = collections_writer_config.data_dir.clone();

        let insert_batch_commit_size = collections_writer_config.insert_batch_commit_size;

        let commit_interval = collections_writer_config.commit_interval;
        let embedding_queue_limit = collections_writer_config.embedding_queue_limit;

        let (sx, rx) = tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(
            embedding_queue_limit as usize,
        );

        let document_count = AtomicU64::new(0);

        let write_side_info_path = data_dir.join("info.json");
        let r = BufferedFile::open(write_side_info_path)
            .and_then(|f| f.read_json_data::<WriteSideInfo>());
        let sender = if let Ok(info) = r {
            let WriteSideInfo::V1(info) = info;
            document_count.store(info.document_count, Ordering::Relaxed);
            sender_creator
                .create(info.offset)
                .await
                .context("Cannot create sender")?
        } else {
            sender_creator
                .create(Offset(0))
                .await
                .context("Cannot create sender")?
        };

        let kv = KV::try_load(KVConfig {
            data_dir: data_dir.join("kv"),
            sender: Some(sender.clone()),
        })
        .context("Cannot load KV")?;
        let kv = Arc::new(kv);
        let segments = SegmentInterface::new(kv.clone());
        let triggers = TriggerInterface::new(kv.clone());
        let hook =
            HooksRuntime::new(kv.clone(), collections_writer_config.javascript_queue_limit).await;
        let hook_runtime = Arc::new(hook);

        let collections_writer = CollectionsWriter::try_load(
            collections_writer_config,
            sx,
            hook_runtime.clone(),
            nlp_service.clone(),
        )
        .await
        .context("Cannot load collections")?;

        let document_storage = DocumentStorage::try_new(data_dir.join("documents"))
            .context("Cannot create document storage")?;

        let write_side = Self {
            document_count,
            collections: collections_writer,
            document_storage,
            data_dir,
            hook_runtime,
            insert_batch_commit_size,
            master_api_key,
            operation_counter: Default::default(),
            sender,
            segments,
            triggers,
            kv,
        };

        let write_side = Arc::new(write_side);

        start_commit_loop(write_side.clone(), commit_interval);
        start_calculate_embedding_loop(ai_service, rx, embedding_queue_limit);

        Ok(write_side)
    }

    #[instrument(skip(self))]
    pub async fn commit(&self) -> Result<()> {
        info!("Committing write side");

        self.collections.commit().await?;

        self.kv.commit().await?;

        let offset = self.sender.get_offset();
        // This load is not atomic with the commit.
        // This means, we save a document count possible higher.
        // Anyway it is not a problem, because the document count is only used for the document id generation
        // So, if something goes wrong, we save an higher number, and this is ok.
        let document_count = self.document_count.load(Ordering::Relaxed);
        let info = WriteSideInfo::V1(WriteSideInfoV1 {
            document_count,
            offset,
        });
        BufferedFile::create_or_overwrite(self.data_dir.join("info.json"))
            .context("Cannot create info file")?
            .write_json_data(&info)
            .context("Cannot write info file")?;

        Ok(())
    }

    pub async fn create_collection(
        &self,
        master_api_key: ApiKey,
        option: CreateCollection,
    ) -> Result<()> {
        self.check_master_api_key(master_api_key)?;

        self.collections
            .create_collection(option, self.sender.clone(), self.hook_runtime.clone())
            .await?;

        Ok(())
    }

    pub async fn insert_documents(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        document_list: DocumentList,
    ) -> Result<()> {
        let document_count = document_list.len();
        info!(?document_count, "Inserting batch of documents");

        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        let sender = self.sender.clone();
        for mut doc in document_list {
            let metric = DOCUMENT_CALCULATION_TIME.create(CollectionLabels {
                collection: collection_id.0.clone(),
            });

            debug!("Inserting doc");
            let doc_id = self.document_count.fetch_add(1, Ordering::Relaxed);

            let doc_id_value = doc.get("id");
            // Forces the id to be set, if not set
            if doc_id_value.is_none() {
                doc.inner.insert(
                    "id".to_string(),
                    serde_json::Value::String(cuid2::create_id()),
                );
            } else if let Some(doc_id_value) = doc_id_value {
                if !doc_id_value.is_string() {
                    // The search result contains the document id and it is defined as a string.
                    // So, if the original document id is not a string, we should overwrite it with a new one
                    // Anyway, this implies the loss of the original document id. For instance we could support number as well
                    // TODO: think better
                    warn!("Document id is not a string, overwriting it with new one");
                    doc.inner.insert(
                        "id".to_string(),
                        serde_json::Value::String(cuid2::create_id()),
                    );
                }
            }

            let doc_id = DocumentId(doc_id);

            self.document_storage
                .insert(doc_id, doc.clone())
                .await
                .context("Cannot inser document into document storage")?;

            // We send the document to index *before* indexing it, so we can
            // guarantee that the document is there during the search.
            // Otherwise, we could find the document without having it yet.
            self.sender
                .send(WriteOperation::Collection(
                    collection_id.clone(),
                    CollectionWriteOperation::InsertDocument {
                        doc_id,
                        doc: DocumentToInsert(doc.into_raw()?),
                    },
                ))
                .await
                .map_err(|e| anyhow!("Error sending document to index writer: {:?}", e))?;

            info!(?doc_id, "Inserting document");
            match collection
                .process_new_document(doc_id, doc, sender.clone(), self.hook_runtime.clone())
                .await
                .context("Cannot process document")
            {
                Ok(_) => {}
                Err(e) => {
                    // If the document cannot be processed, we should remove it from the document storage
                    // and from the read side
                    // NB: check if the error handling is correct

                    self.document_storage
                        .remove(doc_id)
                        .await
                        .context("Cannot remove document from document storage")?;

                    self.sender
                        .send(WriteOperation::Collection(
                            collection_id.clone(),
                            CollectionWriteOperation::DeleteDocuments {
                                doc_ids: vec![doc_id],
                            },
                        ))
                        .await
                        .context("Cannot send delete document operation")?;

                    return Err(e);
                }
            };
            info!("Document inserted");

            drop(metric);

            info!("Doc inserted");
        }

        let mut lock = self.operation_counter.write().await;
        *lock += document_count as u64;
        let should_commit = if *lock >= self.insert_batch_commit_size {
            *lock = 0;
            true
        } else {
            false
        };
        drop(lock);

        if should_commit {
            info!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size reached, committing");
            self.commit().await?;
        } else {
            trace!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size not reached, not committing");
        }

        info!("Batch of documents inserted");

        Ok(())
    }

    pub async fn delete_documents(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        document_ids_to_delete: DeleteDocuments,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .context("Collection not found")?;

        collection.check_write_api_key(write_api_key)?;

        collection
            .delete_documents(document_ids_to_delete, self.sender.clone())
            .await?;

        Ok(())
    }

    pub async fn delete_collection(
        &self,
        master_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<()> {
        self.check_master_api_key(master_api_key).unwrap();

        let deleted = self
            .collections
            .delete_collection(collection_id.clone())
            .await;
        if deleted {
            self.commit()
                .await
                .context("Cannot commit collections after collection deletion")?;

            self.sender
                .send(WriteOperation::DeleteCollection(collection_id))
                .await
                .context("Cannot send delete collection operation")?;
        }

        Ok(())
    }

    pub async fn insert_javascript_hook(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        self.hook_runtime
            .insert_hook(collection_id.clone(), name.clone(), code)
            .await
            .context("Cannot insert hook")?;

        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        collection
            .set_embedding_hook(name)
            .await
            .context("Cannot set embedding hook")?;

        Ok(())
    }

    pub async fn list_collections(&self, master_api_key: ApiKey) -> Result<Vec<CollectionDTO>> {
        self.check_master_api_key(master_api_key)?;

        Ok(self.collections.list().await)
    }

    pub async fn get_collection_dto(
        &self,
        master_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Option<CollectionDTO>> {
        self.check_master_api_key(master_api_key)?;
        let collection = match self.collections.get_collection(collection_id).await {
            Some(collection) => collection,
            None => return Ok(None),
        };
        Ok(Some(collection.as_dto().await))
    }

    pub async fn create_collection_from(
        &self,
        write_api_key: ApiKey,
        request: CreateCollectionFrom,
    ) -> Result<CollectionId> {
        info!("create temporary collection");
        self.check_write_api_key(request.from.clone(), write_api_key)
            .await
            .context("Check write api key fails")?;

        let collection_id_tmp = cuid2::create_id();
        let collection_id_tmp = CollectionId(collection_id_tmp);

        let mut option = self
            .collections
            .collection_options
            .get(&request.from)
            .ok_or_else(|| anyhow!("Collection options not found"))?
            .clone();
        option.language = request.language.or(option.language);
        option.embeddings = request.embeddings.or(option.embeddings);
        option.id = collection_id_tmp.clone();

        self.collections
            .create_collection(option, self.sender.clone(), self.hook_runtime.clone())
            .await?;

        Ok(collection_id_tmp)
    }

    pub async fn swap_collections(
        &self,
        write_api_key: ApiKey,
        request: SwapCollections,
    ) -> Result<()> {
        info!("Replacing collection");
        self.check_write_api_key(request.from.clone(), write_api_key.clone())
            .await
            .context("Check write api key fails")?;
        self.check_write_api_key(request.to.clone(), write_api_key)
            .await
            .context("Check write api key fails")?;

        self.collections
            .replace(request.from.clone(), request.from.clone())
            .await
            .context("Cannot replace collection")?;
        info!("Replaced");

        info!("Substitute collection");
        self.sender
            .send(WriteOperation::SubstituteCollection {
                subject_collection_id: request.from,
                target_collection_id: request.to,
            })
            .await?;

        Ok(())
    }

    pub async fn reindex(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        reindex_config: ReindexConfig,
    ) -> Result<()> {
        info!("Reindexing collection {:?}", collection_id);
        let collection_id_tmp = self
            .create_collection_from(
                write_api_key.clone(),
                CreateCollectionFrom {
                    from: collection_id.clone(),
                    embeddings: reindex_config.embeddings.clone(),
                    language: reindex_config.language,
                },
            )
            .await
            .context("Cannot create temporary collection")?;

        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        let document_ids = collection.get_document_ids().await;
        drop(collection);

        let collection = self
            .collections
            .get_collection(collection_id_tmp.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        let mut stream = self.document_storage.stream_documents(document_ids).await;
        while let Some((doc_id, doc)) = stream.next().await {
            debug!(?doc_id, "Reindexing document");

            let inner = doc.inner;
            let inner: Map<String, Value> =
                serde_json::from_str(inner.get()).context("Cannot deserialize document")?;
            let doc = Document { inner };
            match collection
                .process_new_document(doc_id, doc, self.sender.clone(), self.hook_runtime.clone())
                .await
                .context("Cannot process document")
            {
                Ok(_) => {}
                Err(e) => {
                    // If the document cannot be processed, we should remove it from the document storage
                    // and from the read side
                    // NB: check if the error handling is correct

                    self.sender
                        .send(WriteOperation::Collection(
                            collection_id.clone(),
                            CollectionWriteOperation::DeleteDocuments {
                                doc_ids: vec![doc_id],
                            },
                        ))
                        .await
                        .context("Cannot send delete document operation")?;

                    return Err(e);
                }
            };
            info!("Document reindexed");
        }
        drop(collection);

        self.swap_collections(
            write_api_key,
            SwapCollections {
                from: collection_id_tmp.clone(),
                to: collection_id.clone(),
            },
        )
        .await
        .context("Cannot swap collections")?;

        Ok(())
    }

    pub async fn get_javascript_hook(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        name: HookName,
    ) -> Result<Option<String>> {
        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        Ok(self
            .hook_runtime
            .get_hook(collection_id, name)
            .await
            .map(|hook| hook.code))
    }

    pub async fn delete_javascript_hook(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        _name: HookName,
    ) -> Result<Option<String>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        bail!("Not implemented yet.") // @todo: implement delete hook in HooksRuntime and CollectionsWriter
    }

    pub async fn list_javascript_hooks(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<HashMap<HookName, String>> {
        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        Ok(self
            .hook_runtime
            .list_hooks(collection_id)
            .await
            .context("Cannot list hooks")?
            .into_iter()
            .map(|(name, hook)| (name, hook.code))
            .collect())
    }

    async fn check_write_api_key(
        &self,
        collection_id: CollectionId,
        write_api_key: ApiKey,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        Ok(())
    }

    fn check_master_api_key(&self, master_api_key: ApiKey) -> Result<()> {
        if self.master_api_key != master_api_key {
            return Err(anyhow::anyhow!("Invalid master api key"));
        }

        Ok(())
    }

    pub async fn insert_segment(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        segment: Segment,
    ) -> Result<()> {
        self.check_write_api_key(collection_id.clone(), write_api_key)
            .await?;

        self.segments
            .insert(collection_id.clone(), segment.clone())
            .await
            .context("Cannot insert segment")?;

        Ok(())
    }

    pub async fn delete_segment(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>> {
        self.check_write_api_key(collection_id.clone(), write_api_key)
            .await?;

        self.segments
            .delete(collection_id.clone(), segment_id.clone())
            .await
            .context("Cannot delete segment")
    }

    pub async fn update_segment(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        segment: Segment,
    ) -> Result<()> {
        self.check_write_api_key(collection_id.clone(), write_api_key)
            .await?;

        self.segments
            .delete(collection_id.clone(), segment.id.clone())
            .await
            .context("Cannot delete segment")?;
        self.segments
            .insert(collection_id, segment)
            .await
            .context("Cannot insert segment")?;

        Ok(())
    }

    pub async fn insert_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger: InsertTriggerParams,
        trigger_id: Option<String>,
    ) -> Result<Trigger> {
        self.check_write_api_key(collection_id.clone(), write_api_key)
            .await?;

        let final_trigger_id = match trigger_id {
            Some(id) => id,
            None => {
                let cuid = cuid2::create_id();
                get_trigger_key(collection_id.clone(), cuid, trigger.segment_id.clone())
            }
        };

        let trigger = Trigger {
            id: final_trigger_id,
            name: trigger.name,
            description: trigger.description,
            response: trigger.response,
            segment_id: trigger.segment_id,
        };

        self.triggers
            .insert(trigger.clone())
            .await
            .context("Cannot insert trigger")?;

        Ok(trigger)
    }

    pub async fn get_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Trigger> {
        self.check_write_api_key(collection_id.clone(), write_api_key)
            .await?;

        let trigger = self
            .triggers
            .get(trigger_id)
            .await
            .context("Cannot insert trigger")?;
        let trigger = match trigger {
            Some(trigger) => trigger,
            None => bail!("Trigger not found"),
        };

        Ok(trigger)
    }

    pub async fn delete_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        self.check_write_api_key(collection_id.clone(), write_api_key)
            .await?;

        self.triggers
            .delete(collection_id, trigger_id)
            .await
            .context("Cannot delete trigger")
    }

    pub async fn update_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger: Trigger,
    ) -> Result<()> {
        let updated_trigger = InsertTriggerParams {
            name: trigger.name.clone(),
            description: trigger.description.clone(),
            response: trigger.response.clone(),
            segment_id: trigger.segment_id.clone(),
        };

        self.insert_trigger(
            write_api_key.clone(),
            collection_id.clone(),
            updated_trigger,
            Some(trigger.id),
        )
        .await
        .context("Cannot insert updated trigger")?;

        Ok(())
    }
}

fn start_commit_loop(write_side: Arc<WriteSide>, insert_batch_commit_size: Duration) {
    tokio::task::spawn(async move {
        let start = Instant::now() + insert_batch_commit_size;
        let mut interval = tokio::time::interval_at(start, insert_batch_commit_size);

        // If for some reason we miss a tick, we skip it.
        // In fact, the commit is blocked only by `update` method.
        // If the collection is under heavy load,
        // the commit will be run due to the `insert_batch_commit_size` config.
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            info!(
                "{:?} time reached. Committing write side",
                insert_batch_commit_size.clone()
            );
            if let Err(e) = write_side.commit().await {
                tracing::error!(?e, "Cannot commit write side");
            }
        }
    });
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "version")]
enum WriteSideInfo {
    #[serde(rename = "1")]
    V1(WriteSideInfoV1),
}

#[derive(Serialize, Deserialize)]
struct WriteSideInfoV1 {
    document_count: u64,
    offset: Offset,
}

fn embedding_queue_limit_default() -> u32 {
    50
}
fn javascript_queue_limit_default() -> usize {
    50
}

fn embedding_model_default() -> OramaModelSerializable {
    OramaModelSerializable(crate::ai::OramaModel::BgeSmall)
}

fn default_insert_batch_commit_size() -> u64 {
    1_000
}
