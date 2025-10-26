pub mod s3;
pub mod storage;
use tokio::time::MissedTickBehavior;
use tracing::info;

use super::WriteSide;
use crate::types::{CollectionId, DeleteDocuments, DocumentList, IndexId, WriteApiKey};
use std::{sync::Arc, time};
use tracing::error;

pub enum Operation {
    Insert(DocumentList),
    Delete(DeleteDocuments),
}

pub struct SyncUpdate {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub operation: Operation,
}

async fn apply_sync_operations(
    write_side: &Arc<WriteSide>,
    mut sync_receiver: tokio::sync::mpsc::Receiver<SyncUpdate>,
) {
    while let Some(sync_msg) = sync_receiver.recv().await {
        match sync_msg.operation {
            Operation::Insert(docs) => {
                if let Err(e) = write_side
                    .insert_documents(
                        WriteApiKey::ApiKey(write_side.master_api_key.clone()),
                        sync_msg.collection_id,
                        sync_msg.index_id,
                        docs,
                    )
                    .await
                {
                    error!(
                        error = ?e,
                        "Failed to insert documents from datasource collection: {} index: {}",
                        sync_msg.collection_id,
                        sync_msg.index_id,
                    );
                }
            }
            Operation::Delete(keys_to_remove) => {
                if let Err(e) = write_side
                    .delete_documents(
                        WriteApiKey::ApiKey(write_side.master_api_key.clone()),
                        sync_msg.collection_id,
                        sync_msg.index_id,
                        keys_to_remove,
                    )
                    .await
                {
                    error!(
                        error = ?e,
                        "Failed to delete documents from datasource collection: {} index: {}",
                        sync_msg.collection_id,
                        sync_msg.index_id,
                    );
                }
            }
        }
    }
}

fn spawn_sync_tasks(
    write_side: &Arc<WriteSide>,
    sync_sender: tokio::sync::mpsc::Sender<SyncUpdate>,
    collection_id: CollectionId,
    index_id: IndexId,
    datasources: &Vec<storage::DatasourceEntry>,
) {
    let datasource_dir = write_side.datasource_storage.base_dir();
    for datasource in datasources {
        let task_sender_clone = sync_sender.clone();
        let datasource_dir_clone = datasource_dir.clone();

        match &datasource.datasource {
            storage::DatasourceKind::S3(s3_datasource) => {
                let s3_datasource = s3_datasource.clone();
                let datasource_id = datasource.id;
                // TODO: update resy to implement async correctly
                tokio::task::spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();

                    rt.block_on(async {
                        if let Err(e) = s3::sync_s3_datasource(
                            &datasource_dir_clone,
                            collection_id,
                            index_id,
                            datasource_id,
                            &s3_datasource,
                            task_sender_clone,
                        )
                        .await
                        {
                            error!(error = ?e, "Failed to sync S3 datasource");
                        }
                    })
                });
            }
        }
    }
}

pub fn start_datasource_loop(
    write_side: Arc<WriteSide>,
    interval: time::Duration,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
    mut sync_request_receiver: tokio::sync::mpsc::Receiver<(CollectionId, IndexId)>,
) {
    tokio::task::spawn(async move {
        let start = tokio::time::Instant::now() + interval;
        let mut interval = tokio::time::interval_at(start, interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        'outer: loop {
            let (sync_sender, sync_receiver) = tokio::sync::mpsc::channel(100);

            tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Stopping datasource loop");
                    break 'outer;
                }
                _ = interval.tick() => {
                    info!("Running periodic datasource sync");
                    let datasources = write_side.datasource_storage.get().await;
                    for (collection_id, indexes) in datasources.iter() {
                        for (index_id, dss) in indexes.iter() {
                            spawn_sync_tasks(&write_side, sync_sender.clone(), *collection_id, *index_id, dss);
                        }
                    }
                }
                Some((collection_id, index_id)) = sync_request_receiver.recv() => {
                    info!("Running on-demand datasource sync for collection {} index {}", collection_id, index_id);
                    let datasources = write_side.datasource_storage.get().await;
                    if let Some(indexes) = datasources.get(&collection_id) {
                        if let Some(dss) = indexes.get(&index_id) {
                            spawn_sync_tasks(&write_side, sync_sender.clone(), collection_id, index_id, dss);
                        }
                    }
                }
            };

            drop(sync_sender);
            apply_sync_operations(&write_side, sync_receiver).await;
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to writer side");
        }
    });
}

