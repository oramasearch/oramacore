pub mod s3;
pub mod storage;
use tokio::time::MissedTickBehavior;
use tracing::info;

use super::WriteSide;
use crate::types::{CollectionId, DeleteDocuments, DocumentList, IndexId, WriteApiKey};
use std::{path::PathBuf, sync::Arc, time};
use tracing::error;

enum Operation {
    Insert(DocumentList),
    Delete(DeleteDocuments),
}

pub struct SyncUpdate {
    collection_id: CollectionId,
    index_id: IndexId,
    operation: Operation,
}

pub fn start_datasource_loop(
    write_side: Arc<WriteSide>,
    datasource_dir: PathBuf,
    interval: time::Duration,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    tokio::task::spawn(async move {
        let start = tokio::time::Instant::now() + interval;
        let mut interval = tokio::time::interval_at(start, interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        'outer: loop {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Stopping datasource loop");
                    break 'outer;
                }
                _ = interval.tick() => {}
            };

            let (sync_sender, mut sync_receiver) = tokio::sync::mpsc::channel(100);

            let datasources = write_side.datasource_storage.get().await;
            for (collection_id, indexes) in datasources.iter() {
                for (index_id, datasources) in indexes.iter() {
                    for datasource in datasources {
                        let task_sender_clone = sync_sender.clone();
                        let datasource_dir_clone = datasource_dir.clone();

                        let collection_id = *collection_id;
                        let index_id = *index_id;
                        match &datasource.datasource {
                            storage::DatasourceKind::S3(s3_datasource) => {
                                let s3_datasource = s3_datasource.clone();
                                let datasource_id = datasource.id;
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
            }

            drop(sync_sender);

            while let Some(sync_msg) = sync_receiver.recv().await {
                match sync_msg.operation {
                    Operation::Insert(docs) => {
                        if let Err(e) = write_side
                            .insert_documents(
                                WriteApiKey::ApiKey(write_side.master_api_key),
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
                                WriteApiKey::ApiKey(write_side.master_api_key),
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

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to writer side");
        }
    });
}
