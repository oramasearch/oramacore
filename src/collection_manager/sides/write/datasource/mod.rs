pub mod s3;
pub mod storage;
use tokio::time::MissedTickBehavior;
use tracing::info;

use super::WriteSide;
use std::{path::PathBuf, sync::Arc, time};
use tracing::error;

pub fn start_datasource_loop(
    write_side: Arc<WriteSide>,
    datasource_dir: PathBuf,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    std::thread::spawn(async move || {
        let period = time::Duration::new(5, 0);
        let start = tokio::time::Instant::now() + period;
        let mut interval = tokio::time::interval_at(start, period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        'outer: loop {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Stopping datasource loop");
                    break 'outer;
                }
                _ = interval.tick() => {}
            };

            // TODO: looks like rusqlite doesn't support sync trait, so for now I've found this
            // workaround
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let s3_datasources = write_side.datasource_storage.get().await;

                for (collection_id, indexes) in s3_datasources.iter() {
                    for (index_id, datasources) in indexes.iter() {
                        for datasource in datasources {
                            match &datasource.datasource {
                                storage::DatasourceKind::S3(s3_datasource) => {
                                    if let Err(e) = s3::sync_s3_datasource(
                                        write_side.clone(),
                                        &datasource_dir,
                                        *collection_id,
                                        *index_id,
                                        datasource.id,
                                        s3_datasource,
                                    )
                                    .await
                                    {
                                        error!(error = ?e, "Failed to sync S3 datasource");
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to writer side");
        }
    });
}
