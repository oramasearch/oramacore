use std::{io::Write, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use chrono::Utc;
use fs::create_if_not_exists;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tokio_util::io::ReaderStream;
use tracing::{error, info};

use crate::types::{ApiKey, CollectionId, SearchParams, SearchResult};

#[derive(Deserialize, Clone)]
pub struct AnalyticConfig {
    pub api_key: ApiKey,
}

#[derive(Serialize, Clone, Copy)]
pub enum AnalyticSearchEventInvocationType {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "action")]
    Action,
    #[serde(rename = "answer")]
    Answer,
    #[serde(rename = "nlp_search")]
    NLPSearch,
    #[serde(rename = "party_planner")]
    PartyPlanner,
}

#[cfg_attr(test, derive(Clone))]
pub struct Dur {
    duration: Duration,
}
impl Serialize for Dur {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u128(self.duration.as_millis())
    }
}
impl From<Duration> for Dur {
    fn from(duration: Duration) -> Self {
        Self { duration }
    }
}

#[derive(Serialize)]
#[cfg_attr(test, derive(Clone))]
pub struct AnalyticSearchEvent {
    pub at: i64,
    #[serde(rename = "cid")]
    pub collection_id: CollectionId,
    #[serde(rename = "dms")]
    pub search_time: Dur,
    #[serde(rename = "sp")]
    pub search_params: SearchParams,
    #[serde(rename = "uid", skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(rename = "rc")]
    pub results_count: usize,
    #[serde(rename = "r", skip_serializing_if = "Option::is_none")]
    pub full_results_json: Option<SearchResult>,
    #[serde(rename = "from")]
    pub invocation_type: AnalyticSearchEventInvocationType,
}

impl From<AnalyticSearchEvent> for AnalyticEvent {
    fn from(val: AnalyticSearchEvent) -> Self {
        AnalyticEvent::Search(val)
    }
}

#[derive(Serialize)]
#[cfg_attr(test, derive(Clone))]
pub struct AnalyticAnswerEvent {
    pub at: i64,
    #[serde(rename = "cid")]
    pub collection_id: CollectionId,
    pub full_conversation: (),
    pub question: String,
    pub context: (),
    pub response: String,

    #[serde(rename = "uid", skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl From<AnalyticAnswerEvent> for AnalyticEvent {
    fn from(val: AnalyticAnswerEvent) -> Self {
        AnalyticEvent::Answer(val)
    }
}

#[derive(Serialize)]
#[serde(tag = "et")]
#[cfg_attr(test, derive(Clone))]
pub enum AnalyticEvent {
    #[serde(rename = "s")]
    Search(AnalyticSearchEvent),
    #[serde(rename = "a")]
    Answer(AnalyticAnswerEvent),
}

pub struct AnalyticsStorage {
    api_key: ApiKey,
    sender: tokio::sync::mpsc::Sender<InternalEvent>,
}

impl AnalyticsStorage {
    pub fn try_new(data_dir: PathBuf, config: AnalyticConfig) -> Result<Self> {
        create_if_not_exists(&data_dir)?;

        let (sender, receiver) = tokio::sync::mpsc::channel::<InternalEvent>(100);

        tokio::task::spawn(store_event_loop(data_dir, receiver));

        Ok(Self {
            sender,
            api_key: config.api_key,
        })
    }

    pub fn add_event<EV: Into<AnalyticEvent>>(&self, event: EV) -> Result<()> {
        let internal_event = InternalEvent::NewEvent(event.into());
        if let Err(e) = self.sender.try_send(internal_event) {
            error!(error = ?e, "Failed to send analytic event");
            return Err(anyhow::anyhow!("Failed to send analytic event"));
        }
        Ok(())
    }

    pub async fn get_and_erase(&self, api_key: ApiKey) -> Result<AnalyticLogStream> {
        if self.api_key != api_key {
            return Err(anyhow::anyhow!("Invalid analytics API key"));
        }

        let (sender, receiver) = tokio::sync::oneshot::channel::<PathBuf>();
        let internal_event = InternalEvent::Rotate(sender);
        if let Err(e) = self.sender.try_send(internal_event) {
            error!(error = ?e, "Failed to send rotate signal");
            return Err(anyhow::anyhow!("Failed to send rotate signal"));
        }

        let file_name = receiver
            .await
            .map_err(|_| anyhow::anyhow!("Failed to receive rotate signal"))?;
        let file = tokio::fs::File::open(&file_name).await?;

        let stream: ReaderStream<tokio::fs::File> = ReaderStream::new(file);
        let stream = AnalyticLogStream {
            file_path: file_name.clone(),
            stream,
            already_deleted: false,
        };

        Ok(stream)
    }
}

pub struct AnalyticLogStream {
    file_path: PathBuf,
    stream: ReaderStream<tokio::fs::File>,
    already_deleted: bool,
}

impl futures::Stream for AnalyticLogStream {
    type Item = Result<tokio_util::bytes::Bytes, anyhow::Error>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let ret = self.as_mut().stream.poll_next_unpin(cx);

        if matches!(ret, std::task::Poll::Ready(None)) && !self.already_deleted {
            self.already_deleted = true;
            if let Err(e) = std::fs::remove_file(&self.file_path) {
                error!(error = ?e, "Failed to delete analytics file");
            } else {
                info!("Analytics file {} deleted", self.file_path.display());
            }
        }

        ret.map(|opt| opt.map(|bytes| bytes.map_err(anyhow::Error::from)))
    }
}

async fn store_event_loop(
    base_dir: PathBuf,
    mut receiver: tokio::sync::mpsc::Receiver<InternalEvent>,
) -> Result<()> {
    let now = Utc::now().timestamp();
    let mut file_name = base_dir.join(format!("analytics_{now}.log"));
    let mut file = std::fs::File::options()
        .create(true)
        .write(true)
        .read(true)
        .truncate(false)
        .open(&file_name)
        .with_context(|| format!("Cannot open file at {file_name:?}"))?;

    // 100 is an arbitrary limit for the number of events to process at once
    let limit = 100;

    let mut buffer = Vec::with_capacity(limit);

    loop {
        let rec = receiver.recv_many(&mut buffer, limit).await;
        if rec == 0 {
            info!("No more events to process, exiting event loop");
            // Acording to the documentation, this means the channel is closed
            // because the limit is greater than 0 (it is hard-coded)
            break;
        }

        info!(count = rec, "Processing {} analytic events", rec);
        for event in buffer.drain(..) {
            match event {
                InternalEvent::NewEvent(ev) => {
                    write_to_file(&mut file, &ev)
                        .with_context(|| format!("Cannot write event to file at {file_name:?}"))?;
                }
                InternalEvent::Rotate(sender) => {
                    let previous_file_name = file_name;
                    file_name = loop {
                        let now = Utc::now().timestamp();
                        let new_file_name = base_dir.join(format!("analytics_{now}.log"));
                        if new_file_name != previous_file_name {
                            break new_file_name;
                        }
                        sleep(Duration::from_millis(100)).await; // Force a new timestamp
                    };

                    file = std::fs::File::options()
                        .create(true)
                        .write(true)
                        .read(true)
                        .truncate(false)
                        .open(&file_name)
                        .with_context(|| format!("Cannot open file at {file_name:?}"))?;

                    info!("Analytics file rotated: old_file {previous_file_name:?} new_file {file_name:?}");

                    if let Err(e) = sender.send(previous_file_name) {
                        error!(error = ?e, "Failed to send rotate signal");
                    }
                }
            }

            if let Err(e) = file.flush() {
                error!(error = ?e, "Failed to flush file");
            }
            if let Err(e) = file.sync_all() {
                error!(error = ?e, "Failed to sync file");
            }
        }
    }

    Ok(())
}

fn write_to_file<T: Serialize>(file: &mut std::fs::File, data: &T) -> Result<()> {
    let data = serde_json::to_string(data).context("Cannot serialize data to JSON")?;
    file.write_all(data.as_bytes())
        .context("Cannot write data to file")?;
    if let Err(e) = file.write(b"\n") {
        error!(error = ?e, "Failed to write newline to file");
    }
    Ok(())
}

enum InternalEvent {
    NewEvent(AnalyticEvent),
    Rotate(tokio::sync::oneshot::Sender<PathBuf>),
}

#[cfg(test)]
mod tests {
    use futures::FutureExt;
    use futures::TryStreamExt;
    use serde_json::json;

    use crate::tests::utils::{generate_new_path, init_log, wait_for};

    use super::*;

    #[tokio::test]
    async fn test_analytics_storage_lifecycle() {
        init_log();

        let analytics_api_key = ApiKey::try_new("test_api_key").unwrap();

        let data_dir = generate_new_path();
        let storage = AnalyticsStorage::try_new(
            data_dir.clone(),
            AnalyticConfig {
                api_key: analytics_api_key,
            },
        )
        .unwrap();

        let event = AnalyticEvent::Search(AnalyticSearchEvent {
            at: Utc::now().timestamp(),
            collection_id: CollectionId::try_new("test_collection").unwrap(),
            search_time: Duration::from_secs(1).into(),
            search_params: json!({
                "term": "test",
            })
            .try_into()
            .unwrap(),
            user_id: Some("user123".to_string()),
            results_count: 10,
            full_results_json: None,
            invocation_type: AnalyticSearchEventInvocationType::Direct,
        });
        storage.add_event(event.clone()).unwrap();
        storage.add_event(event.clone()).unwrap();

        let old_file: PathBuf = wait_for(&data_dir, |data_dir| {
            let data_dir = data_dir.clone();
            async move {
                let entry: Vec<_> = std::fs::read_dir(data_dir).unwrap().collect();
                if entry.len() == 1 {
                    // Only the new file should remain
                    return Ok(entry[0].as_ref().unwrap().path());
                }
                Err(anyhow::anyhow!("No file found"))
            }
            .boxed()
        })
        .await
        .unwrap();
        let file_content = std::fs::read_to_string(&old_file).unwrap();
        assert_eq!(file_content.lines().count(), 2);

        let mut stream = storage.get_and_erase(analytics_api_key).await.unwrap();

        storage.add_event(event.clone()).unwrap();

        wait_for(&data_dir, |data_dir| {
            let data_dir = data_dir.clone();
            async move {
                let entry: Vec<_> = std::fs::read_dir(data_dir).unwrap().collect();
                if entry.len() == 2 {
                    // Old file and new file should exist
                    return Ok(entry[0].as_ref().unwrap().path());
                }
                Err(anyhow::anyhow!("No file found"))
            }
            .boxed()
        })
        .await
        .unwrap();

        let mut stream_content = String::new();
        while let Some(Ok(bytes)) = stream.next().await {
            // Process the bytes as needed
            stream_content.push_str(&String::from_utf8_lossy(&bytes));
        }
        assert_eq!(
            stream_content.split('\n').count(),
            3,
            "There should be two lines in the analytics log + 1 for the newline"
        );

        let new_file: PathBuf = wait_for(&data_dir, |data_dir| {
            let data_dir = data_dir.clone();
            async move {
                let entry: Vec<_> = std::fs::read_dir(data_dir).unwrap().collect();
                if entry.len() == 1 {
                    // Only the new file should remain
                    return Ok(entry[0].as_ref().unwrap().path());
                }
                Err(anyhow::anyhow!("No file found"))
            }
            .boxed()
        })
        .await
        .unwrap();

        assert_ne!(
            old_file, new_file,
            "Old file should be different from new file"
        );
    }

    #[tokio::test]
    async fn test_analytics_storage_inner_write() {
        init_log();

        let data_dir = generate_new_path();
        create_if_not_exists(&data_dir).unwrap();
        let file_name = data_dir.join("analytics.log");

        let mut file = std::fs::File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&file_name)
            .unwrap();

        write_to_file(&mut file, &"test data").unwrap();
        write_to_file(&mut file, &"test data2").unwrap();

        let content = std::fs::read_to_string(&file_name)
            .map_err(|e| anyhow::anyhow!("Failed to read file {}: {}", file_name.display(), e))
            .unwrap();

        assert!(content.contains("test data"));
        assert!(content.contains("test data2"));
        assert!(content.lines().count() == 2);
    }
}
