use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use orama_js_pool::OutputChannel;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;

use crate::{lock::OramaSyncLock, types::CollectionId};

struct ChannelEntry {
    sender: Arc<broadcast::Sender<(OutputChannel, String)>>,
    last_used: Instant,
}

pub struct HookLogs {
    channels: Arc<OramaSyncLock<HashMap<CollectionId, ChannelEntry>>>,
}

impl Default for HookLogs {
    fn default() -> Self {
        Self::new()
    }
}

impl HookLogs {
    pub fn new() -> Self {
        let channels: Arc<OramaSyncLock<HashMap<CollectionId, ChannelEntry>>> =
            Arc::new(OramaSyncLock::new("hook_channels", Default::default()));
        let s = Self {
            channels: channels.clone(),
        };
        // Start the cleanup task to remove unused channels
        Self::spawn_cleanup_task(channels);
        s
    }

    pub fn get_sender(
        &self,
        collection_id: &CollectionId,
    ) -> Option<Arc<broadcast::Sender<(OutputChannel, String)>>> {
        let mut lock = self
            .channels
            .write("get_sender")
            .expect("This lock should never panic");
        if let Some(entry) = lock.get_mut(collection_id) {
            entry.last_used = Instant::now();
            Some(entry.sender.clone())
        } else {
            None
        }
    }

    pub fn get_or_create_receiver(
        &self,
        collection_id: CollectionId,
    ) -> broadcast::Receiver<(OutputChannel, String)> {
        let mut lock = self
            .channels
            .write("get_or_create_receiver")
            .expect("This lock should never panic");
        let answer_receiver = match lock.entry(collection_id) {
            Entry::Vacant(a) => {
                let (answer_sender, answer_receiver) = broadcast::channel(100);
                a.insert(ChannelEntry {
                    sender: Arc::new(answer_sender),
                    last_used: Instant::now(),
                });
                answer_receiver
            }
            Entry::Occupied(mut o) => {
                o.get_mut().last_used = Instant::now();
                o.get().sender.subscribe()
            }
        };
        drop(lock);

        answer_receiver
    }

    fn spawn_cleanup_task(
        channels: Arc<OramaSyncLock<HashMap<CollectionId, ChannelEntry>>>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let cleanup_interval = Duration::from_secs(60); // check every minute
            let max_idle = Duration::from_secs(600); // 10 minutes
            loop {
                tokio::time::sleep(cleanup_interval).await;
                let now = Instant::now();
                let mut lock = channels
                    .write("cleanup")
                    .expect("This lock should never panic");
                lock.retain(|_, entry| now.duration_since(entry.last_used) < max_idle);
                drop(lock);
            }
        })
    }
}
