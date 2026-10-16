//! Replication health of the read side.
//!
//! - [`ReplicationStatus`] records receive, apply and commit progress and backs `/health/ready`.
//! - [`spawn_receive_loop`] runs the operation receive loop under a supervisor that exits the
//!   process if the loop ends, and a watchdog that recreates a consumer that went silent while
//!   the stream has newer operations.

use std::{
    any::Any,
    future::Future,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use async_trait::async_trait;
use backoff::{backoff::Backoff, ExponentialBackoff};
use chrono::{DateTime, Utc};
use duration_str::deserialize_duration;
use serde::{Deserialize, Serialize};
use tokio::{
    sync::{broadcast, mpsc},
    time::{sleep, timeout, Instant, MissedTickBehavior},
};
use tracing::{error, info, trace, warn};

use crate::collection_manager::sides::{
    Offset, OperationReceiver, ReplicationRefusal, WriteOperation,
};

/// Replication settings of the read side (`reader_side.replication`).
/// Every field has a default, so config files without this section keep working.
#[derive(Clone, Deserialize)]
pub struct ReplicationConfig {
    /// `/health/ready` reports not ready when the stream has an operation newer than the
    /// applied offset and nothing has been applied for this long.
    #[serde(
        deserialize_with = "deserialize_duration",
        default = "default_apply_stall_threshold"
    )]
    pub apply_stall_threshold: Duration,
    /// The consumer counts as stalled when the stream has operations newer than the applied
    /// offset and nothing has been received for this long. It is then recreated; if the stall
    /// lasts twice this long, the process exits.
    #[serde(
        deserialize_with = "deserialize_duration",
        default = "default_consumer_stall_threshold"
    )]
    pub consumer_stall_threshold: Duration,
    /// How often the watchdog checks the consumer.
    #[serde(
        deserialize_with = "deserialize_duration",
        default = "default_consumer_check_interval"
    )]
    pub consumer_check_interval: Duration,
    /// Called when the receive loop ends without a stop request.
    #[serde(skip)]
    pub exit_hook: ExitHook,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            apply_stall_threshold: default_apply_stall_threshold(),
            consumer_stall_threshold: default_consumer_stall_threshold(),
            consumer_check_interval: default_consumer_check_interval(),
            exit_hook: ExitHook::default(),
        }
    }
}

fn default_apply_stall_threshold() -> Duration {
    Duration::from_secs(300)
}

fn default_consumer_stall_threshold() -> Duration {
    Duration::from_secs(120)
}

fn default_consumer_check_interval() -> Duration {
    Duration::from_secs(30)
}

/// Ends the process with the given status. The default is `std::process::exit`, so a service
/// manager can restart the reader; tests inject a hook that records the call instead.
#[derive(Clone)]
pub struct ExitHook(Arc<dyn Fn(i32) + Send + Sync>);

impl ExitHook {
    pub fn new(exit: impl Fn(i32) + Send + Sync + 'static) -> Self {
        Self(Arc::new(exit))
    }
}

impl Default for ExitHook {
    fn default() -> Self {
        Self::new(|code| std::process::exit(code))
    }
}

/// Receive, apply and commit progress of the read side.
/// Shared by the receive loop, the commit path and `/health/ready`.
/// Timestamps are Unix milliseconds; 0 means "never".
pub struct ReplicationStatus {
    applied_offset: AtomicU64,
    // Stream head observed when the receiver was created. Readiness waits until the
    // reader has applied up to it, so a restarted reader never serves a stale index.
    startup_head: OnceLock<Offset>,
    last_applied_at: AtomicU64,
    received_offset: AtomicU64,
    last_received_at: AtomicU64,
    // Newest offset the watchdog saw on the stream.
    stream_head_offset: AtomicU64,
    stream_head_observed: AtomicBool,
    last_commit_at: AtomicU64,
    last_commit_failed: AtomicBool,
    // Baseline for the staleness checks before anything has been applied or committed.
    started_at: u64,
    apply_stall_threshold: Duration,
    max_commit_age: Duration,
}

/// Body of `/health/ready`. Timestamps are RFC 3339, `null` when the event has not happened
/// since the process started.
#[derive(Debug, Serialize)]
pub struct ReadinessReport {
    pub applied_offset: u64,
    pub last_applied_offset: u64,
    pub last_received_offset: u64,
    pub last_applied_at: Option<DateTime<Utc>>,
    pub last_received_at: Option<DateTime<Utc>>,
    pub last_commit_at: Option<DateTime<Utc>>,
    pub ready: bool,
}

impl ReplicationStatus {
    /// `applied_offset` is the offset the read side resumes from; commits are expected
    /// at least every `commit_interval`.
    pub fn new(
        applied_offset: Offset,
        apply_stall_threshold: Duration,
        commit_interval: Duration,
    ) -> Self {
        Self {
            applied_offset: AtomicU64::new(applied_offset.0),
            startup_head: OnceLock::new(),
            last_applied_at: AtomicU64::new(0),
            received_offset: AtomicU64::new(applied_offset.0),
            last_received_at: AtomicU64::new(0),
            stream_head_offset: AtomicU64::new(0),
            stream_head_observed: AtomicBool::new(false),
            last_commit_at: AtomicU64::new(0),
            last_commit_failed: AtomicBool::new(false),
            started_at: now_millis(),
            apply_stall_threshold,
            max_commit_age: commit_interval.saturating_mul(2),
        }
    }

    pub fn record_received(&self, offset: Offset) {
        self.received_offset.fetch_max(offset.0, Ordering::AcqRel);
        self.last_received_at.store(now_millis(), Ordering::Release);
    }

    /// Records a successfully applied operation.
    pub fn record_applied(&self, offset: Offset) {
        self.applied_offset.fetch_max(offset.0, Ordering::AcqRel);
        self.last_applied_at.store(now_millis(), Ordering::Release);
    }

    pub fn record_stream_head(&self, offset: Offset) {
        self.stream_head_offset
            .fetch_max(offset.0, Ordering::AcqRel);
        self.stream_head_observed.store(true, Ordering::Release);
    }

    /// Holds readiness back until `head` has been applied. Only the first call counts.
    pub fn require_initial_catchup(&self, head: Offset) {
        let _ = self.startup_head.set(head);
    }

    pub fn record_commit(&self, succeeded: bool) {
        if succeeded {
            self.last_commit_at.store(now_millis(), Ordering::Release);
        }
        self.last_commit_failed.store(!succeeded, Ordering::Release);
    }

    pub fn applied_offset(&self) -> Offset {
        Offset(self.applied_offset.load(Ordering::Acquire))
    }

    fn has_applied(&self) -> bool {
        self.applied_offset.load(Ordering::Acquire) != 0
            || self.last_applied_at.load(Ordering::Acquire) != 0
    }

    fn caught_up_to_startup_head(&self, applied_offset: u64) -> bool {
        self.startup_head
            .get()
            .is_none_or(|head| self.has_applied() && applied_offset >= head.0)
    }

    /// First offset not yet applied, where a created or recreated consumer starts:
    /// offset 0 without an apply timestamp means nothing has been applied yet.
    pub fn next_offset(&self) -> Offset {
        let applied = self.applied_offset();
        if applied.is_zero() && self.last_applied_at.load(Ordering::Acquire) == 0 {
            applied
        } else {
            applied.next()
        }
    }

    /// Ready once the startup head (if any) has been applied, unless the stream has
    /// operations newer than the applied offset and nothing has been applied for
    /// `apply_stall_threshold`, or the last commit failed or is older than twice the
    /// commit interval.
    pub fn report(&self) -> ReadinessReport {
        let now = now_millis();
        let applied_offset = self.applied_offset.load(Ordering::Acquire);
        let newest_known_offset = self
            .received_offset
            .load(Ordering::Acquire)
            .max(self.stream_head_offset.load(Ordering::Acquire));
        let last_applied_at = self.last_applied_at.load(Ordering::Acquire);
        let last_received_at = self.last_received_at.load(Ordering::Acquire);
        let last_commit_at = self.last_commit_at.load(Ordering::Acquire);

        // RabbitMQ's first message is offset zero; distinguish it from an empty stream.
        let has_backlog = newest_known_offset > applied_offset
            || (!self.has_applied()
                && (last_received_at != 0 || self.stream_head_observed.load(Ordering::Acquire)));
        let apply_stalled = has_backlog
            && elapsed_since(now, last_applied_at.max(self.started_at))
                > self.apply_stall_threshold;
        let commit_stale = self.last_commit_failed.load(Ordering::Acquire)
            || elapsed_since(now, last_commit_at.max(self.started_at)) > self.max_commit_age;

        ReadinessReport {
            applied_offset,
            last_applied_offset: applied_offset,
            last_received_offset: self.received_offset.load(Ordering::Acquire),
            last_applied_at: to_datetime(last_applied_at),
            last_received_at: to_datetime(last_received_at),
            last_commit_at: to_datetime(last_commit_at),
            ready: self.caught_up_to_startup_head(applied_offset)
                && !apply_stalled
                && !commit_stale,
        }
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn elapsed_since(now: u64, then: u64) -> Duration {
    Duration::from_millis(now.saturating_sub(then))
}

fn to_datetime(millis: u64) -> Option<DateTime<Utc>> {
    if millis == 0 {
        return None;
    }
    DateTime::from_timestamp_millis(millis as i64)
}

/// The stream of replicated operations, as the receive loop sees it.
#[async_trait]
pub trait OperationSource: Send + 'static {
    /// Next operation, or `None` once the source has closed.
    async fn recv(&mut self) -> Option<Result<(Offset, WriteOperation)>>;
    /// Whether a closed source can be recreated.
    fn can_recreate(&self) -> bool;
    /// Closes the current consumer and creates a new one that starts at `from`.
    async fn recreate(&mut self, from: Offset) -> Result<()>;
    /// Offset of the newest operation on the stream, when the source can tell.
    async fn stream_head(&mut self) -> Result<Option<Offset>>;
}

#[async_trait]
impl OperationSource for OperationReceiver {
    async fn recv(&mut self) -> Option<Result<(Offset, WriteOperation)>> {
        OperationReceiver::recv(self).await
    }

    fn can_recreate(&self) -> bool {
        self.should_reconnect()
    }

    async fn recreate(&mut self, from: Offset) -> Result<()> {
        OperationReceiver::recreate(self, from).await
    }

    async fn stream_head(&mut self) -> Result<Option<Offset>> {
        OperationReceiver::stream_head(self).await
    }
}

/// Why the receive loop returned.
enum LoopEnd {
    Stopped,
    Failed(String),
}

/// Starts the operation receive loop and its supervisor.
///
/// The loop takes operations from `source` and hands each one to `apply`. If it ends for any
/// reason other than a stop request (the source closed and cannot be recreated, the watchdog
/// gave up, or a panic), the supervisor logs the last applied offset and calls the exit hook
/// with status 1: a reader must not keep serving without replication. `stop_done_sender` gets
/// one message once the loop has finished, in every case.
pub fn spawn_receive_loop<S, A, F>(
    mut source: S,
    apply: A,
    status: Arc<ReplicationStatus>,
    config: ReplicationConfig,
    stop_receiver: broadcast::Receiver<()>,
    stop_done_sender: mpsc::Sender<()>,
) where
    S: OperationSource,
    A: Fn((Offset, WriteOperation)) -> F + Send + 'static,
    F: Future<Output = Result<()>> + Send,
{
    let exit_hook = config.exit_hook.clone();
    let loop_status = status.clone();
    let receive_loop = tokio::spawn(async move {
        run_receive_loop(&mut source, apply, &loop_status, &config, stop_receiver).await
    });

    tokio::spawn(async move {
        let failure = match receive_loop.await {
            Ok(LoopEnd::Stopped) => None,
            Ok(LoopEnd::Failed(reason)) => Some(reason),
            Err(e) if e.is_panic() => Some(format!("panicked: {}", panic_message(e.into_panic()))),
            Err(e) => Some(format!("task failed: {e}")),
        };

        if let Some(reason) = failure {
            let applied_offset = status.applied_offset().0;
            error!(
                applied_offset,
                reason, "Operation receive loop ended without a stop request. Exiting"
            );
            (exit_hook.0)(1);
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to read side");
        }
    });
}

async fn run_receive_loop<S, A, F>(
    source: &mut S,
    apply: A,
    status: &ReplicationStatus,
    config: &ReplicationConfig,
    mut stop_receiver: broadcast::Receiver<()>,
) -> LoopEnd
where
    S: OperationSource,
    A: Fn((Offset, WriteOperation)) -> F,
    F: Future<Output = Result<()>>,
{
    let mut watchdog = ConsumerWatchdog::new(config.consumer_stall_threshold);
    let mut checks = tokio::time::interval_at(
        Instant::now() + config.consumer_check_interval,
        config.consumer_check_interval,
    );
    checks.set_missed_tick_behavior(MissedTickBehavior::Delay);

    info!("Starting operation receiver");
    loop {
        let op = tokio::select! {
            _ = stop_receiver.recv() => {
                info!("Stopping operation receiver");
                return LoopEnd::Stopped;
            }
            _ = checks.tick() => {
                if let Err(reason) = watchdog.check(source, status).await {
                    return LoopEnd::Failed(reason);
                }
                continue;
            }
            op = source.recv() => op,
        };

        let op = match op {
            Some(Ok(op)) => op,
            Some(Err(e)) => {
                return LoopEnd::Failed(format!("cannot receive operation: {e:#}"));
            }
            None => {
                warn!("Operation receiver is closed");
                if !source.can_recreate() {
                    return LoopEnd::Failed("operation receiver closed".to_string());
                }
                if let Err(reason) = reconnect(source, status.next_offset()).await {
                    return LoopEnd::Failed(reason);
                }
                watchdog.mark_activity();
                continue;
            }
        };

        trace!(?op, "Received operation");
        let offset = op.0;
        status.record_received(offset);
        watchdog.mark_activity();

        if let Err(e) = apply(op).await {
            return LoopEnd::Failed(format!(
                "cannot apply operation at offset {}: {e:#}",
                offset.0
            ));
        }

        status.record_applied(offset);
        watchdog.mark_activity();
        info!("Operation applied {:?}", offset);
    }
}

/// Recreates a closed source from `from`, retrying with exponential backoff.
async fn reconnect<S: OperationSource>(source: &mut S, from: Offset) -> Result<(), String> {
    warn!(offset = from.0, "Reconnecting...");
    timeout(Duration::from_secs(120), async {
        let mut backoff = ExponentialBackoff::default();
        loop {
            match source.recreate(from).await {
                Ok(()) => {
                    info!("Reconnected to operation receiver");
                    return Ok(());
                }
                // Retention will not bring the history back: retrying cannot help.
                Err(e) if e.downcast_ref::<ReplicationRefusal>().is_some() => {
                    return Err(format!("{e:#}"));
                }
                Err(e) => match backoff.next_backoff() {
                    Some(wait) => {
                        warn!(error = ?e, "Cannot reconnect to operation receiver. Retrying");
                        sleep(wait).await;
                    }
                    None => return Err(format!("cannot reconnect to operation receiver: {e:#}")),
                },
            }
        }
    })
    .await
    .map_err(|_| "timed out reconnecting to operation receiver".to_string())?
}

/// Detects a consumer that stopped delivering while the stream has newer operations:
/// for example a connection the broker closed without the client noticing.
struct ConsumerWatchdog {
    stall_threshold: Duration,
    // Last time the consumer delivered something or an operation finished applying.
    // Time spent applying does not count as consumer silence.
    last_activity: Instant,
    // Whether the consumer has already been recreated during the current silence.
    recreated: bool,
}

impl ConsumerWatchdog {
    fn new(stall_threshold: Duration) -> Self {
        Self {
            stall_threshold,
            last_activity: Instant::now(),
            recreated: false,
        }
    }

    fn mark_activity(&mut self) {
        self.last_activity = Instant::now();
        self.recreated = false;
    }

    /// One watchdog round probes the stream head. After `stall_threshold` of silence,
    /// if the head is past the applied offset, it recreates the consumer from
    /// the next offset, once per silence. Returns an error, meaning the process must exit,
    /// when recreating fails or the silence lasts twice `stall_threshold`.
    async fn check<S: OperationSource>(
        &mut self,
        source: &mut S,
        status: &ReplicationStatus,
    ) -> Result<(), String> {
        let silence = self.last_activity.elapsed();
        // Probe every check interval, so readiness also learns about a silent consumer.
        // Bound the whole probe even for alternate OperationSource implementations.
        let head = match timeout(
            self.stall_threshold.min(Duration::from_secs(30)),
            source.stream_head(),
        )
        .await
        {
            Ok(Ok(Some(head))) => head,
            Ok(Ok(None)) => return Ok(()),
            probe_error => {
                if silence >= self.stall_threshold.saturating_mul(2) {
                    return Err(format!("stream head unavailable after {silence:?} of consumer silence: {probe_error:?}"));
                }
                warn!(error = ?probe_error, "Cannot query the stream head");
                return Ok(());
            }
        };
        status.record_stream_head(head);

        let applied = status.applied_offset();
        if head <= applied && status.has_applied() {
            // Nothing new on the stream: the consumer is idle, not stalled.
            return Ok(());
        }

        if silence < self.stall_threshold {
            return Ok(());
        }
        if silence >= self.stall_threshold.saturating_mul(2) {
            return Err(format!(
                "consumer received nothing for {silence:?} while the stream head is at offset {} and the applied offset is {}",
                head.0, applied.0
            ));
        }
        if self.recreated {
            return Ok(());
        }

        let from = status.next_offset();
        error!(
            silence = ?silence,
            stream_head = head.0,
            applied_offset = applied.0,
            "Operation consumer is stalled. Recreating it from offset {}",
            from.0
        );
        match timeout(self.stall_threshold, source.recreate(from)).await {
            Ok(Ok(())) => {
                info!(offset = from.0, "Operation consumer recreated");
                self.recreated = true;
                Ok(())
            }
            Ok(Err(e)) => Err(format!("cannot recreate the stalled consumer: {e:?}")),
            Err(_) => Err("timed out recreating the stalled consumer".to_string()),
        }
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message.to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}
