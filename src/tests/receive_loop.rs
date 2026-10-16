use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::Result;
use tokio::{
    sync::{broadcast, mpsc},
    time::{sleep, timeout},
};

use crate::collection_manager::sides::{
    read::{spawn_receive_loop, ExitHook, OperationSource, ReplicationConfig, ReplicationStatus},
    Offset, ReplicationRefusal, WriteOperation,
};
use crate::tests::utils::init_log;
use crate::types::CollectionId;

/// In-memory stand-in for the operation stream. Offsets `1..=head` are published.
#[derive(Default)]
struct FakeStream {
    head: AtomicU64,
    // While set, `recv` never returns, like a consumer whose connection died unnoticed.
    stuck: AtomicBool,
    // Whether recreating the consumer clears `stuck`.
    recreate_fixes: AtomicBool,
    panic_on_recv: AtomicBool,
    fail_apply: AtomicBool,
    fail_recreate: AtomicBool,
    head_unavailable: AtomicBool,
    // While set, `recv` reports the consumer as closed.
    closed: AtomicBool,
    // Recreating fails as if the stream no longer retained the requested offset.
    refuse_recreate: AtomicBool,
    recreated_from: Mutex<Vec<Offset>>,
}

struct FakeSource {
    stream: Arc<FakeStream>,
    cursor: u64,
}

#[async_trait::async_trait]
impl OperationSource for FakeSource {
    async fn recv(&mut self) -> Option<Result<(Offset, WriteOperation)>> {
        loop {
            if self.stream.panic_on_recv.load(Ordering::SeqCst) {
                panic!("receiver failure injected by the test");
            }
            if self.stream.closed.load(Ordering::SeqCst) {
                return None;
            }
            if !self.stream.stuck.load(Ordering::SeqCst)
                && self.cursor <= self.stream.head.load(Ordering::SeqCst)
            {
                let offset = Offset(self.cursor);
                self.cursor += 1;
                let op = WriteOperation::DeleteCollection(CollectionId::try_new("test").unwrap());
                return Some(Ok((offset, op)));
            }
            sleep(Duration::from_millis(5)).await;
        }
    }

    fn can_recreate(&self) -> bool {
        true
    }

    async fn recreate(&mut self, from: Offset) -> Result<()> {
        self.stream.recreated_from.lock().unwrap().push(from);
        anyhow::ensure!(
            !self.stream.fail_recreate.load(Ordering::SeqCst),
            "injected reconnect failure"
        );
        if self.stream.refuse_recreate.load(Ordering::SeqCst) {
            return Err(ReplicationRefusal::HistoryExpired {
                required: from.0,
                first: from.0 + 10,
            }
            .into());
        }
        if self.stream.recreate_fixes.load(Ordering::SeqCst) {
            self.stream.stuck.store(false, Ordering::SeqCst);
        }
        self.cursor = from.0;
        Ok(())
    }

    async fn stream_head(&mut self) -> Result<Option<Offset>> {
        anyhow::ensure!(
            !self.stream.head_unavailable.load(Ordering::SeqCst),
            "injected probe failure"
        );
        let head = self.stream.head.load(Ordering::SeqCst);
        Ok((head > 0).then_some(Offset(head)))
    }
}

/// A receive loop over a `FakeStream`, recording applied offsets and exit calls.
struct Harness {
    stream: Arc<FakeStream>,
    applied: Arc<Mutex<Vec<u64>>>,
    status: Arc<ReplicationStatus>,
    exits: mpsc::UnboundedReceiver<i32>,
    stop_sender: broadcast::Sender<()>,
    stop_done: mpsc::Receiver<()>,
}

impl Harness {
    fn start(stream: Arc<FakeStream>, stall_threshold: Duration) -> Self {
        let (exit_sender, exits) = mpsc::unbounded_channel();
        let config = ReplicationConfig {
            consumer_stall_threshold: stall_threshold,
            consumer_check_interval: Duration::from_millis(20),
            exit_hook: ExitHook::new(move |code| {
                let _ = exit_sender.send(code);
            }),
            ..Default::default()
        };
        let status = Arc::new(ReplicationStatus::new(
            Offset(0),
            Duration::from_secs(300),
            Duration::from_secs(3_000),
        ));
        let applied = Arc::new(Mutex::new(Vec::new()));
        let (stop_sender, stop_receiver) = broadcast::channel(1);
        let (stop_done_sender, stop_done) = mpsc::channel(1);

        let applied_by_loop = applied.clone();
        let apply_stream = stream.clone();
        spawn_receive_loop(
            FakeSource {
                stream: stream.clone(),
                cursor: 1,
            },
            move |(offset, _): (Offset, WriteOperation)| {
                let applied = applied_by_loop.clone();
                let stream = apply_stream.clone();
                async move {
                    anyhow::ensure!(
                        !stream.fail_apply.load(Ordering::SeqCst),
                        "injected apply failure"
                    );
                    applied.lock().unwrap().push(offset.0);
                    Ok(())
                }
            },
            status.clone(),
            config,
            stop_receiver,
            stop_done_sender,
        );

        Self {
            stream,
            applied,
            status,
            exits,
            stop_sender,
            stop_done,
        }
    }

    async fn wait_applied(&self, expected: &[u64]) {
        timeout(Duration::from_secs(10), async {
            while self.applied.lock().unwrap().as_slice() != expected {
                sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "Applied offsets are {:?}, expected {expected:?}",
                self.applied.lock().unwrap()
            )
        });
    }

    async fn wait_exit(&mut self) -> i32 {
        timeout(Duration::from_secs(10), self.exits.recv())
            .await
            .expect("The exit hook was not called")
            .expect("The exit hook was dropped")
    }

    async fn wait_stop_done(&mut self) {
        timeout(Duration::from_secs(10), self.stop_done.recv())
            .await
            .expect("The receive loop did not report that it finished")
            .expect("The stop-done channel was closed");
    }
}

/// A consumer that stops yielding without closing is recreated from the offset after the
/// last applied one, and application resumes from there.
#[tokio::test(flavor = "multi_thread")]
async fn test_receive_loop_recreates_silent_consumer() {
    init_log();

    let stream = Arc::new(FakeStream::default());
    stream.recreate_fixes.store(true, Ordering::SeqCst);
    let mut harness = Harness::start(stream, Duration::from_millis(200));

    harness.stream.head.store(3, Ordering::SeqCst);
    harness.wait_applied(&[1, 2, 3]).await;

    harness.stream.stuck.store(true, Ordering::SeqCst);
    harness.stream.head.store(5, Ordering::SeqCst);
    harness.wait_applied(&[1, 2, 3, 4, 5]).await;

    assert_eq!(
        *harness.stream.recreated_from.lock().unwrap(),
        vec![Offset(4)]
    );
    assert!(harness.exits.try_recv().is_err());

    harness.stop_sender.send(()).unwrap();
    harness.wait_stop_done().await;
    assert!(harness.exits.try_recv().is_err());
}

/// If recreating does not bring the consumer back, the process exits once the silence
/// lasts twice the stall threshold.
#[tokio::test(flavor = "multi_thread")]
async fn test_receive_loop_exits_when_stall_persists() {
    init_log();

    let stream = Arc::new(FakeStream::default());
    let mut harness = Harness::start(stream, Duration::from_millis(200));

    harness.stream.head.store(2, Ordering::SeqCst);
    harness.wait_applied(&[1, 2]).await;

    harness.stream.stuck.store(true, Ordering::SeqCst);
    harness.stream.head.store(4, Ordering::SeqCst);

    assert_eq!(harness.wait_exit().await, 1);
    assert_eq!(
        *harness.stream.recreated_from.lock().unwrap(),
        vec![Offset(3)]
    );
    assert_eq!(*harness.applied.lock().unwrap(), vec![1, 2]);
    harness.wait_stop_done().await;
}

/// A receive loop that panics leads to the process-exit path.
#[tokio::test(flavor = "multi_thread")]
async fn test_receive_loop_panic_exits_process() {
    init_log();

    let stream = Arc::new(FakeStream::default());
    let mut harness = Harness::start(stream, Duration::from_secs(120));

    harness.stream.head.store(1, Ordering::SeqCst);
    harness.wait_applied(&[1]).await;

    harness.stream.panic_on_recv.store(true, Ordering::SeqCst);

    assert_eq!(harness.wait_exit().await, 1);
    harness.wait_stop_done().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_receive_loop_exits_on_recreate_failure() {
    let stream = Arc::new(FakeStream::default());
    stream.stuck.store(true, Ordering::SeqCst);
    stream.fail_recreate.store(true, Ordering::SeqCst);
    stream.head.store(2, Ordering::SeqCst);
    let mut harness = Harness::start(stream, Duration::from_millis(200));
    assert_eq!(harness.wait_exit().await, 1);
    harness.wait_stop_done().await;
    assert_eq!(harness.status.applied_offset(), Offset(0));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_receive_loop_exits_when_head_cannot_be_probed() {
    let stream = Arc::new(FakeStream::default());
    stream.head_unavailable.store(true, Ordering::SeqCst);
    let mut harness = Harness::start(stream, Duration::from_millis(200));
    assert_eq!(harness.wait_exit().await, 1);
    harness.wait_stop_done().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_failed_apply_does_not_advance_applied_offset() {
    let stream = Arc::new(FakeStream::default());
    let mut harness = Harness::start(stream, Duration::from_secs(120));
    harness.stream.head.store(1, Ordering::SeqCst);
    harness.wait_applied(&[1]).await;
    harness.stream.fail_apply.store(true, Ordering::SeqCst);
    harness.stream.head.store(2, Ordering::SeqCst);
    assert_eq!(harness.wait_exit().await, 1);
    harness.wait_stop_done().await;
    assert_eq!(harness.status.applied_offset(), Offset(1));
    assert_eq!(harness.status.report().last_received_offset, 2);
}

/// A refusal is permanent: the loop exits on the first attempt instead of retrying the
/// reconnect until it times out.
#[tokio::test(flavor = "multi_thread")]
async fn test_receive_loop_exits_without_retrying_a_refused_reconnect() {
    init_log();

    let stream = Arc::new(FakeStream::default());
    let mut harness = Harness::start(stream, Duration::from_secs(120));

    harness.stream.head.store(2, Ordering::SeqCst);
    harness.wait_applied(&[1, 2]).await;

    harness.stream.refuse_recreate.store(true, Ordering::SeqCst);
    harness.stream.closed.store(true, Ordering::SeqCst);

    assert_eq!(harness.wait_exit().await, 1);
    assert_eq!(
        *harness.stream.recreated_from.lock().unwrap(),
        vec![Offset(3)]
    );
    harness.wait_stop_done().await;
}
