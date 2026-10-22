use anyhow::{bail, Context, Result};
use futures::{Stream, StreamExt};
use rabbitmq_stream_client::error::{ClientError, ConsumerCreateError};
use rabbitmq_stream_client::{
    error::{ProducerCreateError, ProducerPublishError, StreamCreateError},
    types::{ByteCapacity, Message, OffsetSpecification, ResponseCode},
    ClientOptions, Consumer, Environment, NoDedup, OnClosed, Producer,
};
use serde::Deserialize;
use std::time::Duration;
use std::{
    pin::Pin,
    sync::{atomic::AtomicBool, Arc},
    task::Poll,
};
use tokio::sync::Notify;
use tokio::time::{sleep, timeout, timeout_at, Instant};
use tracing::{debug, error, info, trace, warn};

use crate::lock::OramaAsyncLock;
use crate::types::CollectionId;

use super::{Offset, WriteOperation};

struct MyHAProducerInner {
    environment: Box<Environment>,
    stream: String,
    producer: OramaAsyncLock<Producer<NoDedup>>,
    notify: Notify,
    is_opened: AtomicBool,
}

#[derive(Clone)]
struct MyHAProducer(Arc<MyHAProducerInner>);

#[async_trait::async_trait]
impl OnClosed for MyHAProducer {
    async fn on_closed(&self, unconfirmed: Vec<Message>) {
        use backoff::future::retry;
        use backoff::ExponentialBackoff;

        info!("Producer is closed. Creating new one");

        self.0
            .is_opened
            .store(false, std::sync::atomic::Ordering::SeqCst);

        let mut producer = self.0.producer.write("recreate").await;

        let inner = self.0.clone();
        let new_producer: std::result::Result<Producer<NoDedup>, ProducerCreateError> =
            retry(ExponentialBackoff::default(), move || {
                let inner = inner.clone();
                async move {
                    inner
                        .environment
                        .producer()
                        .build(&self.0.stream)
                        .await
                        .map_err(backoff::Error::transient)
                }
            })
            .await;

        let new_producer = match new_producer {
            Ok(p) => p,
            Err(e) => {
                error!(error =?e,"Failed to create new producer: {:?}", e);
                return;
            }
        };

        new_producer.set_on_closed(Box::new(self.clone())).await;

        if !unconfirmed.is_empty() {
            info!("Resending {} unconfirmed messages.", unconfirmed.len());
            if let Err(e) = producer.batch_send_with_confirm(unconfirmed).await {
                error!(error =?e, "Error resending unconfirmed messages.");
            }
        }

        **producer = new_producer;
        drop(producer);

        self.0
            .is_opened
            .store(true, std::sync::atomic::Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }
}

impl MyHAProducer {
    async fn new(
        environment: Box<Environment>,
        producer_name: &str,
        stream_name: &str,
    ) -> Result<Self> {
        let producer = environment
            .producer()
            .client_provided_name(producer_name)
            .build(stream_name)
            .await
            .context("Failed to create rabbit producer")?;

        let inner = MyHAProducerInner {
            environment,
            stream: stream_name.to_string(),
            producer: OramaAsyncLock::new("orama_rw_lock", producer),
            notify: Notify::new(),
            is_opened: AtomicBool::new(true),
        };
        let s = Self(Arc::new(inner));

        let p = s.0.producer.write("set_on_closed").await;

        p.set_on_closed(Box::new(s.clone())).await;
        drop(p);

        Ok(s)
    }

    async fn batch_send(&self, messages: Vec<Message>) -> Result<(), ProducerPublishError> {
        if !self.0.is_opened.load(std::sync::atomic::Ordering::SeqCst) {
            self.0.notify.notified().await;
        }

        let producer = self.0.producer.read("batch_send").await;
        let err = producer.batch_send(messages.clone(), async |_| {}).await;
        drop(producer);

        if let Err(e) = err {
            if matches!(
                e,
                ProducerPublishError::Timeout | ProducerPublishError::Closed
            ) {
                return Box::pin(self.batch_send(messages)).await;
            } else {
                return Err(e);
            }
        }

        Ok(())
    }
}

pub struct RabbitOperationReceiverCreator {
    environment: Box<Environment>,
    config: RabbitMQConsumerConfig,
}

impl RabbitOperationReceiverCreator {
    pub async fn try_new(config: InputRabbitMQConfig) -> Result<Self> {
        let environment = create_environment(config.client_options.clone()).await?;
        ensure_queue_exists(&environment, &config.consumer_config.stream_name).await?;

        Ok(Self {
            environment: Box::new(environment),
            config: config.consumer_config,
        })
    }

    /// `starting_offset` is the first operation not yet in the durable index.
    /// Fails with a [`ReplicationRefusal`] if the stream no longer retains it.
    pub async fn create(self, starting_offset: Offset) -> Result<RabbitOperationReceiver> {
        let startup_head =
            validate_stream_start(&self.environment, &self.config, starting_offset).await?;
        let consumer = create_consumer(&self.environment, &self.config, starting_offset)
            .await
            .context("Cannot create consumer")?;

        Ok(RabbitOperationReceiver {
            consumer,
            environment: self.environment,
            config: self.config,
            expected_offset: starting_offset,
            startup_head,
        })
    }
}

pub struct RabbitOperationSenderCreator {
    environment: Box<Environment>,
    producer_config: RabbitMQProducerConfig,
}

impl RabbitOperationSenderCreator {
    pub async fn try_new(config: OutputRabbitMQConfig) -> Result<Self> {
        let environment = create_environment(config.client_options).await?;
        ensure_queue_exists(&environment, &config.producer_config.stream_name).await?;

        Ok(Self {
            environment: Box::new(environment),
            producer_config: config.producer_config,
        })
    }

    pub async fn create(self) -> Result<RabbitOperationSender> {
        let producer = create_producer(self.environment.clone(), self.producer_config)
            .await
            .context("Cannot create producer")?;

        Ok(RabbitOperationSender::new(producer))
    }
}

#[derive(Clone)]
pub struct RabbitOperationSender {
    producer: MyHAProducer,
}
impl RabbitOperationSender {
    fn new(producer: MyHAProducer) -> Self {
        Self { producer }
    }

    pub async fn send_batch(&self, operations: &[WriteOperation]) -> Result<()> {
        let mut messages = Vec::with_capacity(operations.len());

        for operation in operations {
            let coll_id: Option<CollectionId> = match operation {
                WriteOperation::Collection(coll_id, _) => Some(*coll_id),
                WriteOperation::DeleteCollection(id) => Some(*id),
                WriteOperation::CreateCollection { id, .. } => Some(*id),
                WriteOperation::CreateCollection2 { id, .. } => Some(*id),
                #[allow(deprecated)]
                WriteOperation::KV(_) | WriteOperation::DocumentStorage(_) => None,
            };

            let op_type_id = operation.get_type_id();

            let message_body =
                bincode::serialize(&operation).context("Cannot serialize operation")?;

            let prop = Message::builder()
                .message_annotations()
                .message_builder()
                .body(message_body)
                .application_properties();

            let prop = if let Some(coll_id) = coll_id {
                prop.insert("coll_id", coll_id.to_string())
            } else {
                prop
            };
            let message = prop
                .insert("v", 1)
                .insert("op_type_id", op_type_id)
                .message_builder()
                .build();

            messages.push(message);
        }

        // Same problem as in `send`
        // TODO: think about a better solution
        self.producer
            .batch_send(messages)
            .await
            .context("Cannot send messages")?;

        Ok(())
    }

    pub async fn send(&self, operation: &WriteOperation) -> Result<()> {
        let coll_id: Option<CollectionId> = match operation {
            WriteOperation::Collection(coll_id, _) => Some(*coll_id),
            WriteOperation::DeleteCollection(id) => Some(*id),
            WriteOperation::CreateCollection { id, .. } => Some(*id),
            WriteOperation::CreateCollection2 { id, .. } => Some(*id),
            #[allow(deprecated)]
            WriteOperation::KV(_) | WriteOperation::DocumentStorage(_) => None,
        };

        let op_type_id = operation.get_type_id();

        let message_body = bincode::serialize(&operation).context("Cannot serialize operation")?;

        let prop = Message::builder()
            .message_annotations()
            .message_builder()
            .body(message_body)
            .application_properties();

        let prop = if let Some(coll_id) = coll_id {
            prop.insert("coll_id", coll_id.to_string())
        } else {
            prop
        };
        let message = prop
            .insert("v", 1)
            .insert("op_type_id", op_type_id)
            .message_builder()
            .build();

        // We aren't waiting for the result of the send
        // but we should care about the result
        // RabbitMQ will return an ACK if the message was sent successfully.
        // If that ACK is not received, it could be a problem.
        // Anyway, waiting for the result will slow down the insertion process
        // We should propose a better solution in the future
        // TODO: think about a better solution
        self.producer
            .batch_send(vec![message])
            .await
            .context("Cannot send message")?;

        Ok(())
    }
}

pub struct RabbitOperationReceiver {
    consumer: Consumer,
    environment: Box<Environment>,
    config: RabbitMQConsumerConfig,
    // Offset the next delivery must have. Anything else means the broker skipped
    // operations, and applying it would leave a partial index.
    expected_offset: Offset,
    // Stream head observed at creation; `None` if the stream was empty.
    startup_head: Option<Offset>,
}

// Bounds the broker round trips made while closing a consumer or probing the stream head,
// so a dead connection cannot hang the caller.
const CONSUMER_CLOSE_TIMEOUT: Duration = Duration::from_secs(5);
const HEAD_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
// The probe has read the last chunk once no message arrives for this long.
const HEAD_PROBE_IDLE: Duration = Duration::from_millis(500);

impl RabbitOperationReceiver {
    /// Stream head observed when the receiver was created, or `None` if the stream was empty.
    pub fn startup_head(&self) -> Option<Offset> {
        self.startup_head
    }

    /// Closes the current consumer and creates a new one that starts at `from`.
    /// Used both after the consumer ends and when it goes silent.
    pub async fn recreate(&mut self, from: Offset) -> Result<()> {
        // The old connection may be dead: closing it is best effort.
        match timeout(CONSUMER_CLOSE_TIMEOUT, self.consumer.handle().close()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => debug!(error = ?e, "Cannot close the previous consumer"),
            Err(_) => warn!("Timed out closing the previous consumer"),
        }

        validate_stream_start(&self.environment, &self.config, from).await?;
        self.expected_offset = from;
        self.consumer = create_consumer(&self.environment, &self.config, from)
            .await
            .context("Cannot recreate consumer")?;
        Ok(())
    }

    /// Returns the offset of the newest message in the stream, or `None` if the stream is empty.
    /// The stream client has no call for the stream head, so this opens a short-lived consumer
    /// on its own connection at the last chunk and reads that chunk to the end.
    pub async fn query_stream_head(&self) -> Result<Option<Offset>> {
        probe_stream_edge(
            &self.environment,
            &self.config,
            OffsetSpecification::Last,
            true,
        )
        .await
    }
}

impl Stream for RabbitOperationReceiver {
    type Item = Result<(Offset, WriteOperation), anyhow::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let polled = self.consumer.poll_next_unpin(cx);

        trace!("Polling RabbitMQ consumer: {:?}", polled);

        let delivery = match polled {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => return Poll::Ready(None),
            Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e.into()))),
            Poll::Ready(Some(Ok(delivery))) => delivery,
        };

        let offset = Offset(delivery.offset());
        if let Err(refusal) = validate_delivery_offset(self.expected_offset, offset) {
            return Poll::Ready(Some(Err(refusal.into())));
        }
        let Some(next) = offset.0.checked_add(1) else {
            return Poll::Ready(Some(Err(anyhow::anyhow!("Stream offset overflow"))));
        };
        self.expected_offset = Offset(next);
        let message = delivery.message();

        let data = match message.data() {
            None => {
                let e = anyhow::anyhow!("No data in message");
                error!(error =?e,"No data in message");
                return Poll::Ready(Some(Err(e)));
            }
            Some(data) => data,
        };

        debug!(
            "Received message. application_props: {:?}",
            message.application_properties()
        );

        let message_body: WriteOperation =
            match bincode::deserialize(data).context("Cannot deserialize operation") {
                Ok(op) => op,
                Err(e) => {
                    return Poll::Ready(Some(Err(anyhow::anyhow!(
                        "Error deserializing message: {e:?}"
                    ))));
                }
            };

        Poll::Ready(Some(Ok((offset, message_body))))
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct RabbitMQProducerConfig {
    pub stream_name: String,
    pub producer_name: String,
}
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct RabbitMQConsumerConfig {
    pub stream_name: String,
    pub consumer_name: String,
}

#[derive(Deserialize, Clone)]
pub struct OutputRabbitMQConfig {
    #[serde(flatten)]
    pub client_options: ClientOptions,
    #[serde(flatten)]
    pub producer_config: RabbitMQProducerConfig,
}
#[derive(Deserialize, Clone)]
pub struct InputRabbitMQConfig {
    #[serde(flatten)]
    pub client_options: ClientOptions,
    #[serde(flatten)]
    pub consumer_config: RabbitMQConsumerConfig,
}

async fn create_environment(client_options: ClientOptions) -> Result<Environment> {
    let environment = Environment::from_client_option(client_options)
        .await
        .context("Cannot create environment")?;

    Ok(environment)
}

async fn create_producer(
    environment: Box<Environment>,
    producer_config: RabbitMQProducerConfig,
) -> Result<MyHAProducer> {
    MyHAProducer::new(
        environment,
        &producer_config.producer_name,
        &producer_config.stream_name,
    )
    .await
    .context("Failed to create HA producer")
}

/// Why a reader refuses to replicate from the stream. Each message starts with a stable
/// code that tools match on to report the cause; keep the codes unchanged.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ReplicationRefusal {
    #[error("REPLICATION_HISTORY_EXPIRED: reader requires offset {required} but first retained offset is {first}")]
    HistoryExpired { required: u64, first: u64 },
    #[error(
        "REPLICATION_STREAM_BEHIND: reader requires offset {required} ahead of stream head {head}"
    )]
    AheadOfHead { required: u64, head: u64 },
    #[error(
        "REPLICATION_STREAM_BEHIND: reader requires offset {required} but the stream is empty"
    )]
    EmptyStream { required: u64 },
    #[error("REPLICATION_HISTORY_EXPIRED: reader requires offset {expected} but broker delivered {delivered}")]
    UnexpectedDelivery { expected: u64, delivered: u64 },
}

/// Checks that the stream still retains `required` before a consumer starts there. When the
/// requested offset is gone, the broker silently starts the consumer at the first retained
/// offset, so this must be checked up front. Returns the observed stream head, or `None`
/// for an empty stream.
///
/// A probe that times out on a non-empty stream looks empty. That is only accepted for
/// `required == 0`, and then the broker delivers the first retained offset, which
/// `validate_delivery_offset` refuses.
async fn validate_stream_start(
    environment: &Environment,
    config: &RabbitMQConsumerConfig,
    required: Offset,
) -> Result<Option<Offset>> {
    let Some(first) =
        probe_stream_edge(environment, config, OffsetSpecification::First, false).await?
    else {
        validate_empty_stream(required)?;
        return Ok(None);
    };
    let head = probe_stream_edge(environment, config, OffsetSpecification::Last, true)
        .await?
        .context("Cannot observe the stream head")?;
    validate_retained_range(required, first, head)?;
    Ok(Some(head))
}

/// A reader may start at `required` if the stream holds `first..=head`
/// and `required` is in `first..=head + 1`.
pub(crate) fn validate_retained_range(
    required: Offset,
    first: Offset,
    head: Offset,
) -> Result<(), ReplicationRefusal> {
    if required < first {
        return Err(ReplicationRefusal::HistoryExpired {
            required: required.0,
            first: first.0,
        });
    }
    if required.0 > head.0.saturating_add(1) {
        return Err(ReplicationRefusal::AheadOfHead {
            required: required.0,
            head: head.0,
        });
    }
    Ok(())
}

/// Retention never removes the active segment, so a stream that ever had a message is never
/// empty. An empty stream can only serve a reader that has applied nothing.
pub(crate) fn validate_empty_stream(required: Offset) -> Result<(), ReplicationRefusal> {
    if required.is_zero() {
        Ok(())
    } else {
        Err(ReplicationRefusal::EmptyStream {
            required: required.0,
        })
    }
}

/// Catches retention that moved between `validate_stream_start` and the consumer start,
/// and any other gap in the deliveries.
pub(crate) fn validate_delivery_offset(
    expected: Offset,
    delivered: Offset,
) -> Result<(), ReplicationRefusal> {
    if expected == delivered {
        Ok(())
    } else {
        Err(ReplicationRefusal::UnexpectedDelivery {
            expected: expected.0,
            delivered: delivered.0,
        })
    }
}

/// Opens a short-lived consumer on its own connection at `offset` and returns the offset of
/// the first message it receives, or with `read_to_end` the last one before it goes idle.
/// `None` means nothing arrived. The stream client has no call for the stream bounds.
async fn probe_stream_edge(
    environment: &Environment,
    config: &RabbitMQConsumerConfig,
    offset: OffsetSpecification,
    read_to_end: bool,
) -> Result<Option<Offset>> {
    let mut probe = timeout(
        HEAD_PROBE_TIMEOUT,
        environment
            .consumer()
            .client_provided_name(&format!("{}-stream-probe", config.consumer_name))
            .offset(offset)
            .build(&config.stream_name),
    )
    .await
    .context("Timed out creating the stream probe")?
    .context("Cannot create the stream probe")?;

    let mut observed: Option<u64> = None;
    let deadline = Instant::now() + HEAD_PROBE_TIMEOUT;
    let result = loop {
        let idle_deadline = (Instant::now() + HEAD_PROBE_IDLE).min(deadline);
        match timeout_at(idle_deadline, probe.next()).await {
            Ok(Some(Ok(delivery))) => {
                observed = Some(observed.map_or(delivery.offset(), |o| o.max(delivery.offset())));
                if !read_to_end {
                    break Ok(observed.map(Offset));
                }
            }
            Ok(Some(Err(e))) => break Err(anyhow::Error::from(e)),
            Ok(None) => break Err(anyhow::anyhow!("Stream probe closed unexpectedly")),
            // No new messages within the idle window: the chunk has been read.
            Err(_) => break Ok(observed.map(Offset)),
        }
    };

    if let Err(e) = timeout(CONSUMER_CLOSE_TIMEOUT, probe.handle().close()).await {
        debug!(error = ?e, "Timed out closing the stream probe");
    }

    result.context("Cannot read from the stream probe")
}

async fn create_consumer(
    environment: &Environment,
    config: &RabbitMQConsumerConfig,
    starting_offset: Offset,
) -> Result<Consumer> {
    create_consumer_attempt(environment, config, starting_offset, 0).await
}

async fn create_consumer_attempt(
    environment: &Environment,
    config: &RabbitMQConsumerConfig,
    starting_offset: Offset,
    attempt_count: u8,
) -> Result<Consumer> {
    if attempt_count > 5 {
        warn!("RabbitMQ continue to have some problems. Attempts are greater than 5");
    }
    if attempt_count > u8::MAX / 2 {
        // RabbitMQ has some internal problems and the reconnection mechanism doesn't work
        error!(
            "Stop retries at {}. RabbitMQ connection failure",
            attempt_count
        );
        bail!("RabbitMQ continue to not accept the consumer");
    }
    info!(
        "Creating rabbitmq consumer {} - {:?}. Attempt {}",
        &config.stream_name, starting_offset, attempt_count
    );
    let consumer = environment
        .consumer()
        .client_provided_name(&config.consumer_name)
        .offset(OffsetSpecification::Offset(starting_offset.0))
        .build(&config.stream_name)
        .await;

    let consumer = match consumer {
        Ok(consumer) => consumer,
        // RabbitMQ needs to stabilize the stream
        // and it tooks time
        // We need to put this code inside the rabbit SDK, but for now it is fine here
        Err(ConsumerCreateError::Create { status, .. })
        | Err(ConsumerCreateError::Client(ClientError::RequestError(status)))
            if status == ResponseCode::StreamNotAvailable =>
        {
            let waiting_time = random_wait_with_backoff(attempt_count + 1);
            sleep(Duration::from_millis(waiting_time)).await;
            return Box::pin(create_consumer_attempt(
                environment,
                config,
                starting_offset,
                attempt_count + 1,
            ))
            .await;
        }
        Err(e) => {
            return Err(e).context("Failed to create rabbit consumer");
        }
    };

    info!("Created rabbitmq consumer {}", &config.stream_name);
    Ok(consumer)
}

fn random_wait_with_backoff(attemp: u8) -> u64 {
    let r: u64 = rand::random_range(0..8_000);
    let base_wait = 3_000 + r;

    (base_wait * (1 << (attemp - 1))).min(15_000)
}

async fn ensure_queue_exists(environment: &Environment, stream_name: &str) -> Result<()> {
    let create_response = environment
        .stream_creator()
        .max_length(ByteCapacity::GB(5))
        .create(stream_name)
        .await;

    match create_response {
        Ok(_) => Ok(()),
        Err(StreamCreateError::Create {
            status: ResponseCode::StreamAlreadyExists,
            ..
        }) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RabbitOperationSender>();
        assert_send_sync::<RabbitOperationReceiver>();
    }
}

#[cfg(test)]
#[path = "../../../tests/replication_retention_broker.rs"]
mod replication_retention_broker;
