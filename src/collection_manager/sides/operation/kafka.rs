use anyhow::{bail, Context, Result};
use futures::{FutureExt, Stream};
use rdkafka::{
    client::ClientContext,
    config::ClientConfig,
    consumer::{BaseConsumer, Consumer, ConsumerContext, Rebalance, StreamConsumer},
    message::{Headers, OwnedHeaders},
    producer::{FutureProducer, FutureRecord},
    Message, Offset as KafkaOffset, TopicPartitionList,
};
use serde::Deserialize;
use std::{pin::Pin, sync::Arc, task::Poll, time::Duration};
use tokio::sync::Notify;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

use crate::types::CollectionId;

use super::{Offset, WriteOperation};

/// Custom context for consumer to handle rebalancing events
struct KafkaConsumerContext {
    notify: Arc<Notify>,
}

impl ClientContext for KafkaConsumerContext {}

impl ConsumerContext for KafkaConsumerContext {
    fn pre_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance) {
        info!("Pre rebalance {:?}", rebalance);
    }

    fn post_rebalance(&self, _consumer: &BaseConsumer<Self>, rebalance: &Rebalance) {
        info!("Post rebalance {:?}", rebalance);
        self.notify.notify_waiters();
    }
}

#[derive(Clone)]
pub struct KafkaOperationSender {
    producer: Arc<FutureProducer>,
    topic: String,
}

impl KafkaOperationSender {
    fn new(producer: FutureProducer, topic: String) -> Self {
        Self {
            producer: Arc::new(producer),
            topic,
        }
    }

    pub async fn send_batch(&self, operations: &[(WriteOperation, Offset)]) -> Result<()> {
        let mut records_and_payloads = Vec::with_capacity(operations.len());
        for (operation, offset) in operations {
            let coll_id: Option<CollectionId> = match operation {
                WriteOperation::Collection(coll_id, _) => Some(*coll_id),
                WriteOperation::DeleteCollection(id) => Some(*id),
                WriteOperation::CreateCollection { id, .. } => Some(*id),
                WriteOperation::CreateCollection2 { id, .. } => Some(*id),
                WriteOperation::KV(_) | WriteOperation::DocumentStorage(_) => None,
            };

            let op_type_id = operation.get_type_id();

            let message_body =
                bincode::serialize(&operation).context("Cannot serialize operation")?;

            let v_bytes = 1u32.to_le_bytes();
            let op_type_id_bytes = op_type_id.as_bytes();
            let offset_bytes = offset.0.to_le_bytes();

            let mut headers = OwnedHeaders::new()
                .insert(rdkafka::message::Header {
                    key: "v",
                    value: Some(&v_bytes),
                })
                .insert(rdkafka::message::Header {
                    key: "op_type_id",
                    value: Some(op_type_id_bytes),
                })
                .insert(rdkafka::message::Header {
                    key: "offset",
                    value: Some(&offset_bytes),
                });

            if let Some(coll_id) = coll_id {
                let key = coll_id.as_str().as_bytes();
                headers = headers.insert(rdkafka::message::Header {
                    key: "coll_id",
                    value: Some(key),
                });
            } else {
                let key = b"";
                headers = headers.insert(rdkafka::message::Header {
                    key: "coll_id",
                    value: Some(key),
                });
            }

            records_and_payloads.push((message_body, headers));
        }

        // Phase 2: Queue all send futures (rdkafka batches internally)
        let mut send_futures = Vec::with_capacity(records_and_payloads.len());

        for (message_body, headers) in &records_and_payloads {
            let record: FutureRecord<String, Vec<u8>> = FutureRecord::to(&self.topic)
                .payload(message_body)
                .headers(headers.clone());

            let send_future = self.producer.send(record, Duration::from_secs(5));
            send_futures.push(send_future);
        }

        // Phase 3: Await all confirmations in order (preserves message ordering)
        for send_future in send_futures {
            send_future
                .await
                .map_err(|(err, _)| err)
                .context("Cannot send message to Kafka")?;
        }

        Ok(())
    }

    pub async fn send(&self, operation: &WriteOperation, offset: Offset) -> Result<()> {
        self.send_batch(&[(operation.clone(), offset)]).await
    }
}

pub struct KafkaOperationReceiverCreator {
    consumer_config: KafkaConsumerConfig,
}

impl KafkaOperationReceiverCreator {
    pub async fn try_new(config: InputKafkaConfig) -> Result<Self> {
        // Validate configuration by attempting to create a test consumer
        let mut client_config = ClientConfig::new();
        client_config.set("bootstrap.servers", &config.consumer_config.brokers);
        client_config.set("group.id", &config.consumer_config.group_id);

        // Test connection
        let _test_consumer: StreamConsumer = client_config
            .create()
            .context("Cannot create test Kafka consumer")?;

        Ok(Self {
            consumer_config: config.consumer_config,
        })
    }

    pub fn get_initial_offset(&self) -> Offset {
        Offset(self.consumer_config.initial_offset)
    }

    pub async fn create(self, last_offset: Offset) -> Result<KafkaOperationReceiver> {
        // We save the last offset in the consumer config
        // So, what we want to do is to start from the next offset
        let starting_offset = if last_offset.is_zero() {
            last_offset
        } else {
            last_offset.next()
        };

        let notify = Arc::new(Notify::new());
        let context = KafkaConsumerContext {
            notify: notify.clone(),
        };

        let consumer = create_consumer(&self.consumer_config, starting_offset, context)
            .await
            .context("Cannot create Kafka consumer")?;

        Ok(KafkaOperationReceiver {
            consumer,
            config: self.consumer_config,
            last_offset,
            notify,
        })
    }
}

pub struct KafkaOperationSenderCreator {
    producer_config: KafkaProducerConfig,
}

impl KafkaOperationSenderCreator {
    pub async fn try_new(config: OutputKafkaConfig) -> Result<Self> {
        // Validate configuration by attempting to create a test producer
        let mut client_config = ClientConfig::new();
        client_config.set("bootstrap.servers", &config.producer_config.brokers);
        client_config.set("client.id", &config.producer_config.client_id);

        // Test connection
        let _test_producer: FutureProducer = client_config
            .create()
            .context("Cannot create test Kafka producer")?;

        Ok(Self {
            producer_config: config.producer_config,
        })
    }

    pub fn get_initial_offset(&self) -> Offset {
        Offset(self.producer_config.initial_offset)
    }

    pub async fn create(self) -> Result<KafkaOperationSender> {
        let producer = create_producer(&self.producer_config)
            .await
            .context("Cannot create Kafka producer")?;

        Ok(KafkaOperationSender::new(
            producer,
            self.producer_config.topic.clone(),
        ))
    }
}

pub struct KafkaOperationReceiver {
    consumer: StreamConsumer<KafkaConsumerContext>,
    config: KafkaConsumerConfig,
    last_offset: Offset,
    notify: Arc<Notify>,
}

impl KafkaOperationReceiver {
    pub async fn reconnect(&mut self) -> Result<()> {
        let context = KafkaConsumerContext {
            notify: self.notify.clone(),
        };

        self.consumer = create_consumer(&self.config, self.last_offset.next(), context)
            .await
            .context("Cannot reconnect to Kafka")?;
        Ok(())
    }
}

impl Stream for KafkaOperationReceiver {
    type Item = Result<(Offset, WriteOperation), anyhow::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let message = match self.consumer.recv().now_or_never() {
            None => {
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
            Some(Err(e)) => {
                error!("Error receiving from Kafka consumer: {:?}", e);
                return Poll::Ready(Some(Err(e.into())));
            }
            Some(Ok(message)) => message,
        };

        let offset = if let Some(headers) = message.headers() {
            if let Some(offset_header) = headers.iter().find(|h| h.key == "offset") {
                if let Some(value) = offset_header.value {
                    if value.len() == 8 {
                        let offset_u64 = u64::from_le_bytes(value.try_into().unwrap());
                        debug!("Using writer-generated offset from header: {}", offset_u64);
                        Offset(offset_u64)
                    } else {
                        warn!("Offset header present but wrong size, falling back to Kafka offset");
                        Offset(message.offset() as u64)
                    }
                } else {
                    debug!("Offset header present but empty, using Kafka offset");
                    Offset(message.offset() as u64)
                }
            } else {
                debug!("No offset header found, using Kafka offset");
                Offset(message.offset() as u64)
            }
        } else {
            debug!("No headers in message, using Kafka offset");
            Offset(message.offset() as u64)
        };

        let data = match message.payload() {
            None => {
                error!("No payload in Kafka message");
                return Poll::Ready(Some(Err(anyhow::anyhow!("No payload in message"))));
            }
            Some(data) => data,
        };

        debug!("Received Kafka message with offset: {:?}", offset);

        let message_body: WriteOperation =
            match bincode::deserialize(data).context("Cannot deserialize operation") {
                Ok(op) => op,
                Err(e) => {
                    return Poll::Ready(Some(Err(anyhow::anyhow!(
                        "Error deserializing message: {:?}",
                        e
                    ))));
                }
            };

        // Save last offset in case of reconnect (now we can borrow self mutably)
        self.last_offset = offset;

        Poll::Ready(Some(Ok((offset, message_body))))
    }
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct KafkaProducerConfig {
    pub brokers: String,
    pub topic: String,
    pub client_id: String,
    #[serde(default)]
    pub initial_offset: u64,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct KafkaConsumerConfig {
    pub brokers: String,
    pub topic: String,
    pub group_id: String,
    pub client_id: String,
    #[serde(default)]
    pub initial_offset: u64,
}

#[derive(Deserialize, Clone)]
pub struct OutputKafkaConfig {
    #[serde(flatten)]
    pub producer_config: KafkaProducerConfig,
}

#[derive(Deserialize, Clone)]
pub struct InputKafkaConfig {
    #[serde(flatten)]
    pub consumer_config: KafkaConsumerConfig,
}

async fn create_producer(producer_config: &KafkaProducerConfig) -> Result<FutureProducer> {
    let mut client_config = ClientConfig::new();
    client_config.set("bootstrap.servers", &producer_config.brokers);
    client_config.set("client.id", &producer_config.client_id);
    client_config.set("message.timeout.ms", "5000");
    client_config.set("queue.buffering.max.messages", "100000");
    client_config.set("queue.buffering.max.kbytes", "1048576");

    let producer: FutureProducer = client_config
        .create()
        .context("Cannot create Kafka producer")?;

    info!(
        "Created Kafka producer for topic: {}",
        producer_config.topic
    );

    Ok(producer)
}

async fn create_consumer(
    config: &KafkaConsumerConfig,
    starting_offset: Offset,
    context: KafkaConsumerContext,
) -> Result<StreamConsumer<KafkaConsumerContext>> {
    create_consumer_attempt(config, starting_offset, 0, context).await
}

async fn create_consumer_attempt(
    config: &KafkaConsumerConfig,
    starting_offset: Offset,
    attempt_count: u8,
    context: KafkaConsumerContext,
) -> Result<StreamConsumer<KafkaConsumerContext>> {
    if attempt_count > 5 {
        warn!("Kafka continues to have connection problems. Attempts greater than 5");
    }
    if attempt_count > u8::MAX / 2 {
        error!(
            "Stop retries at {}. Kafka connection failure",
            attempt_count
        );
        bail!("Kafka continues to not accept the consumer");
    }

    info!(
        "Creating Kafka consumer for topic {} with offset {:?}. Attempt {}",
        config.topic, starting_offset, attempt_count
    );

    let mut client_config = ClientConfig::new();
    client_config.set("bootstrap.servers", &config.brokers);
    client_config.set("group.id", &config.group_id);
    client_config.set("client.id", &config.client_id);
    client_config.set("enable.auto.commit", "true");
    client_config.set("auto.commit.interval.ms", "5000");
    client_config.set("session.timeout.ms", "6000");
    client_config.set("enable.partition.eof", "false");

    let consumer: StreamConsumer<KafkaConsumerContext> = client_config
        .create_with_context(context)
        .context("Cannot create Kafka consumer")?;

    consumer
        .subscribe(&[&config.topic])
        .context("Cannot subscribe to Kafka topic")?;

    // Wait a bit for assignment
    sleep(Duration::from_millis(100)).await;

    // Seek to the desired offset if not starting from beginning
    if starting_offset.0 > 0 {
        let mut tpl = TopicPartitionList::new();
        // Note: This assumes partition 0. For production, you'd want to handle multiple partitions
        tpl.add_partition_offset(
            &config.topic,
            0,
            KafkaOffset::Offset(starting_offset.0 as i64),
        )
        .ok();

        if let Err(e) = consumer.assign(&tpl) {
            warn!("Failed to seek to offset {:?}: {:?}", starting_offset, e);
            // Continue anyway - auto.offset.reset will handle it
        }
    }

    info!("Created Kafka consumer for topic: {}", config.topic);

    Ok(consumer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<KafkaOperationSender>();
        assert_send_sync::<KafkaOperationReceiver>();
    }
}
