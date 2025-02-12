use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::{collections::HashMap, fmt::Debug};

use anyhow::{Context, Result};
use rabbitmq_stream_client::error::StreamCreateError;
use rabbitmq_stream_client::types::{ByteCapacity, Message, OffsetSpecification, ResponseCode};
use rabbitmq_stream_client::{ClientOptions, Consumer, Environment, NoDedup, Producer};
use redact::Secret;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::error;

use crate::collection_manager::dto::{FieldId, Number};
use crate::metrics::{Empty, OPERATION_GAUGE};
use crate::types::{CollectionId, DocumentId, RawJSONDocument};

use crate::collection_manager::dto::{ApiKey, TypedField};

#[derive(Debug, Clone)]
pub enum GenericWriteOperation {
    CreateCollection,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Term(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermStringField {
    pub positions: Vec<usize>,
}

pub type InsertStringTerms = HashMap<Term, TermStringField>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentFieldIndexOperation {
    IndexString {
        field_length: u16,
        terms: InsertStringTerms,
    },
    IndexEmbedding {
        value: Vec<f32>,
    },
    IndexNumber {
        value: Number,
    },
    IndexBoolean {
        value: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionWriteOperation {
    InsertDocument {
        doc_id: DocumentId,
        doc: RawJSONDocument,
    },
    DeleteDocuments {
        doc_ids: Vec<DocumentId>,
    },
    CreateField {
        field_id: FieldId,
        field_name: String,
        field: TypedField,
    },
    Index(DocumentId, FieldId, DocumentFieldIndexOperation),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteOperation {
    CreateCollection {
        id: CollectionId,
        #[serde(
            deserialize_with = "deserialize_api_key",
            serialize_with = "serialize_api_key"
        )]
        read_api_key: ApiKey,
        // Params here... but which ones?
        // TODO: add params
    },
    Collection(CollectionId, CollectionWriteOperation),
}

fn serialize_api_key<S>(x: &ApiKey, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    s.serialize_str(x.0.expose_secret())
}
fn deserialize_api_key<'de, D>(deserializer: D) -> Result<ApiKey, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    String::deserialize(deserializer).map(|s| ApiKey(Secret::from(s)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Offset(pub u64);

impl Offset {
    pub fn next(self) -> Offset {
        Offset(self.0 + 1)
    }
}

#[derive(Clone)]
pub enum OperationSender {
    InMemory {
        offset_counter: Arc<AtomicU64>,
        sender: tokio::sync::mpsc::Sender<(Offset, WriteOperation)>,
    },
    RabbitMQ {
        producer: Producer<NoDedup>,
    },
}

impl OperationSender {
    pub fn get_offset(&self) -> Offset {
        match self {
            OperationSender::InMemory { offset_counter, .. } => {
                Offset(offset_counter.load(std::sync::atomic::Ordering::SeqCst))
            }
            OperationSender::RabbitMQ { .. } => Offset(0),
        }
    }

    pub async fn send(&self, operation: WriteOperation) -> Result<()> {
        OPERATION_GAUGE.create(Empty {}).increment_by(1);

        match self {
            OperationSender::InMemory {
                offset_counter,
                sender,
            } => {
                let offset = offset_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                sender.send((Offset(offset), operation)).await?;
            }
            OperationSender::RabbitMQ { producer } => {
                let coll_id = match &operation {
                    WriteOperation::Collection(coll_id, _) => coll_id.clone(),
                    WriteOperation::CreateCollection { id, .. } => id.clone(),
                };

                let message_body =
                    bincode::serialize(&operation).context("Cannot serialize operation")?;

                let message = Message::builder()
                    .message_annotations()
                    .message_builder()
                    .body(message_body)
                    .application_properties()
                    .insert("coll_id", coll_id.0)
                    .insert("v", 1)
                    .message_builder()
                    .build();

                producer
                    .send(message, move |r| async move {
                        if let Err(e) = r {
                            error!("Message send error {:?}", e);
                        }
                    })
                    .await?;
            }
        }

        Ok(())
    }
}

pub enum OperationReceiver {
    InMemory {
        receiver: tokio::sync::mpsc::Receiver<(Offset, WriteOperation)>,
    },
    RabbitMQ {
        consumer: Consumer,
    },
}

impl OperationReceiver {
    pub async fn recv(&mut self) -> Option<(Offset, WriteOperation)> {
        let r = match self {
            Self::InMemory { receiver } => receiver.recv().await,
            Self::RabbitMQ { consumer, .. } => {
                let delivery = consumer.next().await;

                match delivery {
                    Some(Ok(delivery)) => {
                        let message = delivery.message();
                        let offset = Offset(delivery.offset());
                        let data = match message.data() {
                            None => {
                                error!("No data in message");
                                return None;
                            }
                            Some(data) => data,
                        };

                        let message_body: WriteOperation = match bincode::deserialize(&data)
                            .context("Cannot deserialize operation")
                        {
                            Ok(op) => op,
                            Err(e) => {
                                error!("Error deserializing message: {:?}", e);
                                return None;
                            }
                        };

                        Some((offset, message_body))
                    }
                    Some(Err(e)) => {
                        error!("Error receiving message: {:?}", e);
                        None
                    }
                    None => None,
                }
            }
        };

        OPERATION_GAUGE.create(Empty {}).decrement_by(1);
        r
    }
}

pub enum OperationSenderCreator {
    InMemory {
        sender: tokio::sync::mpsc::Sender<(Offset, WriteOperation)>,
    },
    RabbitMQ {
        environment: Environment,
        producer_config: RabbitMQProducerConfig,
    },
}
impl OperationSenderCreator {
    pub async fn create(self, offset: Offset) -> Result<OperationSender> {
        match self {
            OperationSenderCreator::InMemory { sender } => {
                let offset_counter = Arc::new(AtomicU64::new(offset.0));
                Ok(OperationSender::InMemory {
                    offset_counter,
                    sender,
                })
            }
            OperationSenderCreator::RabbitMQ {
                environment,
                producer_config,
            } => {
                let producer = create_producer(environment, producer_config)
                    .await
                    .context("Cannot create producer")?;

                Ok(OperationSender::RabbitMQ { producer: producer })
            }
        }
    }
}

pub enum OperationReceiverCreator {
    InMemory {
        receiver: tokio::sync::mpsc::Receiver<(Offset, WriteOperation)>,
    },
    RabbitMQ {
        environment: Environment,
        consumer_config: RabbitMQConsumerConfig,
    },
}

impl OperationReceiverCreator {
    pub async fn create(self, last_offset: Offset) -> Result<OperationReceiver> {
        match self {
            OperationReceiverCreator::InMemory { receiver } => {
                Ok(OperationReceiver::InMemory { receiver })
            }
            OperationReceiverCreator::RabbitMQ {
                environment,
                consumer_config,
            } => {
                let consumer = create_consumer(
                    environment,
                    &consumer_config,
                    // We save the last offset in the consumer config
                    // So, what we want to do is to start from the next offset
                    last_offset.next(),
                )
                .await
                .context("Cannot create consumer")?;

                Ok(OperationReceiver::RabbitMQ { consumer })
            }
        }
    }
}

pub async fn channel_creator(
    write_side: Option<OutputSideChannelType>,
    read_side: Option<InputSideChannelType>,
) -> Result<(
    Option<OperationSenderCreator>,
    Option<OperationReceiverCreator>,
)> {
    match (write_side, read_side) {
        (
            Some(OutputSideChannelType::InMemory {
                capacity: write_side_capacity,
            }),
            Some(InputSideChannelType::InMemory {
                capacity: read_side_capacity,
            }),
        ) => {
            if write_side_capacity != read_side_capacity {
                return Err(anyhow::anyhow!(
                    "write_side_capacity and read_side_capacity must be equal"
                ));
            }

            let (sender, receiver) = tokio::sync::mpsc::channel(write_side_capacity);
            return Ok((
                Some(OperationSenderCreator::InMemory { sender }),
                Some(OperationReceiverCreator::InMemory { receiver }),
            ));
        }
        (Some(OutputSideChannelType::InMemory { .. }), _)
        | (_, Some(InputSideChannelType::InMemory { .. })) => {
            return Err(anyhow::anyhow!(
                "write_side and read_side must both be in-memory"
            ));
        }
        (
            Some(OutputSideChannelType::RabbitMQ(write_rabbit_config)),
            Some(InputSideChannelType::RabbitMQ(read_rabbit_config)),
        ) => {
            let writer_env = create_environment(write_rabbit_config.client_options).await?;
            let reader_env = create_environment(read_rabbit_config.client_options).await?;

            ensure_queue_exists(
                &writer_env,
                &write_rabbit_config.producer_config.stream_name,
            )
            .await?;
            ensure_queue_exists(&reader_env, &read_rabbit_config.consumer_config.stream_name)
                .await?;

            let sender = OperationSenderCreator::RabbitMQ {
                environment: writer_env,
                producer_config: write_rabbit_config.producer_config,
            };
            let receiver = OperationReceiverCreator::RabbitMQ {
                environment: reader_env,
                consumer_config: read_rabbit_config.consumer_config,
            };

            return Ok((Some(sender), Some(receiver)));
        }
        (Some(OutputSideChannelType::RabbitMQ(write_rabbit_config)), None) => {
            let writer_env = create_environment(write_rabbit_config.client_options).await?;

            ensure_queue_exists(
                &writer_env,
                &write_rabbit_config.producer_config.stream_name,
            )
            .await?;

            let sender = OperationSenderCreator::RabbitMQ {
                environment: writer_env,
                producer_config: write_rabbit_config.producer_config,
            };
            return Ok((Some(sender), None));
        }
        (None, Some(InputSideChannelType::RabbitMQ(read_rabbit_config))) => {
            let reader_env = create_environment(read_rabbit_config.client_options).await?;

            ensure_queue_exists(&reader_env, &read_rabbit_config.consumer_config.stream_name)
                .await?;

            let receiver = OperationReceiverCreator::RabbitMQ {
                environment: reader_env,
                consumer_config: read_rabbit_config.consumer_config,
            };

            return Ok((None, Some(receiver)));
        }
        (None, None) => {
            return Err(anyhow::anyhow!("write_side or read_side must be provided"));
        }
    }
}

async fn create_environment(client_options: ClientOptions) -> Result<Environment> {
    let environment = Environment::from_client_option(client_options)
        .await
        .context("Cannot create environment")?;

    Ok(environment)
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct RabbitMQProducerConfig {
    stream_name: String,
    client_provided_name: String,
}
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct RabbitMQConsumerConfig {
    stream_name: String,
    client_provided_name: String,
}

#[derive(Deserialize, Clone)]
pub struct OutputRabbitMQConfig {
    #[serde(flatten)]
    client_options: ClientOptions,
    #[serde(flatten)]
    producer_config: RabbitMQProducerConfig,
}
#[derive(Deserialize, Clone)]
pub struct InputRabbitMQConfig {
    #[serde(flatten)]
    client_options: ClientOptions,
    #[serde(flatten)]
    consumer_config: RabbitMQConsumerConfig,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum OutputSideChannelType {
    #[serde(rename = "in-memory")]
    InMemory {
        #[serde(default = "default_in_memory_capacity")]
        capacity: usize,
    },
    #[serde(rename = "rabbitmq")]
    RabbitMQ(OutputRabbitMQConfig),
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum InputSideChannelType {
    #[serde(rename = "in-memory")]
    InMemory {
        #[serde(default = "default_in_memory_capacity")]
        capacity: usize,
    },
    #[serde(rename = "rabbitmq")]
    RabbitMQ(InputRabbitMQConfig),
}

fn default_in_memory_capacity() -> usize {
    10_000
}

async fn create_producer(
    environment: Environment,
    producer_config: RabbitMQProducerConfig,
) -> Result<Producer<NoDedup>> {
    environment
        .producer()
        .client_provided_name(&producer_config.client_provided_name)
        .build(&producer_config.stream_name)
        .await
        .context("Failed to create rabbit producer")
}

async fn create_consumer(
    environment: Environment,
    config: &RabbitMQConsumerConfig,
    starting_offset: Offset,
) -> Result<Consumer> {
    environment
        .consumer()
        .client_provided_name(&config.client_provided_name)
        .offset(OffsetSpecification::Offset(starting_offset.0))
        .build(&config.stream_name)
        .await
        .context("Failed to create rabbit consumer")
}

async fn ensure_queue_exists(environment: &Environment, stream_name: &str) -> Result<()> {
    let create_response = environment
        .stream_creator()
        // TODO: put those values in the config
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
    use rabbitmq_stream_client::types::SuperStreamProducer;

    use super::*;

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SuperStreamProducer<NoDedup>>();
        assert_send_sync::<OperationSender>();
        assert_send_sync::<OperationSenderCreator>();
        assert_send_sync::<OperationReceiver>();
        assert_send_sync::<OperationReceiverCreator>();
    }
}
