use std::fmt::Debug;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use anyhow::{Context, Result};
use rabbitmq_stream_client::types::Message;
use rabbitmq_stream_client::{Consumer, Environment, NoDedup, Producer};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{debug, error, trace, warn};

use crate::metrics::{Empty, OPERATION_GAUGE};

mod rabbit;
pub use rabbit::*;
mod op;
pub use op::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Offset(pub u64);

impl Offset {
    pub fn next(self) -> Offset {
        Offset(self.0 + 1)
    }
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone)]
pub enum OperationSender {
    InMemory {
        offset_counter: Arc<AtomicU64>,
        sender: tokio::sync::mpsc::Sender<(Offset, Vec<u8>)>,
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

        trace!("Sending operation: {:?}", operation);
        match self {
            OperationSender::InMemory {
                offset_counter,
                sender,
            } => {
                let offset = offset_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                let message_body =
                    bincode::serialize(&operation).context("Cannot serialize operation")?;

                sender.send((Offset(offset), message_body)).await?;
            }
            OperationSender::RabbitMQ { producer } => {
                let coll_id = match &operation {
                    WriteOperation::Collection(coll_id, _) => coll_id.clone(),
                    WriteOperation::CreateCollection { id, .. } => id.clone(),
                };

                let op_type_id = operation.get_type_id();

                let message_body =
                    bincode::serialize(&operation).context("Cannot serialize operation")?;

                let message = Message::builder()
                    .message_annotations()
                    .message_builder()
                    .body(message_body)
                    .application_properties()
                    .insert("coll_id", coll_id.0)
                    .insert("v", 1)
                    .insert("op_type_id", op_type_id)
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
        receiver: tokio::sync::mpsc::Receiver<(Offset, Vec<u8>)>,
    },
    RabbitMQ {
        consumer: Consumer,
    },
}

impl OperationReceiver {
    pub async fn recv(&mut self) -> Option<(Offset, WriteOperation)> {
        let r = match self {
            Self::InMemory { receiver } => {
                let (offset, data) = match receiver.recv().await {
                    None => {
                        warn!("No message received");
                        return None;
                    }
                    Some((offset, data)) => (offset, data),
                };

                let message_body: WriteOperation =
                    match bincode::deserialize(&data).context("Cannot deserialize operation") {
                        Ok(op) => op,
                        Err(e) => {
                            error!("Error deserializing message: {:?}", e);
                            return None;
                        }
                    };

                Some((offset, message_body))
            }
            Self::RabbitMQ { consumer, .. } => {
                trace!("Waiting for message");
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

                        debug!(
                            "Received message. application_props: {:?}",
                            message.application_properties()
                        );

                        let message_body: WriteOperation = match bincode::deserialize(data)
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
                    None => {
                        warn!("No message received");
                        None
                    }
                }
            }
        };

        OPERATION_GAUGE.create(Empty {}).decrement_by(1);
        r
    }
}

pub enum OperationSenderCreator {
    InMemory {
        sender: tokio::sync::mpsc::Sender<(Offset, Vec<u8>)>,
    },
    RabbitMQ {
        environment: Box<Environment>,
        producer_config: RabbitMQProducerConfig,
        name: String,
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
                name,
            } => {
                let producer = create_producer(*environment, producer_config, &name)
                    .await
                    .context("Cannot create producer")?;

                Ok(OperationSender::RabbitMQ { producer })
            }
        }
    }
}

pub enum OperationReceiverCreator {
    InMemory {
        receiver: tokio::sync::mpsc::Receiver<(Offset, Vec<u8>)>,
    },
    RabbitMQ {
        environment: Box<Environment>,
        consumer_config: RabbitMQConsumerConfig,
        name: String,
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
                name,
            } => {
                // We save the last offset in the consumer config
                // So, what we want to do is to start from the next offset
                let starting_offset = if last_offset.is_zero() {
                    last_offset
                } else {
                    last_offset.next()
                };
                let consumer =
                    create_consumer(*environment, &consumer_config, &name, starting_offset)
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
            Ok((
                Some(OperationSenderCreator::InMemory { sender }),
                Some(OperationReceiverCreator::InMemory { receiver }),
            ))
        }
        (Some(OutputSideChannelType::InMemory { .. }), _)
        | (_, Some(InputSideChannelType::InMemory { .. })) => Err(anyhow::anyhow!(
            "write_side and read_side must both be in-memory"
        )),
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
                environment: Box::new(writer_env),
                producer_config: write_rabbit_config.producer_config,
                name: "writer".to_string(),
            };
            let receiver = OperationReceiverCreator::RabbitMQ {
                environment: Box::new(reader_env),
                consumer_config: read_rabbit_config.consumer_config,
                name: "reader".to_string(),
            };

            Ok((Some(sender), Some(receiver)))
        }
        (Some(OutputSideChannelType::RabbitMQ(write_rabbit_config)), None) => {
            let writer_env = create_environment(write_rabbit_config.client_options).await?;

            ensure_queue_exists(
                &writer_env,
                &write_rabbit_config.producer_config.stream_name,
            )
            .await?;

            let sender = OperationSenderCreator::RabbitMQ {
                environment: Box::new(writer_env),
                producer_config: write_rabbit_config.producer_config,
                name: "writer".to_string(),
            };
            Ok((Some(sender), None))
        }
        (None, Some(InputSideChannelType::RabbitMQ(read_rabbit_config))) => {
            let reader_env = create_environment(read_rabbit_config.client_options).await?;

            ensure_queue_exists(&reader_env, &read_rabbit_config.consumer_config.stream_name)
                .await?;

            let receiver = OperationReceiverCreator::RabbitMQ {
                environment: Box::new(reader_env),
                consumer_config: read_rabbit_config.consumer_config,
                name: "reader".to_string(),
            };

            Ok((None, Some(receiver)))
        }
        (None, None) => Err(anyhow::anyhow!("write_side or read_side must be provided")),
    }
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
    RabbitMQ(Box<OutputRabbitMQConfig>),
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
    RabbitMQ(Box<InputRabbitMQConfig>),
}

fn default_in_memory_capacity() -> usize {
    10_000
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
