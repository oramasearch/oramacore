use std::fmt::Debug;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{error, trace, warn};

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
    RabbitMQ(RabbitOperationSender),
}

impl OperationSender {
    pub fn get_offset(&self) -> Offset {
        match self {
            OperationSender::InMemory { offset_counter, .. } => {
                Offset(offset_counter.load(std::sync::atomic::Ordering::SeqCst))
            }
            // This method is invoked only when the write side is committing.
            // In RabbitMQ case, we don't care to store that offset,
            // because RabbitMQ server keeps track of it.
            OperationSender::RabbitMQ { .. } => Offset(0),
        }
    }

    pub async fn send(&self, operation: WriteOperation) -> Result<()> {
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
            OperationSender::RabbitMQ(sender) => {
                sender.send(&operation).await?;
            }
        }

        Ok(())
    }
}

pub enum OperationReceiver {
    InMemory {
        receiver: tokio::sync::mpsc::Receiver<(Offset, Vec<u8>)>,
    },
    RabbitMQ(RabbitOperationReceiver),
}

impl OperationReceiver {
    pub async fn recv(&mut self) -> Option<Result<(Offset, WriteOperation)>> {
        match self {
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

                Some(Ok((offset, message_body)))
            }
            Self::RabbitMQ(receiver) => receiver.next().await,
        }
    }

    pub async fn reconnect(&mut self) -> Result<()> {
        match self {
            Self::InMemory { .. } => Ok(()),
            Self::RabbitMQ(receiver) => receiver.reconnect().await,
        }
    }
}

pub enum OperationSenderCreator {
    InMemory {
        sender: tokio::sync::mpsc::Sender<(Offset, Vec<u8>)>,
    },
    RabbitMQ(RabbitOperationSenderCreator),
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
            OperationSenderCreator::RabbitMQ(creator) => {
                let sender = creator
                    .create(offset)
                    .await
                    .context("Cannot create RabbitMQ sender")?;
                Ok(OperationSender::RabbitMQ(sender))
            }
        }
    }
}

pub enum OperationReceiverCreator {
    InMemory {
        receiver: tokio::sync::mpsc::Receiver<(Offset, Vec<u8>)>,
    },
    RabbitMQ(RabbitOperationReceiverCreator),
}

impl OperationReceiverCreator {
    pub async fn create(self, last_offset: Offset) -> Result<OperationReceiver> {
        match self {
            OperationReceiverCreator::InMemory { receiver } => {
                Ok(OperationReceiver::InMemory { receiver })
            }
            OperationReceiverCreator::RabbitMQ(creator) => {
                let receiver = creator
                    .create(last_offset)
                    .await
                    .context("Cannot create RabbitMQ receiver")?;
                Ok(OperationReceiver::RabbitMQ(receiver))
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
            let sender_creator = OperationSenderCreator::RabbitMQ(
                RabbitOperationSenderCreator::try_new(write_rabbit_config)
                    .await
                    .context("Cannot create RabbitMQ sender")?,
            );
            let receiver_creator = OperationReceiverCreator::RabbitMQ(
                RabbitOperationReceiverCreator::try_new(read_rabbit_config)
                    .await
                    .context("Cannot create RabbitMQ receiver")?,
            );

            Ok((Some(sender_creator), Some(receiver_creator)))
        }
        (Some(OutputSideChannelType::RabbitMQ(write_rabbit_config)), None) => {
            let sender_creator = OperationSenderCreator::RabbitMQ(
                RabbitOperationSenderCreator::try_new(write_rabbit_config)
                    .await
                    .context("Cannot create RabbitMQ sender")?,
            );
            Ok((Some(sender_creator), None))
        }
        (None, Some(InputSideChannelType::RabbitMQ(read_rabbit_config))) => {
            let receiver_creator = OperationReceiverCreator::RabbitMQ(
                RabbitOperationReceiverCreator::try_new(read_rabbit_config)
                    .await
                    .context("Cannot create RabbitMQ receiver")?,
            );

            Ok((None, Some(receiver_creator)))
        }
        (None, None) => Err(anyhow::anyhow!("write_side or read_side must be provided")),
    }
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
#[allow(clippy::large_enum_variant)]
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
#[allow(clippy::large_enum_variant)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OperationSender>();
        assert_send_sync::<OperationSenderCreator>();
        assert_send_sync::<OperationReceiver>();
        assert_send_sync::<OperationReceiverCreator>();
    }
}
