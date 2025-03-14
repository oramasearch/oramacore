use std::{pin::Pin, task::Poll, time::Instant};

use anyhow::{Context, Result};
use futures::{Stream, StreamExt};
use rabbitmq_stream_client::{
    error::StreamCreateError,
    types::{ByteCapacity, Message, OffsetSpecification, ResponseCode},
    ClientOptions, Consumer, Environment, NoDedup, Producer,
};
use serde::Deserialize;
use tracing::{debug, error, info};

use crate::{
    metrics::{rabbit::RABBITMQ_ENQUEUE_CALCULATION_TIME, Empty},
    types::CollectionId,
};

use super::{Offset, WriteOperation};

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

    pub async fn create(self, last_offset: Offset) -> Result<RabbitOperationReceiver> {
        // We save the last offset in the consumer config
        // So, what we want to do is to start from the next offset
        let starting_offset = if last_offset.is_zero() {
            last_offset
        } else {
            last_offset.next()
        };

        let consumer = create_consumer(&self.environment, &self.config, starting_offset)
            .await
            .context("Cannot create consumer")?;

        Ok(RabbitOperationReceiver {
            consumer,
            environment: self.environment,
            config: self.config,
            last_offset,
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

    pub async fn create(self, _: Offset) -> Result<RabbitOperationSender> {
        let producer = create_producer(&self.environment, self.producer_config)
            .await
            .context("Cannot create producer")?;

        Ok(RabbitOperationSender::new(producer))
    }
}

#[derive(Clone)]
pub struct RabbitOperationSender {
    producer: Producer<NoDedup>,
}
impl RabbitOperationSender {
    pub fn new(producer: Producer<NoDedup>) -> Self {
        Self { producer }
    }

    pub async fn send(&self, operation: &WriteOperation) -> Result<()> {
        let coll_id: Option<CollectionId> = match operation {
            WriteOperation::Collection(coll_id, _) => Some(coll_id.clone()),
            WriteOperation::DeleteCollection(id) => Some(id.clone()),
            WriteOperation::CreateCollection { id, .. } => Some(id.clone()),
            WriteOperation::KV(_) => None,
            WriteOperation::SubstituteCollection {
                target_collection_id,
                ..
            } => Some(target_collection_id.clone()),
        };

        let op_type_id = operation.get_type_id();

        let message_body = bincode::serialize(&operation).context("Cannot serialize operation")?;

        let prop = Message::builder()
            .message_annotations()
            .message_builder()
            .body(message_body)
            .application_properties();

        let prop = if let Some(coll_id) = coll_id {
            prop.insert("coll_id", coll_id.0)
        } else {
            prop
        };
        let message = prop
            .insert("v", 1)
            .insert("op_type_id", op_type_id)
            .message_builder()
            .build();

        // Send interface accepts a Fn and not an FnOnce.
        // So, we cannot drop the TimeHistogram here as usual.
        // TODO: change the send interface to accept FnOnce
        let i = Instant::now();
        self.producer
            .send(message, move |r| {
                RABBITMQ_ENQUEUE_CALCULATION_TIME.track(Empty {}, i.elapsed());

                if let Err(e) = r {
                    error!("Message send error {:?}", e);
                }
                async {}
            })
            .await
            .context("Cannot send message")?;

        Ok(())
    }
}

pub struct RabbitOperationReceiver {
    consumer: Consumer,
    environment: Box<Environment>,
    config: RabbitMQConsumerConfig,
    last_offset: Offset,
}

impl RabbitOperationReceiver {
    pub async fn reconnect(&mut self) -> Result<()> {
        self.consumer = create_consumer(
            &self.environment,
            &self.config,
            Offset(self.last_offset.0 + 1),
        )
        .await
        .context("Cannot reconnect")?;
        Ok(())
    }
}

impl Stream for RabbitOperationReceiver {
    type Item = Result<(Offset, WriteOperation), anyhow::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let delivery = match self.consumer.poll_next_unpin(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(None) => return Poll::Ready(None),
            Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e.into()))),
            Poll::Ready(Some(Ok(delivery))) => delivery,
        };

        let message = delivery.message();
        let offset = Offset(delivery.offset());

        // Save last offset in case of reconnect
        self.last_offset = offset;

        let data = match message.data() {
            None => {
                error!("No data in message");
                return Poll::Ready(Some(Err(anyhow::anyhow!("No data in message"))));
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
                        "Error deserializing message: {:?}",
                        e
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
    println!("Creating environment {:#?}", client_options);
    let environment = Environment::from_client_option(client_options)
        .await
        .context("Cannot create environment")?;

    Ok(environment)
}

async fn create_producer(
    environment: &Environment,
    producer_config: RabbitMQProducerConfig,
) -> Result<Producer<NoDedup>> {
    environment
        .producer()
        .client_provided_name(&producer_config.producer_name)
        .build(&producer_config.stream_name)
        .await
        .context("Failed to create rabbit producer")
}

async fn create_consumer(
    environment: &Environment,
    config: &RabbitMQConsumerConfig,
    starting_offset: Offset,
) -> Result<Consumer> {
    info!(
        "Creating rabbitmq consumer {} - {:?}",
        &config.stream_name, starting_offset
    );
    let consumer = environment
        .consumer()
        .client_provided_name(&config.consumer_name)
        .offset(OffsetSpecification::Offset(starting_offset.0))
        .build(&config.stream_name)
        .await
        .context("Failed to create rabbit consumer")?;
    info!("Created rabbitmq consumer {}", &config.stream_name);
    Ok(consumer)
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
