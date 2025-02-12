use anyhow::{Context, Result};
use rabbitmq_stream_client::{
    error::StreamCreateError,
    types::{ByteCapacity, OffsetSpecification, ResponseCode},
    ClientOptions, Consumer, Environment, NoDedup, Producer,
};
use serde::Deserialize;
use tracing::info;

use super::Offset;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct RabbitMQProducerConfig {
    pub stream_name: String,
}
#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct RabbitMQConsumerConfig {
    pub stream_name: String,
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

pub async fn create_environment(client_options: ClientOptions) -> Result<Environment> {
    let environment = Environment::from_client_option(client_options)
        .await
        .context("Cannot create environment")?;

    Ok(environment)
}

pub async fn create_producer(
    environment: Environment,
    producer_config: RabbitMQProducerConfig,
    name: &str,
) -> Result<Producer<NoDedup>> {
    environment
        .producer()
        .client_provided_name(name)
        .build(&producer_config.stream_name)
        .await
        .context("Failed to create rabbit producer")
}

pub async fn create_consumer(
    environment: Environment,
    config: &RabbitMQConsumerConfig,
    name: &str,
    starting_offset: Offset,
) -> Result<Consumer> {
    info!(
        "Creating rabbitmq consumer {} - {:?}",
        &config.stream_name, starting_offset
    );
    let consumer = environment
        .consumer()
        .client_provided_name(name)
        .offset(OffsetSpecification::Offset(starting_offset.0))
        .build(&config.stream_name)
        .await
        .context("Failed to create rabbit consumer")?;
    info!("Created rabbitmq consumer {}", &config.stream_name);
    Ok(consumer)
}

pub async fn ensure_queue_exists(environment: &Environment, stream_name: &str) -> Result<()> {
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
