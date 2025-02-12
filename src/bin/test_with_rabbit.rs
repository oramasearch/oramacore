#![allow(dead_code)]

use std::time::Duration;

use anyhow::Result;

use oramacore::{
    collection_manager::{
        dto::ApiKey,
        sides::{
            channel_creator, create_consumer, create_environment, create_producer,
            InputRabbitMQConfig, InputSideChannelType, Offset, OutputRabbitMQConfig,
            OutputSideChannelType, RabbitMQConsumerConfig, RabbitMQProducerConfig, WriteOperation,
        },
    },
    types::CollectionId,
};
use rabbitmq_stream_client::{types::Message, ClientOptions};
use redact::Secret;
use tokio::time::sleep;
use tokio_stream::StreamExt;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter, Registry};

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = Registry::default().with(fmt::layer().compact().with_ansi(true));
    let subscriber = subscriber.with(EnvFilter::from_default_env());
    tracing::subscriber::set_global_default(subscriber).unwrap();

    main2().await?;

    Ok(())
}

async fn main1() -> Result<()> {
    let producer_env = create_environment(ClientOptions::default()).await?;
    let consumer_env = create_environment(ClientOptions::default()).await?;

    let producer = create_producer(
        producer_env,
        RabbitMQProducerConfig {
            stream_name: "foo".to_string(),
        },
        "writer",
    )
    .await?;
    let mut consumer = create_consumer(
        consumer_env,
        &RabbitMQConsumerConfig {
            stream_name: "foo".to_string(),
        },
        "reader",
        Offset(0),
    )
    .await?;

    println!("Sending message");

    let msg = Message::builder()
        .body(format!("stream message_{}", 0))
        .build();
    producer.send(msg, |_| async {}).await?;

    println!("Receiving message");
    let a = consumer.next().await;
    println!("{:?}", a);

    println!("Done");

    Ok(())
}

async fn main2() -> Result<()> {
    let output: OutputSideChannelType = OutputSideChannelType::RabbitMQ(OutputRabbitMQConfig {
        client_options: ClientOptions::default(),
        producer_config: RabbitMQProducerConfig {
            stream_name: "foo".to_string(),
        },
    });
    let input: InputSideChannelType = InputSideChannelType::RabbitMQ(InputRabbitMQConfig {
        client_options: ClientOptions::default(),
        consumer_config: RabbitMQConsumerConfig {
            stream_name: "foo".to_string(),
        },
    });
    let (producer_creator, consumer_creator) = channel_creator(Some(output), Some(input)).await?;

    let producer_creator = producer_creator.unwrap();
    let consumer_creator = consumer_creator.unwrap();

    let producer = producer_creator.create(Offset(0)).await?;
    let mut consumer = consumer_creator.create(Offset(0)).await?;
    println!("Created producer and consumer");

    println!("Sending message");
    producer
        .send(WriteOperation::CreateCollection {
            id: CollectionId("foo".to_string()),
            read_api_key: ApiKey(Secret::from("too".to_string())),
        })
        .await?;

    println!("Receiving message");
    let a = consumer.recv().await;
    println!("{:?}", a);

    println!("Done");

    sleep(Duration::from_secs(1)).await;

    Ok(())
}
