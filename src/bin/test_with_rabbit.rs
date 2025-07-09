#![allow(dead_code)]

use std::time::Duration;

use anyhow::Result;

use nlp::locales::Locale;
use oramacore::{
    collection_manager::sides::{
        channel_creator, InputRabbitMQConfig, InputSideChannelType, Offset, OutputRabbitMQConfig,
        OutputSideChannelType, RabbitMQConsumerConfig, RabbitMQProducerConfig, WriteOperation,
    },
    types::{ApiKey, CollectionId},
};
use rabbitmq_stream_client::ClientOptions;
use tokio::time::sleep;
use tracing_subscriber::{fmt, layer::SubscriberExt, EnvFilter, Registry};

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = Registry::default().with(fmt::layer().compact().with_ansi(true));
    let subscriber = subscriber.with(EnvFilter::from_default_env());
    tracing::subscriber::set_global_default(subscriber).unwrap();

    main2().await?;

    Ok(())
}

async fn main2() -> Result<()> {
    let output: OutputSideChannelType = OutputSideChannelType::RabbitMQ(OutputRabbitMQConfig {
        client_options: ClientOptions::default(),
        producer_config: RabbitMQProducerConfig {
            stream_name: "foo".to_string(),
            producer_name: "producer".to_string(),
        },
    });
    let input: InputSideChannelType = InputSideChannelType::RabbitMQ(InputRabbitMQConfig {
        client_options: ClientOptions::default(),
        consumer_config: RabbitMQConsumerConfig {
            stream_name: "foo".to_string(),
            consumer_name: "consumer".to_string(),
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
            id: CollectionId::try_new("foo").unwrap(),
            read_api_key: ApiKey::try_new("too").unwrap(),
            default_locale: Locale::EN,
            description: None,
        })
        .await?;

    println!("Receiving message");
    let a = consumer.recv().await;
    println!("{a:?}");

    println!("Done");

    sleep(Duration::from_secs(1)).await;

    Ok(())
}
