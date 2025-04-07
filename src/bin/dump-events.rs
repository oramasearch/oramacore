use std::fs::OpenOptions;
use std::io::Write;

use anyhow::Context;
use futures::StreamExt;
use oramacore::collection_manager::sides::WriteOperation;
use rabbitmq_stream_client::types::OffsetSpecification;
use rabbitmq_stream_client::{ClientOptions, Environment};
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let client_options = ClientOptions::builder()
        .host("127.0.0.1")
        .port(5552)
        .user("oramacore")
        .password("guest")
        .v_host("/")
        .build();
    let environment = Environment::from_client_option(client_options)
        .await
        .context("Cannot create environment")?;

    let consumer_name = "dump-events".to_string();
    let stream_name = "oramacore-operations".to_string();

    let mut consumer = environment
        .consumer()
        .client_provided_name(&consumer_name)
        .offset(OffsetSpecification::First)
        .build(&stream_name)
        .await
        .context("Failed to create rabbit consumer")?;
    info!("Created rabbitmq consumer {}", &stream_name);

    let fs = OpenOptions::new()
        .create(true)
        .append(true)
        .open("dump-rabbit-events.log")
        .context("Failed to open events.log")?;
    let mut writer = std::io::BufWriter::new(fs);
    let mut i = 0;
    while let Some(Ok(delivery)) = consumer.next().await {
        let a = delivery.offset();
        println!("Got message with offset {}", a);

        if a > 30_000 {
            break;
        }

        let data = delivery.message().data().unwrap();
        let message_body: WriteOperation = bincode::deserialize(data).unwrap();

        let coll_id = match &message_body {
            WriteOperation::KV(_) => continue,
            WriteOperation::SubstituteCollection { .. } => continue,
            WriteOperation::DeleteCollection(collection_id) => collection_id,
            WriteOperation::CreateCollection { id, .. } => id,
            WriteOperation::Collection(colk_id, _) => colk_id,
        };
        if coll_id.0.as_str() != "my-coll" {
            continue;
        }

        serde_json::to_writer(&mut writer, &message_body)
            .context("Failed to serialize message body")?;
        writer.write(b"\n").context("Failed to write newline")?;
        i += 1;

        writer.flush().context("Failed to flush writer")?;

        if a > 30_000 {
            break;
        }
    }

    writer.flush().context("Failed to flush writer")?;

    Ok(())
}
