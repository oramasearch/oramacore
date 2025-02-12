use std::{
    sync::{atomic::AtomicU32, Arc},
    time::Duration,
};

use anyhow::Result;

use rabbitmq_stream_client::{
    error::StreamCreateError,
    types::{ByteCapacity, Message, ResponseCode},
};
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    use rabbitmq_stream_client::Environment;

    let environment = Environment::builder().build().await?;
    let stream = "oramacore-operations";
    let create_response = environment
        .stream_creator()
        .max_length(ByteCapacity::GB(5))
        .create(&stream)
        .await;
    if let Err(e) = create_response {
        if let StreamCreateError::Create { stream, status } = e {
            match status {
                // we can ignore this error because the stream already exists
                ResponseCode::StreamAlreadyExists => {}
                err => {
                    println!("Error creating stream: {:?} {:?}", stream, err);
                }
            }
        }
    }

    let producer = environment
        .producer()
        .client_provided_name("producer")
        .build(stream)
        .await
        .unwrap();

    let msg = Message::builder()
        .body(format!("stream message_{}", 0))
        .build();
    producer
        .send(msg, move |confirmation_status| async move {
            println!("Confirmation status: {:?}", confirmation_status);
        })
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    let _ = producer.close().await;

    println!("Done");

    Ok(())
}
