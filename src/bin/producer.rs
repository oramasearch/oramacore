use std::time::Duration;

use anyhow::Result;

use rabbitmq_stream_client::{
    error::StreamCreateError,
    types::{
        ByteCapacity, HashRoutingMurmurStrategy, Message, ResponseCode,
        RoutingStrategy,
    },
};
use tokio::time::sleep;

fn hash_strategy_value_extractor(message: &Message) -> String {
    message
        .application_properties()
        .unwrap()
        .get("id")
        .unwrap()
        .clone()
        .try_into()
        .unwrap()
}

#[tokio::main]
async fn main() -> Result<()> {
    use rabbitmq_stream_client::Environment;
    let environment = Environment::builder().build().await?;
    let super_stream = "hello-rust-super-stream";

    /*
    let delete_stream = environment.delete_super_stream(super_stream).await;
    match delete_stream {
        Ok(_) => {
            println!("Successfully deleted super stream {}", super_stream);
        }
        Err(err) => {
            println!(
                "Failed to delete super stream {}. error {}",
                super_stream, err
            );
        }
    }
    */

    let create_response = environment
        .stream_creator()
        .max_length(ByteCapacity::GB(5))
        .create_super_stream(super_stream, 3, None)
        .await;
    match create_response {
        Err(StreamCreateError::Create { status: ResponseCode::StreamAlreadyExists, .. }) => {}
        Err(e) => {
            println!("Error creating stream: {:?}", e);
        }
        Ok(_) => {}
    }

    let mut super_stream_producer = environment
        .super_stream_producer(RoutingStrategy::HashRoutingStrategy(
            HashRoutingMurmurStrategy {
                routing_extractor: &hash_strategy_value_extractor,
            },
        ))
        .client_provided_name("my super stream producer for hello rust")
        .build(super_stream)
        .await
        .unwrap();

    let message = Message::builder()
        .body("super stream message_0".to_string())
        .application_properties()
        .insert("id", 0.to_string())
        .message_builder()
        .build();
    println!("Sending message {:?}...", message);
    let r = super_stream_producer
        .send(message, move |r| async move {
            println!("AA - Message sent {:?}", r);
        })
        .await;
    println!("Message sent {:?}", r);

    sleep(Duration::from_secs(1)).await;

    let _ = super_stream_producer.close().await;

    println!("Done");

    Ok(())
}
