use futures::StreamExt;
use rabbitmq_stream_client::error::StreamCreateError;
use rabbitmq_stream_client::types::{ByteCapacity, OffsetSpecification, ResponseCode};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let name = args.get(1).cloned().unwrap_or("default".to_string());

    use rabbitmq_stream_client::Environment;
    let environment = Environment::builder().build().await?;
    let stream = "oramacore-operations";
    let create_response = environment
        .stream_creator()
        .max_length(ByteCapacity::GB(5))
        .create(stream)
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

    let mut consumer = environment
        .consumer()
        .name(&name)
        .client_provided_name(&name)
        .name_optional(Some(name))
        .offset(OffsetSpecification::First)
        .build(stream)
        .await
        .unwrap();

    while let Some(Ok(delivery)) = consumer.next().await {
        println!(
            "Got message: {:#?} from stream: {} with offset: {}",
            delivery
                .message()
                .data()
                .map(|data| String::from_utf8(data.to_vec()).unwrap())
                .unwrap(),
            delivery.stream(),
            delivery.offset()
        );
    }

    let _ = consumer.handle().close().await;

    Ok(())
}
