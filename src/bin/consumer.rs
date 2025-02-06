use futures::StreamExt;
use rabbitmq_stream_client::error::StreamCreateError;
use rabbitmq_stream_client::types::{
    ByteCapacity, OffsetSpecification, ResponseCode, SuperStreamConsumer,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let name = args.get(1).cloned().unwrap_or("default".to_string());

    println!("Starting super stream consumer example with name: {}", name);

    use rabbitmq_stream_client::Environment;
    let environment = Environment::builder().build().await?;
    let super_stream = "hello-rust-super-stream";

    let create_response = environment
        .stream_creator()
        .max_length(ByteCapacity::GB(5))
        .create_super_stream(super_stream, 3, None)
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
    println!(
        "Super stream consumer example, consuming messages from the super stream {}",
        super_stream
    );
    let mut super_stream_consumer: SuperStreamConsumer = environment
        .super_stream_consumer()
        .offset(OffsetSpecification::Offset(3))
        .enable_single_active_consumer(true)
        .name(&name)
        .client_provided_name(&name)
        .consumer_update(move |active, message_context| async move {
            assert_eq!(active, 1, "Active should always be 1");
            let name = message_context.name();
            let stream = message_context.stream();
            let client = message_context.client();

            println!(
                "single active consumer: is active: {} on stream: {} with consumer_name: {}",
                active, stream, name
            );
            let stored_offset = client.query_offset(name, stream.as_str()).await;

            if let Err(_) = stored_offset {
                return OffsetSpecification::First;
            }
            let stored_offset_u = stored_offset.unwrap();
            println!("offset: {} stored", stored_offset_u.clone());
            OffsetSpecification::Offset(stored_offset_u)
        })
        .build(super_stream)
        .await
        .unwrap();

    while let Some(Ok(delivery)) = super_stream_consumer.next().await {
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

    println!("Stopping super stream consumer...");
    let _ = super_stream_consumer.handle().close().await;
    println!("Super stream consumer stopped");
    Ok(())
}
