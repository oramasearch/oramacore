//! Retention checks against a real RabbitMQ stream. Ignored by default: run them against a
//! disposable broker with `rabbitmq_stream` enabled and a `retention-test` user (password
//! `retention-test`), with `ORAMA_RETENTION_TEST_BROKER_PORT` set to its stream port:
//! `cargo test replication_retention_broker -- --ignored`.

use super::*;

/// Creates a stream small enough that a few hundred KB of messages roll and drop segments.
async fn small_stream(prefix: &str) -> Result<(Environment, RabbitMQConsumerConfig)> {
    let port = std::env::var("ORAMA_RETENTION_TEST_BROKER_PORT")
        .context("ORAMA_RETENTION_TEST_BROKER_PORT is required")?
        .parse()?;
    let options = ClientOptions::builder()
        .host("127.0.0.1")
        .port(port)
        .user("retention-test")
        .password("retention-test")
        .load_balancer_mode(true)
        .build();
    let environment = Environment::from_client_option(options).await?;
    let stream = format!("{prefix}-{}", cuid2::create_id());
    environment
        .stream_creator()
        .max_length(ByteCapacity::KB(32))
        .max_segment_size(ByteCapacity::KB(16))
        .create(&stream)
        .await?;
    let config = RabbitMQConsumerConfig {
        stream_name: stream,
        consumer_name: "retention-test".into(),
    };
    Ok((environment, config))
}

async fn publish(environment: &Environment, stream: &str, count: usize) -> Result<()> {
    let producer = environment.producer().build(stream).await?;
    for _ in 0..count {
        producer
            .send_with_confirm(Message::builder().body(vec![b'x'; 8192]).build())
            .await?;
    }
    producer.close().await?;
    Ok(())
}

/// Waits until retention has removed the stream's first segment and returns the first
/// retained offset.
async fn wait_for_truncation(
    environment: &Environment,
    config: &RabbitMQConsumerConfig,
) -> Result<Offset> {
    timeout(Duration::from_secs(15), async {
        loop {
            if let Some(first) =
                probe_stream_edge(environment, config, OffsetSpecification::First, false).await?
            {
                if first.0 > 0 {
                    return Ok::<_, anyhow::Error>(first);
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .context("Retention did not truncate the stream")?
}

fn refusal(error: anyhow::Error) -> ReplicationRefusal {
    error
        .downcast::<ReplicationRefusal>()
        .expect("A replication refusal")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a disposable broker; set ORAMA_RETENTION_TEST_BROKER_PORT and run with --ignored"]
async fn real_broker_retention_refuses_empty_stale_ahead_and_clamped_replay() -> Result<()> {
    let (environment, config) = small_stream("replication-retention").await?;
    let result = async {
        // Empty stream: a reader that has applied nothing starts, any other is refused.
        assert_eq!(
            validate_stream_start(&environment, &config, Offset(0)).await?,
            None
        );
        assert_eq!(
            refusal(
                validate_stream_start(&environment, &config, Offset(1))
                    .await
                    .unwrap_err()
            ),
            ReplicationRefusal::EmptyStream { required: 1 }
        );

        publish(&environment, &config.stream_name, 64).await?;
        let first = wait_for_truncation(&environment, &config).await?;
        let head = probe_stream_edge(&environment, &config, OffsetSpecification::Last, true)
            .await?
            .context("head")?;

        assert_eq!(
            refusal(
                validate_stream_start(&environment, &config, Offset(0))
                    .await
                    .unwrap_err()
            ),
            ReplicationRefusal::HistoryExpired {
                required: 0,
                first: first.0
            },
            "a new reader cannot start on a truncated stream"
        );
        assert_eq!(
            refusal(
                validate_stream_start(&environment, &config, Offset(first.0 - 1))
                    .await
                    .unwrap_err()
            ),
            ReplicationRefusal::HistoryExpired {
                required: first.0 - 1,
                first: first.0
            }
        );
        assert_eq!(
            refusal(
                validate_stream_start(&environment, &config, head.next().next())
                    .await
                    .unwrap_err()
            ),
            ReplicationRefusal::AheadOfHead {
                required: head.0 + 2,
                head: head.0
            }
        );
        assert_eq!(
            validate_stream_start(&environment, &config, first).await?,
            Some(head)
        );
        assert_eq!(
            validate_stream_start(&environment, &config, head.next()).await?,
            Some(head)
        );

        // The broker clamps a consumer that asks for a removed offset to the first retained
        // one; the receiver refuses that delivery instead of applying it.
        let mut receiver = RabbitOperationReceiver {
            consumer: create_consumer(&environment, &config, Offset(0)).await?,
            environment: Box::new(environment.clone()),
            config: config.clone(),
            expected_offset: Offset(0),
            startup_head: Some(head),
        };
        let error = timeout(Duration::from_secs(5), receiver.next())
            .await?
            .context("delivery")?
            .unwrap_err();
        assert_eq!(
            refusal(error),
            ReplicationRefusal::UnexpectedDelivery {
                expected: 0,
                delivered: first.0
            }
        );

        // Recreating from a removed offset is refused before a consumer is created.
        assert!(matches!(
            refusal(receiver.recreate(Offset(0)).await.unwrap_err()),
            ReplicationRefusal::HistoryExpired { required: 0, .. }
        ));
        receiver.consumer.handle().close().await.ok();
        Ok::<_, anyhow::Error>(())
    }
    .await;
    environment.delete_stream(&config.stream_name).await?;
    result
}

/// Retention removes whole segments and never the active one, so a stream that has had a
/// message never looks empty. This is what makes accepting an empty stream at offset 0 safe.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a disposable broker; set ORAMA_RETENTION_TEST_BROKER_PORT and run with --ignored"]
async fn real_broker_retention_never_empties_a_stream() -> Result<()> {
    let (environment, config) = small_stream("replication-retention-roll").await?;
    let result = async {
        for _ in 0..4 {
            publish(&environment, &config.stream_name, 32).await?;
            let first = wait_for_truncation(&environment, &config).await?;
            assert!(first.0 > 0);
            let head = probe_stream_edge(&environment, &config, OffsetSpecification::Last, true)
                .await?
                .context("a truncated stream still has a head")?;
            assert!(head >= first);
        }
        // Give retention time to run after the last publish, then check again.
        sleep(Duration::from_secs(2)).await;
        let first = probe_stream_edge(&environment, &config, OffsetSpecification::First, false)
            .await?
            .context("the stream is not empty")?;
        assert!(first.0 > 0);
        Ok::<_, anyhow::Error>(())
    }
    .await;
    environment.delete_stream(&config.stream_name).await?;
    result
}
