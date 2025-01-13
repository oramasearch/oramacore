mod collection;
mod collections;
mod fields;
mod operation;

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context, Result};
pub use collections::{CollectionsWriter, CollectionsWriterConfig};
pub use operation::*;

#[cfg(any(test, feature = "benchmarking"))]
pub use fields::*;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

use crate::embeddings::EmbeddingService;

#[derive(Debug)]
pub struct RequestInput {
    pub text: String,
}
#[derive(Debug)]
pub struct Request {
    pub model_name: String,
    pub input: RequestInput,
}

async fn process<I>(embedding_server: Arc<EmbeddingService>, cache: I) -> Result<()>
where
    I: Iterator<Item = (String, Vec<RequestInput>)>,
{
    println!("Processing cache");
    for (model_name, inputs) in cache {
        println!("Model: {}", model_name);
        let model = embedding_server.get_model(model_name).await.unwrap();
        let inputs = inputs.into_iter().map(|input| input.text).collect();

        println!("Inputs: {:?}", inputs);
        let output = model.embed(inputs).await.context("Failed to embed text")?;

        println!("Output: {:?}", output);
    }

    Ok(())
}

pub fn start_calculate_embedding_loop(
    embedding_server: Arc<EmbeddingService>,
    timeout: Duration,
) -> tokio::sync::mpsc::Sender<Request> {
    let (sx, rx) = tokio::sync::mpsc::channel::<Request>(1);

    tokio::task::spawn(async move {
        let rx = ReceiverStream::new(rx);
        let rx = rx.timeout(timeout);
        tokio::pin!(rx);

        let mut cache: HashMap<String, Vec<RequestInput>> = Default::default();

        loop {
            use std::result::Result::Ok;

            let item = rx.try_next().await;

            println!("Item: {:?}", item);

            match item {
                Ok(None) => {
                    println!("None");
                }
                Ok(Some(Request { model_name, input })) => {
                    println!("Some");

                    let inputs = cache.entry(model_name).or_default();
                    inputs.push(input);

                    if inputs.len() < 10 {
                        continue;
                    }

                    process(embedding_server.clone(), cache.drain())
                        .await
                        .unwrap();
                }
                Err(e) => {
                    println!("Error: {:?}", e);
                    // timeout

                    process(embedding_server.clone(), cache.drain())
                        .await
                        .unwrap();
                }
            }
        }
    });

    sx
}

#[cfg(test)]
mod tests {
    use core::panic;
    use std::{collections::HashMap, sync::Arc};

    use anyhow::{Context, Result};
    use serde_json::json;
    use tokio::time::sleep;

    use crate::{
        collection_manager::dto::CreateCollectionOptionDTO,
        embeddings::{
            grpc::{GrpcModelConfig, GrpcRepoConfig},
            EmbeddingConfig, EmbeddingService, ModelConfig,
        },
        test_utils::generate_new_path,
        types::CollectionId,
    };

    use super::*;

    #[tokio::test]
    async fn test_side_writer_serialize() -> Result<()> {
        let embedding_config = EmbeddingConfig {
            preload: vec![],
            grpc: None,
            hugging_face: None,
            fastembed: None,
            models: HashMap::new(),
        };
        let embedding_service = EmbeddingService::try_new(embedding_config)
            .await
            .with_context(|| "Failed to initialize the EmbeddingService")?;
        let embedding_service = Arc::new(embedding_service);

        let config = CollectionsWriterConfig {
            data_dir: generate_new_path(),
        };

        let collection = {
            let (sender, receiver) = tokio::sync::broadcast::channel(1_0000);

            let writer = CollectionsWriter::new(sender, embedding_service.clone(), config.clone());

            let collection_id = CollectionId("test-collection".to_string());
            writer
                .create_collection(CreateCollectionOptionDTO {
                    id: collection_id.0.clone(),
                    description: None,
                    language: None,
                    typed_fields: Default::default(),
                })
                .await?;

            writer
                .write(
                    collection_id,
                    vec![
                        json!({
                            "name": "John Doe",
                        }),
                        json!({
                            "name": "Jane Doe",
                        }),
                    ]
                    .try_into()?,
                )
                .await?;

            let collections = writer.list().await;

            writer.commit().await?;

            drop(receiver);

            collections
        };

        let after = {
            let (sender, receiver) = tokio::sync::broadcast::channel(1_0000);

            let mut writer = CollectionsWriter::new(sender, embedding_service.clone(), config);

            writer
                .load()
                .await
                .context("Cannot load collections writer")?;

            let collections = writer.list().await;

            drop(receiver);

            collections
        };

        assert_eq!(collection, after);

        Ok(())
    }

    #[tokio::test]
    async fn test_embedding_grpc_server() -> Result<()> {
        let embedding_config = EmbeddingConfig {
            preload: vec![],
            grpc: Some(GrpcRepoConfig {
                host: "127.0.0.1".parse().unwrap(),
                port: 50051,
                api_key: None,
            }),
            hugging_face: None,
            fastembed: None,
            models: HashMap::from_iter([(
                "my-model".to_string(),
                ModelConfig::Grpc(GrpcModelConfig {
                    real_model_name: "BGESmall".to_string(),
                    dimensions: 384,
                }),
            )]),
        };
        println!("EmbeddingConfig: {:?}", embedding_config);
        let embedding_service = EmbeddingService::try_new(embedding_config)
            .await
            .expect("Failed to initialize the EmbeddingService");
        println!("EmbeddingService: {:?}", embedding_service);
        let embedding_service = Arc::new(embedding_service);

        let sx = start_calculate_embedding_loop(embedding_service.clone(), Duration::from_secs(2));

        sx.send(Request {
            model_name: "my-model".to_string(),
            input: RequestInput {
                text: "foo".to_string(),
            },
        })
        .await
        .unwrap();
        sx.send(Request {
            model_name: "my-model".to_string(),
            input: RequestInput {
                text: "bar".to_string(),
            },
        })
        .await
        .unwrap();
        sx.send(Request {
            model_name: "my-model".to_string(),
            input: RequestInput {
                text: "baz".to_string(),
            },
        })
        .await
        .unwrap();

        sleep(Duration::from_secs(10_000)).await;

        panic!("This test should not reach here");

        Ok(())
    }
}
