use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use tokio::sync::mpsc::Receiver;
use tracing::{debug, info};

use crate::{
    collection_manager::{
        dto::FieldId,
        sides::{CollectionWriteOperation, DocumentFieldIndexOperation},
    },
    embeddings::EmbeddingService,
    metrics::{EmbeddingCalculationLabels, EMBEDDING_CALCULATION_METRIC},
    types::{CollectionId, DocumentId},
};

use super::WriteOperation;

#[derive(Debug)]
pub struct EmbeddingCalculationRequestInput {
    pub text: String,
    pub coll_id: CollectionId,
    pub doc_id: DocumentId,
    pub field_id: FieldId,
    pub op_sender: tokio::sync::broadcast::Sender<WriteOperation>,
}
#[derive(Debug)]
pub struct EmbeddingCalculationRequest {
    pub model_name: String,
    pub input: EmbeddingCalculationRequestInput,
}

async fn process<I>(embedding_server: Arc<EmbeddingService>, cache: I) -> Result<()>
where
    I: Iterator<Item = (String, Vec<EmbeddingCalculationRequestInput>)>,
{
    info!("Process embedding batch");

    for (model_name, inputs) in cache {
        info!(model_name = ?model_name, inputs = %inputs.len(), "Process embedding batch");

        let metric = EMBEDDING_CALCULATION_METRIC.create(EmbeddingCalculationLabels {
            model: model_name.clone(),
        });

        let model = embedding_server.get_model(model_name).await.unwrap();
        let text_inputs: Vec<&String> = inputs.iter().map(|input| &input.text).collect();

        let output = model
            .embed_passage(text_inputs)
            .await
            .context("Failed to embed text")?;

        info!("Embedding done");

        drop(metric);

        for (input, output) in inputs.into_iter().zip(output.into_iter()) {
            let EmbeddingCalculationRequestInput {
                doc_id,
                coll_id,
                field_id,
                op_sender,
                ..
            } = input;

            op_sender
                .send(WriteOperation::Collection(
                    coll_id,
                    CollectionWriteOperation::Index(
                        doc_id,
                        field_id,
                        DocumentFieldIndexOperation::IndexEmbedding { value: output },
                    ),
                ))
                .unwrap();
        }

        info!("Embedding sent to the read side");
    }

    debug!("Embedding batch processed");

    Ok(())
}

pub fn start_calculate_embedding_loop(
    embedding_server: Arc<EmbeddingService>,
    mut receiver: Receiver<EmbeddingCalculationRequest>,
    limit: usize,
) {
    // `limit` is the number of items to process in a batch
    assert!(limit > 0);

    tokio::task::spawn(async move {
        let mut buffer = Vec::with_capacity(limit);

        let mut cache: HashMap<String, Vec<EmbeddingCalculationRequestInput>> = Default::default();

        loop {
            // `recv_many` waits for at least one available item
            let item_count = receiver.recv_many(&mut buffer, limit).await;
            // `recv_many` returns 0 if the channel is closed
            if item_count == 0 {
                break;
            }

            for item in buffer.drain(..) {
                let EmbeddingCalculationRequest { model_name, input } = item;

                let inputs = cache.entry(model_name).or_default();
                inputs.push(input);
            }

            process(embedding_server.clone(), cache.drain())
                .await
                .unwrap();
        }
    });
}

#[cfg(test)]
mod tests {
    use crate::ai::grpc::{GrpcModelConfig, GrpcRepoConfig};

    #[cfg(feature = "test-python")]
    #[tokio::test]
    async fn test_embedding_grpc_server() -> anyhow::Result<()> {
        use std::{collections::HashMap, sync::Arc};

        use crate::embeddings::{EmbeddingConfig, ModelConfig};

        use super::*;

        let (sx, rx) = tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(1);

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
        let embedding_service = EmbeddingService::try_new(embedding_config)
            .await
            .expect("Failed to initialize the EmbeddingService");
        let embedding_service = Arc::new(embedding_service);

        let (sender, mut receiver) = tokio::sync::broadcast::channel(100);

        start_calculate_embedding_loop(embedding_service.clone(), rx, 10);

        sx.send(EmbeddingCalculationRequest {
            model_name: "my-model".to_string(),
            input: EmbeddingCalculationRequestInput {
                text: "foo".to_string(),
                coll_id: CollectionId("my-collection".to_string()),
                doc_id: DocumentId(1),
                field_id: FieldId(1),
                op_sender: sender.clone(),
            },
        })
        .await
        .unwrap();
        sx.send(EmbeddingCalculationRequest {
            model_name: "my-model".to_string(),
            input: EmbeddingCalculationRequestInput {
                text: "bar".to_string(),
                coll_id: CollectionId("my-collection".to_string()),
                doc_id: DocumentId(2),
                field_id: FieldId(1),
                op_sender: sender.clone(),
            },
        })
        .await
        .unwrap();
        sx.send(EmbeddingCalculationRequest {
            model_name: "my-model".to_string(),
            input: EmbeddingCalculationRequestInput {
                text: "baz".to_string(),
                coll_id: CollectionId("my-collection".to_string()),
                doc_id: DocumentId(3),
                field_id: FieldId(1),
                op_sender: sender.clone(),
            },
        })
        .await
        .unwrap();

        let a = receiver.recv().await.unwrap();
        let b = receiver.recv().await.unwrap();
        let c = receiver.recv().await.unwrap();

        assert!(matches!(
            a,
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Index(
                    DocumentId(1),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexEmbedding { value: _ }
                )
            )
        ));
        assert!(matches!(
            b,
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Index(
                    DocumentId(2),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexEmbedding { value: _ }
                )
            )
        ));
        assert!(matches!(
            c,
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Index(
                    DocumentId(3),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexEmbedding { value: _ }
                )
            )
        ));

        sx.send(EmbeddingCalculationRequest {
            model_name: "my-model".to_string(),
            input: EmbeddingCalculationRequestInput {
                text: "baz".to_string(),
                coll_id: CollectionId("my-collection".to_string()),
                doc_id: DocumentId(4),
                field_id: FieldId(1),
                op_sender: sender.clone(),
            },
        })
        .await
        .unwrap();

        let d = receiver.recv().await.unwrap();
        assert!(matches!(
            d,
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Index(
                    DocumentId(4),
                    FieldId(1),
                    DocumentFieldIndexOperation::IndexEmbedding { value: _ }
                )
            )
        ));

        Ok(())
    }
}
