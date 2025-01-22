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

use super::{OperationSender, WriteOperation};

pub struct EmbeddingCalculationRequestInput {
    pub text: String,
    pub coll_id: CollectionId,
    pub doc_id: DocumentId,
    pub field_id: FieldId,
    pub op_sender: OperationSender,
}

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
