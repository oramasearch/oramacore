use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use tokio::sync::mpsc::Receiver;
use tracing::{debug, info};

use crate::{
    ai::{AIService, OramaModel},
    collection_manager::{
        dto::FieldId,
        sides::{CollectionWriteOperation, DocumentFieldIndexOperation},
    },
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
    pub model: OramaModel,
    pub input: EmbeddingCalculationRequestInput,
}

async fn process<I>(ai_service: Arc<AIService>, cache: I) -> Result<()>
where
    I: Iterator<Item = (OramaModel, Vec<EmbeddingCalculationRequestInput>)>,
{
    info!("Process embedding batch");

    for (model, inputs) in cache {
        let model_name = model.as_str_name();
        info!(model_name = ?model_name, inputs = %inputs.len(), "Process embedding batch");

        let metric = EMBEDDING_CALCULATION_METRIC.create(EmbeddingCalculationLabels {
            model: model_name.to_string(),
        });

        let text_inputs: Vec<&String> = inputs.iter().map(|input| &input.text).collect();
        let output = ai_service
            .embed_passage(model, text_inputs)
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
    ai_service: Arc<AIService>,
    mut receiver: Receiver<EmbeddingCalculationRequest>,
    limit: usize,
) {
    // `limit` is the number of items to process in a batch
    assert!(limit > 0);

    tokio::task::spawn(async move {
        let mut buffer = Vec::with_capacity(limit);

        let mut cache: HashMap<OramaModel, Vec<EmbeddingCalculationRequestInput>> =
            Default::default();

        loop {
            // `recv_many` waits for at least one available item
            let item_count = receiver.recv_many(&mut buffer, limit).await;
            // `recv_many` returns 0 if the channel is closed
            if item_count == 0 {
                break;
            }

            for item in buffer.drain(..) {
                let EmbeddingCalculationRequest { model, input } = item;

                let inputs = cache.entry(model).or_default();
                inputs.push(input);
            }

            process(ai_service.clone(), cache.drain()).await.unwrap();
        }
    });
}
