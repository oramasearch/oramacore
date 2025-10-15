use std::{collections::HashMap, sync::Arc};

use tokio::sync::mpsc::Receiver;
use tracing::{error, info, trace, warn};

use crate::{
    collection_manager::sides::{
        CollectionWriteOperation, IndexWriteOperation, OperationSender, WriteOperation,
    },
    python::embeddings::{EmbeddingsService, Intent, Model},
    types::{CollectionId, DocumentId, FieldId, IndexId},
};

pub struct EmbeddingCalculationRequestInput {
    pub text: String,
    pub doc_id: DocumentId,
    pub field_id: FieldId,
}

#[derive(Debug)]
pub struct MultiEmbeddingCalculationRequest {
    pub model: Model,
    pub coll_id: CollectionId,
    pub doc_id: DocumentId,
    pub field_id: FieldId,
    pub index_id: IndexId,
    pub text: Vec<String>,
}

async fn process<I>(
    op_sender: &OperationSender,
    embeddings_service: Arc<EmbeddingsService>,
    cache: I,
) where
    I: Iterator<
        Item = (
            Model,
            HashMap<(CollectionId, IndexId), Vec<EmbeddingCalculationRequestInput>>,
        ),
    >,
{
    info!("Processing embedding batch");

    for (model, inputs_per_collection_index) in cache {
        let model_name = model.clone().to_string();

        let mut res: HashMap<
            (CollectionId, IndexId),
            HashMap<FieldId, HashMap<DocumentId, Vec<Vec<f32>>>>,
        > = HashMap::new();
        for ((collection_id, index_id), inputs) in inputs_per_collection_index {
            trace!(model_name = ?model_name, inputs = ?inputs.len(), "Process embedding batch");
            let text_inputs: Vec<String> =
                inputs.iter().map(|input| input.text.to_string()).collect(); // @todo: to_string() here is used to clone the string, check if we can remove this cloning

            // If something goes wrong, we will just log it and continue
            // We should put a circuit breaker here like https://docs.rs/tokio-retry2/latest/tokio_retry2/
            // TODO: Add circuit breaker
            let _ = match embeddings_service.calculate_embeddings(
                text_inputs,
                Some(Intent::Passage),
                model.clone(),
            ) {
                Ok(embeddings) => {
                    let output = inputs.into_iter().zip(embeddings.into_iter());

                    for (req, embeddings) in output {
                        let EmbeddingCalculationRequestInput {
                            doc_id, field_id, ..
                        } = req;
                        let entry = res.entry((collection_id, index_id)).or_default();
                        let entry = entry.entry(field_id).or_default();
                        let entry = entry.entry(doc_id).or_default();
                        entry.push(embeddings);
                    }

                    Ok(())
                }
                Err(e) => {
                    warn!("Failed to calculate embeddings: {:?}", e);
                    Err(e)
                }
            };
        }

        let ops = res
            .into_iter()
            .map(|((collection_id, index_id), data)| {
                WriteOperation::Collection(
                    collection_id,
                    CollectionWriteOperation::IndexWriteOperation(
                        index_id,
                        IndexWriteOperation::IndexEmbedding {
                            data: data
                                .into_iter()
                                .map(|(field_id, data)| {
                                    let data: Vec<_> = data.into_iter().collect();
                                    (field_id, data)
                                })
                                .collect(),
                        },
                    ),
                )
            })
            .collect::<Vec<_>>();
        if let Err(e) = op_sender.send_batch(ops).await {
            error!(error = ?e, "Failed to send embedding batch");
        }
    }

    info!("Embedding batch processed");
}

pub fn start_calculate_embedding_loop(
    embeddings_service: Arc<EmbeddingsService>,
    mut receiver: Receiver<MultiEmbeddingCalculationRequest>,
    op_sender: OperationSender,
    limit: u32,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    // `limit` is the number of items to process in a batch
    assert!(limit > 0);

    tokio::task::spawn(async move {
        info!("Starting embedding calculation loop...");

        let mut buffer = Vec::with_capacity(limit as usize);
        let mut cache: HashMap<
            Model,
            HashMap<(CollectionId, IndexId), Vec<EmbeddingCalculationRequestInput>>,
        > = Default::default();

        'outer: loop {
            // `recv_many` waits for at least one available item
            let item_count = tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Stopping operation receiver");
                    break 'outer;
                }
                op = receiver.recv_many(&mut buffer, limit as usize) => {
                    op
                }
            };

            // let item_count = receiver.recv_many(&mut buffer, limit as usize).await;
            // `recv_many` returns 0 if the channel is closed
            if item_count == 0 {
                warn!("Embedding calculation receiver closed");
                break;
            }

            for item in buffer.drain(..) {
                let MultiEmbeddingCalculationRequest {
                    model,
                    coll_id,
                    doc_id,
                    field_id,
                    index_id,
                    text,
                } = item;

                let inputs = cache.entry(model).or_default();
                let inputs = inputs.entry((coll_id, index_id)).or_default();
                for t in text {
                    let input = EmbeddingCalculationRequestInput {
                        text: t,
                        doc_id,
                        field_id,
                    };
                    inputs.push(input);
                }
            }

            process(&op_sender, embeddings_service.clone(), cache.drain()).await;
        }

        loop {
            let item = match receiver.try_recv() {
                Ok(item) => item,
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                    break;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    warn!("Embedding calculation receiver closed");
                    break;
                }
            };

            let MultiEmbeddingCalculationRequest {
                model,
                coll_id,
                doc_id,
                field_id,
                index_id,
                text,
            } = item;

            let inputs = cache.entry(model).or_default();
            let inputs = inputs.entry((coll_id, index_id)).or_default();
            for t in text {
                let input = EmbeddingCalculationRequestInput {
                    text: t,
                    doc_id,
                    field_id,
                };
                inputs.push(input);
            }
        }

        process(&op_sender, embeddings_service.clone(), cache.drain()).await;

        warn!("Stop embedding calculation loop");

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e , "Failed to send stop signal");
        }
    });
}
