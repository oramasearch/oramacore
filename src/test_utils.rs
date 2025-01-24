use std::{
    fs,
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use grpc_def::{ChatStreamResponse, Embedding, PlannedAnswerResponse};
use tempdir::TempDir;
use tokio::time::sleep;
use tokio_stream::Stream;
use tonic::{transport::Server, Response, Status};

use crate::{
    collection_manager::{
        dto::{FieldId, LanguageDTO, TypedField},
        sides::{
            channel, CollectionWriteOperation, DocumentFieldIndexOperation, StringField,
            WriteOperation,
        },
    },
    indexes::string::{
        CommittedStringFieldIndex, StringIndex, StringIndexConfig, UncommittedStringFieldIndex,
    },
    nlp::TextParser,
    types::{CollectionId, Document, DocumentId},
};

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

pub async fn create_string_index(
    fields: Vec<(FieldId, String)>,
    documents: Vec<Document>,
) -> Result<StringIndex> {
    let index = StringIndex::new(StringIndexConfig {});

    let (sx, mut rx) = channel(1_0000);

    let collection_id = CollectionId("collection".to_string());

    let mut string_fields = Vec::new();
    for (field_id, field_name) in fields {
        sx.send(WriteOperation::Collection(
            collection_id.clone(),
            CollectionWriteOperation::CreateField {
                field_id,
                field_name: field_name.clone(),
                field: TypedField::Text(LanguageDTO::English),
            },
        ))
        .await
        .unwrap();

        string_fields.push(StringField::new(
            Arc::new(TextParser::from_locale(crate::nlp::locales::Locale::EN)),
            collection_id.clone(),
            field_id,
            field_name,
        ))
    }

    for (id, doc) in documents.into_iter().enumerate() {
        let document_id = DocumentId(id as u64);
        let flatten = doc.into_flatten();

        for string_field in &string_fields {
            string_field
                .get_write_operations(document_id, &flatten, sx.clone())
                .await
                .unwrap()
        }
    }

    drop(sx);

    while let Some((offset, operation)) = rx.recv().await {
        match operation {
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::CreateField { field_id, .. },
            ) => {
                index.add_field(offset, field_id);
            }
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Index(
                    doc_id,
                    field_id,
                    DocumentFieldIndexOperation::IndexString {
                        field_length,
                        terms,
                    },
                ),
            ) => {
                index
                    .insert(offset, doc_id, field_id, field_length, terms)
                    .await?;
            }
            _ => unreachable!(),
        };
    }

    Ok(index)
}

pub async fn create_uncommitted_string_field_index(
    documents: Vec<Document>,
) -> Result<UncommittedStringFieldIndex> {
    create_uncommitted_string_field_index_from(documents, 0).await
}

pub async fn create_uncommitted_string_field_index_from(
    documents: Vec<Document>,
    starting_doc_id: u64,
) -> Result<UncommittedStringFieldIndex> {
    let collection_id = CollectionId("collection".to_string());
    let field_name = "field".to_string();
    let field_id = FieldId(0);

    let string_field = StringField::new(
        Arc::new(TextParser::from_locale(crate::nlp::locales::Locale::EN)),
        collection_id.clone(),
        field_id,
        field_name.clone(),
    );

    let (sx, mut rx) = channel(1_0000);

    sx.send(WriteOperation::Collection(
        collection_id.clone(),
        CollectionWriteOperation::CreateField {
            field_id,
            field_name: field_name.clone(),
            field: TypedField::Text(LanguageDTO::English),
        },
    ))
    .await?;

    for (id, doc) in documents.into_iter().enumerate() {
        let document_id = DocumentId(starting_doc_id + id as u64);
        let flatten = doc.into_flatten();
        string_field
            .get_write_operations(document_id, &flatten, sx.clone())
            .await
            .with_context(|| {
                format!("Test get_write_operations {:?} {:?}", document_id, flatten)
            })?;
    }

    drop(sx);

    let mut index = None;
    while let Some((offset, operation)) = rx.recv().await {
        match operation {
            WriteOperation::Collection(_, CollectionWriteOperation::CreateField { .. }) => {
                index = Some(UncommittedStringFieldIndex::new(offset));
            }
            WriteOperation::Collection(
                _,
                CollectionWriteOperation::Index(
                    document_id,
                    _,
                    DocumentFieldIndexOperation::IndexString {
                        field_length,
                        terms,
                    },
                ),
            ) => {
                index
                    .as_ref()
                    .unwrap()
                    .insert(offset, document_id, field_length, terms)
                    .await
                    .with_context(|| {
                        format!("test cannot insert index_string {:?}", document_id)
                    })?;
            }
            _ => unreachable!(),
        };
    }

    Ok(index.unwrap())
}

pub async fn create_committed_string_field_index(
    documents: Vec<Document>,
) -> Result<Option<CommittedStringFieldIndex>> {
    let index = create_string_index(vec![(FieldId(1), "field".to_string())], documents).await?;

    index.commit(generate_new_path()).await?;

    Ok(index.remove_committed_field(FieldId(1)))
}

pub mod grpc_def {
    tonic::include_proto!("orama_ai_service");
}

pub struct GRPCServer {
    fastembed_model: TextEmbedding,
}

type EchoResult<T> = Result<Response<T>, Status>;
type ResponseStream = Pin<Box<dyn Stream<Item = Result<ChatStreamResponse, Status>> + Send>>;
type PlannedAnswerResponseStream =
    Pin<Box<dyn Stream<Item = Result<PlannedAnswerResponse, Status>> + Send>>;

#[tonic::async_trait]
impl grpc_def::llm_service_server::LlmService for GRPCServer {
    type ChatStreamStream = ResponseStream;
    type PlannedAnswerStream = PlannedAnswerResponseStream;

    async fn check_health(
        &self,
        _req: tonic::Request<grpc_def::HealthCheckRequest>,
    ) -> Result<tonic::Response<grpc_def::HealthCheckResponse>, Status> {
        Ok(tonic::Response::new(grpc_def::HealthCheckResponse {
            status: "ok".to_string(),
        }))
    }

    async fn get_embedding(
        &self,
        req: tonic::Request<grpc_def::EmbeddingRequest>,
    ) -> Result<tonic::Response<grpc_def::EmbeddingResponse>, Status> {
        let req = req.into_inner();
        // `0` means `BgeSmall`
        if req.model != 0 {
            return Err(Status::invalid_argument("Invalid model"));
        }

        let embed = self
            .fastembed_model
            .embed(req.input, None)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(tonic::Response::new(grpc_def::EmbeddingResponse {
            embeddings_result: embed
                .into_iter()
                .map(|v| Embedding { embeddings: v })
                .collect(),
            dimensions: 384,
        }))
    }

    async fn chat(
        &self,
        _req: tonic::Request<grpc_def::ChatRequest>,
    ) -> Result<tonic::Response<grpc_def::ChatResponse>, Status> {
        todo!()
    }

    async fn chat_stream(
        &self,
        _req: tonic::Request<grpc_def::ChatRequest>,
    ) -> EchoResult<Self::ChatStreamStream> {
        todo!()
    }

    async fn planned_answer(
        &self,
        _req: tonic::Request<grpc_def::PlannedAnswerRequest>,
    ) -> EchoResult<Self::PlannedAnswerStream> {
        todo!()
    }
}
pub async fn create_grpc_server() -> Result<SocketAddr> {
    let model = EmbeddingModel::BGESmallENV15;

    let init_option = InitOptions::new(model.clone())
        .with_cache_dir(std::env::temp_dir())
        .with_show_download_progress(false);

    let text_embedding = TextEmbedding::try_new(init_option)
        .with_context(|| format!("Failed to initialize the Fastembed: {model}"))?;

    let server = GRPCServer {
        fastembed_model: text_embedding,
    };

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let add = listener.local_addr().unwrap();
    drop(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(grpc_def::llm_service_server::LlmServiceServer::new(server))
            .serve(add)
            .await
            .unwrap();
    });

    // Waiting for the server to start
    loop {
        let c = grpc_def::llm_service_client::LlmServiceClient::connect(format!("http://{}", add))
            .await;
        if c.is_ok() {
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }

    Ok(add)
}
