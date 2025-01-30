use std::{
    fs,
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    pin::Pin,
    time::Duration,
};

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use grpc_def::{ChatStreamResponse, Embedding, PlannedAnswerResponse};
use tempdir::TempDir;
use tokio::time::sleep;
use tokio_stream::Stream;
use tonic::{transport::Server, Response, Status};

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = TempDir::new("test").expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
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
