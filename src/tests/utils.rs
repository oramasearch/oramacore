use std::{
    fs,
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result};
use duration_string::DurationString;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use grpc_def::Embedding;
use http::uri::Scheme;
use redact::Secret;
use serde_json::json;
use std::str::FromStr;
use tokio::time::sleep;
use tonic::{transport::Server, Response, Status};
use tracing::info;

use crate::{
    ai::AIServiceConfig,
    build_orama,
    collection_manager::{
        dto::ApiKey,
        sides::{
            hooks::{HooksRuntimeConfig, SelectEmbeddingsPropertiesHooksRuntimeConfig},
            CollectionsWriterConfig, IndexesConfig, InputSideChannelType, OramaModelSerializable,
            OutputSideChannelType, ReadSide, ReadSideConfig, WriteSide, WriteSideConfig,
        },
    },
    types::{CollectionId, DocumentList},
    web_server::HttpConfig,
    OramacoreConfig,
};

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    println!("Temp dir: {:?}", tmp_dir.path());
    let dir = tmp_dir.path().to_path_buf();
    fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

pub fn hooks_runtime_config() -> HooksRuntimeConfig {
    HooksRuntimeConfig {
        select_embeddings_properties: SelectEmbeddingsPropertiesHooksRuntimeConfig {
            check_interval: DurationString::from_str("1s").unwrap(),
            max_idle_time: DurationString::from_str("1s").unwrap(),
            instances_count_per_code: 1,
            queue_capacity: 1,
            max_execution_time: DurationString::from_str("1s").unwrap(),
            max_startup_time: DurationString::from_str("1s").unwrap(),
        },
    }
}

pub async fn create(mut config: OramacoreConfig) -> Result<(Arc<WriteSide>, Arc<ReadSide>)> {
    if config.ai_server.port == 0 {
        let address = create_grpc_server().await?;
        config.ai_server.host = address.ip();
        config.ai_server.port = address.port();
        info!("AI server started on {}", address);
    }

    let (write_side, read_side) = build_orama(config).await?;

    let write_side = write_side.unwrap();
    let read_side = read_side.unwrap();

    Ok((write_side, read_side))
}

pub async fn create_collection(
    write_side: Arc<WriteSide>,
    collection_id: CollectionId,
) -> Result<()> {
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;
    sleep(Duration::from_millis(100)).await;

    Ok(())
}

pub async fn insert_docs<I>(
    write_side: Arc<WriteSide>,
    write_api_key: ApiKey,
    collection_id: CollectionId,
    docs: I,
) -> Result<()>
where
    I: IntoIterator<Item = serde_json::Value>,
{
    let document_list: Vec<serde_json::value::Value> = docs.into_iter().collect();
    let document_list: DocumentList = document_list.try_into()?;

    write_side
        .insert_documents(write_api_key, collection_id, document_list)
        .await?;

    sleep(Duration::from_millis(1_000)).await;

    Ok(())
}

pub mod grpc_def {
    tonic::include_proto!("orama_ai_service");
}

pub struct GRPCServer {
    fastembed_model: TextEmbedding,
}

type EchoResult<T> = Result<Response<T>, Status>;

#[tonic::async_trait]
impl grpc_def::llm_service_server::LlmService for GRPCServer {
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
}
pub async fn create_grpc_server() -> Result<SocketAddr> {
    let model = EmbeddingModel::BGESmallENV15;

    let init_option = InitOptions::new(model.clone())
        .with_cache_dir(std::env::temp_dir())
        .with_show_download_progress(false);

    info!("Initializing the Fastembed: {model}");
    let text_embedding = TextEmbedding::try_new(init_option)
        .with_context(|| format!("Failed to initialize the Fastembed: {model}"))?;

    let server = GRPCServer {
        fastembed_model: text_embedding,
    };

    info!("Checking which port is available on system");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let add = listener.local_addr().unwrap();
    drop(listener);
    info!("Starting the server on: {}", add);

    tokio::spawn(async move {
        Server::builder()
            .add_service(grpc_def::llm_service_server::LlmServiceServer::new(server))
            .serve(add)
            .await
            .expect("Nooope");
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

pub fn create_oramacore_config() -> OramacoreConfig {
    OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        ai_server: AIServiceConfig {
            host: "0.0.0.0".parse().unwrap(),
            port: 0,
            api_key: None,
            max_connections: 1,
            scheme: Scheme::HTTP,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey(Secret::new("my-master-api-key".to_string())),
            output: OutputSideChannelType::InMemory { capacity: 100 },
            hooks: hooks_runtime_config(),
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: OramaModelSerializable(crate::ai::OramaModel::BgeSmall),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
        reader_side: ReadSideConfig {
            input: InputSideChannelType::InMemory { capacity: 100 },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
    }
}
