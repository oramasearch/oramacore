use std::{
    fs,
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use anyhow::{Context, Result};
use duration_string::DurationString;
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use grpc_def::Embedding;
use http::uri::Scheme;
use redact::Secret;
use serde_json::json;
use std::str::FromStr;
use tokio::time::sleep;
use tonic::{transport::Server, Status};
use tracing::info;

use crate::{
    ai::AIServiceConfig,
    build_orama,
    collection_manager::sides::{
        hooks::{HooksRuntimeConfig, SelectEmbeddingsPropertiesHooksRuntimeConfig},
        CollectionsWriterConfig, IndexesConfig, InputSideChannelType, OramaModelSerializable,
        OutputSideChannelType, ReadSide, ReadSideConfig, WriteSide, WriteSideConfig,
    },
    types::{ApiKey, CollectionId, DocumentList, InsertDocumentsResult},
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
        config.ai_server.host = address.ip().to_string();
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
                "id": collection_id,
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
) -> Result<InsertDocumentsResult>
where
    I: IntoIterator<Item = serde_json::Value>,
{
    let document_list: Vec<serde_json::value::Value> = docs.into_iter().collect();
    let document_list: DocumentList = document_list.try_into()?;

    let result = write_side
        .insert_documents(write_api_key, collection_id, document_list)
        .await?;

    sleep(Duration::from_millis(1_000)).await;

    Ok(result)
}

pub mod grpc_def {
    tonic::include_proto!("orama_ai_service");
}

pub struct GRPCServer {
    fastembed_model: Arc<TextEmbedding>,
    context_evaluator: Arc<TextEmbedding>,
}

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
        // `6` means `SentenceTransformersParaphraseMultilingualMiniLML12v2`
        let model = match req.model {
            0 => self.fastembed_model.clone(),
            6 => self.context_evaluator.clone(),
            _ => return Err(Status::invalid_argument("Invalid model")),
        };

        let embed = model
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

static CELL: OnceLock<Result<(Arc<TextEmbedding>, Arc<TextEmbedding>)>> = OnceLock::new();

pub async fn create_grpc_server() -> Result<SocketAddr> {
    let model = EmbeddingModel::BGESmallENV15;

    let text_embedding = CELL.get_or_init(|| {
        let cwd = std::env::current_dir().unwrap();
        let cache_dir = cwd.join(".custom_models");
        std::fs::create_dir_all(&cache_dir).expect("Cannot create cache dir");

        let init_option = InitOptions::new(model.clone())
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(false);

        let context_evaluator_model_dir =
            cache_dir.join("sentenceTransformersParaphraseMultilingualMiniLML12v2");
        std::fs::create_dir_all(&context_evaluator_model_dir).expect("Cannot create cache dir");

        let onnx_file = std::fs::read(context_evaluator_model_dir.join("model.onnx"))
            .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let config_file = std::fs::read(context_evaluator_model_dir.join("config.json"))
            .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let tokenizer_file = std::fs::read(context_evaluator_model_dir.join("tokenizer.json"))
            .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let special_tokens_map_file = std::fs::read(
            context_evaluator_model_dir.join("special_tokens_map.json"),
        )
        .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let tokenizer_config_file = std::fs::read(
            context_evaluator_model_dir.join("tokenizer_config.json"),
        )
        .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let tokenizer_files = TokenizerFiles {
            config_file,
            special_tokens_map_file,
            tokenizer_file,
            tokenizer_config_file,
        };
        let user_defined_model =
            UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

        // Try creating a TextEmbedding instance from the user-defined model
        let context_evaluator: TextEmbedding = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .unwrap();

        info!("Initializing the Fastembed: {model}");
        let text_embedding = TextEmbedding::try_new(init_option)
            .with_context(|| format!("Failed to initialize the Fastembed: {model}"))?;
        Ok((Arc::new(text_embedding), Arc::new(context_evaluator)))
    });
    let (text_embedding, context_evaluator) = text_embedding.as_ref().unwrap().clone();

    let server = GRPCServer {
        fastembed_model: text_embedding,
        context_evaluator,
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
            llm: crate::ai::AIServiceLLMConfig {
                host: "localhost".to_string(),
                port: 8000,
                model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
            },
            remote_llms: None,
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
                notifier: None,
            },
        },
    }
}
