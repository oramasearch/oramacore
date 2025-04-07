#![cfg(all(feature = "reader", feature = "writer"))]

use crate::ai::AIServiceConfig;
use crate::collection_manager::dto::ApiKey;
use crate::collection_manager::sides::{
    CollectionsWriterConfig, InputSideChannelType, OutputSideChannelType,
    ReadSideConfig,
};
use crate::collection_manager::sides::{IndexesConfig, WriteSideConfig};
use crate::{build_orama, OramacoreConfig};
use ai_service_client::OramaModel;
use anyhow::Result;
use futures::future::Either;
use futures::{future, pin_mut};
use http::uri::Scheme;
use hurl::runner::{self, HurlResult, VariableSet};
use hurl::runner::{RunnerOptionsBuilder, Value};
use hurl::util::logger::{LoggerOptionsBuilder, Verbosity};
use hurl_core::typing::Count;
use redact::Secret;
use std::time::Duration;
use tokio::task::spawn_blocking;
use tokio::time::sleep;
use types::OramaModelSerializable;
use crate::web_server::{HttpConfig, WebServer};

use super::utils::{create_grpc_server, generate_new_path, hooks_runtime_config};

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8080;

async fn wait_for_server() {
    loop {
        let resp = reqwest::Client::new()
            .get(format!("http://{HOST}:{PORT}/"))
            .send()
            .await;

        match resp {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text().await {
                        Ok(_) => {
                            sleep(Duration::from_secs(1)).await;
                            break;
                        }
                        Err(_) => sleep(Duration::from_secs(1)).await,
                    }
                }
            }
            Err(_) => sleep(Duration::from_secs(1)).await,
        }
    }
}

async fn start_server() {
    let address = create_grpc_server().await.unwrap();

    let (collections_writer, collections_reader) = build_orama(OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        ai_server: AIServiceConfig {
            scheme: Scheme::HTTP,
            host: address.ip().to_string(),
            port: address.port(),
            api_key: None,
            max_connections: 1,
            llm: crate::ai::AIServiceLLMConfig {
                port: 8000,
                host: "localhost".to_string(),
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
                default_embedding_model: OramaModelSerializable(OramaModel::MultilingualE5Base),
                insert_batch_commit_size: 10,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
        reader_side: ReadSideConfig {
            input: InputSideChannelType::InMemory { capacity: 100 },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                insert_batch_commit_size: 10,
                commit_interval: Duration::from_secs(3_000),
                notifier: None,
            },
        },
    })
    .await
    .unwrap();

    let web_server = WebServer::new(collections_writer, collections_reader, None);

    let http_config = HttpConfig {
        host: HOST.parse().unwrap(),
        port: PORT,
        allow_cors: true,
        with_prometheus: false,
    };
    web_server.start(http_config).await.unwrap();
}

async fn run_hurl_test(content: &'static str) -> Result<HurlResult> {
    let r = spawn_blocking(move || {
        let runner_opts = RunnerOptionsBuilder::new()
            .fail_fast(true)
            .delay(Duration::from_secs(1))
            .retry(Some(Count::Finite(2)))
            .retry_interval(Duration::from_secs(1))
            .timeout(Duration::from_secs(2))
            .connect_timeout(Duration::from_secs(2))
            .build();
        let logger_opts = LoggerOptionsBuilder::new()
            .verbosity(Some(Verbosity::VeryVerbose))
            .build();

        let mut variables = VariableSet::new();
        variables
            .insert(
                "base_url".to_string(),
                Value::String(format!("http://{}:{}", HOST, PORT)),
            )
            .unwrap();

        runner::run(content, None, &runner_opts, &variables, &logger_opts)
    })
    .await;

    Ok(r.unwrap().unwrap())
}

async fn run_fulltext_search_test() {
    let content = include_str!("./hurl/api-test.hurl");

    let result = run_hurl_test(content).await.unwrap();
    assert!(result.success);
}

async fn run_embedding_search_test() {
    let content = include_str!("./hurl/embedding-api-test.hurl");
    let result = run_hurl_test(content).await.unwrap();
    assert!(result.success);
}

async fn run_tests() {
    wait_for_server().await;

    run_fulltext_search_test().await;
    run_embedding_search_test().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_hurl() {
    let _ = tracing_subscriber::fmt::try_init();

    let future1 = start_server();
    let future2 = run_tests();

    pin_mut!(future1);
    pin_mut!(future2);

    match future::select(future1, future2).await {
        Either::Left((value1, _)) => value1,
        Either::Right((value2, _)) => value2,
    };
}
