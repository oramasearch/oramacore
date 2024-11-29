use anyhow::Result;
use futures::future::Either;
use futures::{future, pin_mut};
use hurl::runner::{self, HurlResult};
use hurl::runner::{RunnerOptionsBuilder, Value};
use hurl::util::logger::{LoggerOptionsBuilder, Verbosity};
use hurl_core::typing::Count;
use rustorama::{build_orama, ReadSideConfig, WriteSideConfig};
use std::collections::HashMap;
use std::time::Duration;
use tokio::task::spawn_blocking;
use tokio::time::sleep;

use rustorama::embeddings::{EmbeddingConfig, EmbeddingPreload};
use rustorama::web_server::{HttpConfig, WebServer};

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
    let (collections_writer, collections_reader, mut receiver) = build_orama(
        EmbeddingConfig {
            cache_path: std::env::temp_dir().to_str().unwrap().to_string(),
            hugging_face: None,
            preload: EmbeddingPreload::Bool(false),
        },
        WriteSideConfig {
            output: rustorama::SideChannelType::InMemory,
        },
        ReadSideConfig {
            output: rustorama::SideChannelType::InMemory,
        },
    )
    .await
    .unwrap();

    let web_server = WebServer::new(collections_writer, collections_reader.clone());

    let collections_reader = collections_reader.unwrap();
    tokio::spawn(async move {
        while let Ok(op) = receiver.recv().await {
            collections_reader.update(op).await.expect("OUCH!");
        }
    });

    let http_config = HttpConfig {
        host: HOST.parse().unwrap(),
        port: PORT,
        allow_cors: true,
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

        let variables: HashMap<_, _> = vec![(
            "base_url".to_string(),
            Value::String(format!("http://{}:{}", HOST, PORT)),
        )]
        .into_iter()
        .collect();
        runner::run(content, None, &runner_opts, &variables, &logger_opts)
    })
    .await;

    Ok(r.unwrap().unwrap())
}

async fn run_fulltext_search_test() {
    let content = include_str!("../api-test.hurl");

    let result = run_hurl_test(content).await.unwrap();
    assert!(result.success);
}

async fn run_embedding_search_test() {
    let content = include_str!("../embedding-api-test.hurl");
    let result = run_hurl_test(content).await.unwrap();
    assert!(result.success);
}

async fn run_tests() {
    wait_for_server().await;

    run_fulltext_search_test().await;
    run_embedding_search_test().await;
}

#[tokio::test(flavor = "multi_thread")]
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
