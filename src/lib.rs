pub mod indexes;
pub mod types;

pub mod code_parser;
pub mod nlp;

pub mod collection_manager;
pub mod document_storage;

pub mod web_server;

pub mod embeddings;

#[cfg(test)]
mod tests {
    use futures::future::Either;
    use futures::{future, pin_mut};
    use hurl::runner;
    use hurl::runner::{RunnerOptionsBuilder, Value};
    use hurl::util::logger::LoggerOptionsBuilder;
    use hurl_core::typing::Count;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::task::spawn_blocking;
    use tokio::time::sleep;

    use crate::collection_manager::{CollectionManager, CollectionsConfiguration};
    use crate::embeddings::{EmbeddingConfig, EmbeddingService};
    use crate::web_server::{HttpConfig, WebServer};

    #[tokio::test(flavor = "multi_thread", worker_threads = 10)]
    async fn test_hurl() {
        let _ = tracing_subscriber::fmt::try_init();

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

        async fn run() {
            let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
                cache_path: std::env::temp_dir().to_str().unwrap().to_string(),
                hugging_face: None,
                preload_all: false,
            }).unwrap();
            let embedding_service = Arc::new(embedding_service);
            let manager = CollectionManager::new(CollectionsConfiguration {
                embedding_service
            });
            let manager = Arc::new(manager);
            let web_server = WebServer::new(manager);

            let http_config = HttpConfig {
                host: HOST.parse().unwrap(),
                port: PORT,
                allow_cors: true,
            };
            web_server.start(http_config).await.unwrap();
        }
        async fn run_test() {
            wait_for_server().await;

            let content = include_str!("../api-test.hurl");

            let runner_opts = RunnerOptionsBuilder::new()
                .fail_fast(true)
                .delay(Duration::from_secs(1))
                .retry(Some(Count::Finite(2)))
                .retry_interval(Duration::from_secs(1))
                .timeout(Duration::from_secs(2))
                .connect_timeout(Duration::from_secs(2))
                .build();
            let logger_opts = LoggerOptionsBuilder::new()
                // .verbosity(Some(Verbosity::VeryVerbose))
                .build();

            let variables: HashMap<_, _> = vec![(
                "base_url".to_string(),
                Value::String(format!("http://{}:{}", HOST, PORT)),
            )]
            .into_iter()
            .collect();

            let result = spawn_blocking(move || {
                runner::run(content, None, &runner_opts, &variables, &logger_opts)
            })
            .await
            .expect("Spawn blocking task failed")
            .expect("Failed to run test");
            assert!(result.success);
        }

        let future1 = run();
        let future2 = run_test();

        pin_mut!(future1);
        pin_mut!(future2);

        match future::select(future1, future2).await {
            Either::Left((value1, _)) => value1,
            Either::Right((value2, _)) => value2,
        };
    }
}
