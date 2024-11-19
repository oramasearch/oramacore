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
    use hurl::util::logger::{LoggerOptionsBuilder, Verbosity};
    use hurl_core::typing::Count;
    use tokio::task::spawn_blocking;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::sleep;

    use crate::collection_manager::{CollectionManager, CollectionsConfiguration};
    use crate::web_server::{HttpConfig, WebServer};

    #[tokio::test(flavor = "multi_thread", worker_threads = 10)]
    async fn test_hurl() {
        tracing_subscriber::fmt::init();

        const HOST: &str = "127.0.0.1";
        const PORT: u16 = 8080;

        async fn wait_for_server() {
            loop {
                println!("--- Waiting for server to start...");
                let resp = reqwest::Client::new()
                    .get(format!("http://{HOST}:{PORT}/"))
                    .send()
                    .await;

                match resp {
                    Ok(resp) => {
                        if resp.status().is_success() {
                            match resp.text().await {
                                Ok(text) => {
                                    println!("Server started: {}", text);

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
            let manager = CollectionManager::new(CollectionsConfiguration {});
            let manager = Arc::new(manager);
            let web_server = WebServer::new(manager);

            let http_config = HttpConfig {
                host: HOST.parse().unwrap(),
                port: PORT,
                allow_cors: true,
            };
            println!(
                "Starting web server on {}:{}",
                http_config.host, http_config.port
            );
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
                .verbosity(Some(Verbosity::VeryVerbose))
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
