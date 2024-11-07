use std::sync::Arc;

use anyhow::{Context, Result};
use collection_manager::{CollectionManager, CollectionsConfiguration};
use config::Config;
use serde::Deserialize;
use storage::Storage;
use web_server::{HttpConfig, WebServer};

#[derive(Debug, Deserialize, Clone)]
struct RustoramaConfig {
    data_dir: String,
    http: HttpConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.json".to_string());

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Json5))
        .add_source(config::Environment::with_prefix("RUSTORAMA"))
        .build()
        .context("Failed to load configuration")?;

    let config = settings
        .try_deserialize::<RustoramaConfig>()
        .context("Failed to deserialize configuration")?;

    start(config).await?;

    Ok(())
}

async fn start(config: RustoramaConfig) -> Result<()> {
    let manager = create_manager(config.clone());
    let manager = Arc::new(manager);
    let web_server = WebServer::new(manager);

    web_server.start(config.http).await?;

    Ok(())
}

fn create_manager(config: RustoramaConfig) -> CollectionManager {
    let storage = Arc::new(Storage::from_path(&config.data_dir));
    CollectionManager::new(CollectionsConfiguration { storage })
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::future::Either;
    use futures::{future, pin_mut};
    use hurl::runner;
    use hurl::runner::{RunnerOptionsBuilder, Value};
    use hurl::util::logger::{LoggerOptionsBuilder, Verbosity};
    use std::collections::HashMap;
    use std::time::Duration;
    use tempdir::TempDir;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_hurl() {
        const HOST: &str = "127.0.0.1";
        const PORT: u16 = 8080;

        async fn run() {
            let data_dir = TempDir::new("string_index_test").unwrap();
            let data_dir: String = data_dir.into_path().to_str().unwrap().to_string();
            let config = RustoramaConfig {
                data_dir,
                http: HttpConfig {
                    host: HOST.parse().unwrap(),
                    port: PORT,
                    allow_cors: true,
                },
            };
            start(config).await.unwrap();
        }
        async fn run_test() {
            sleep(Duration::from_secs(5)).await;

            let content = include_str!("../../api-test.hurl");

            let runner_opts = RunnerOptionsBuilder::new().follow_location(true).build();
            let logger_opts = LoggerOptionsBuilder::new()
                .verbosity(Some(Verbosity::VeryVerbose))
                .build();

            let variables: HashMap<_, _> = vec![(
                "base_url".to_string(),
                Value::String(format!("http://{}:{}", HOST, PORT)),
            )]
            .into_iter()
            .collect();

            let result = runner::run(content, None, &runner_opts, &variables, &logger_opts);
            assert!(result.unwrap().success);
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
