use std::sync::Arc;

use anyhow::{Context, Result};
use config::Config;
use rustorama::collection_manager::{CollectionManager, CollectionsConfiguration};
use rustorama::web_server::{HttpConfig, WebServer};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
struct RustoramaConfig {
    http: HttpConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.jsonc".to_string());

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

    println!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );
    web_server.start(config.http).await?;

    Ok(())
}

fn create_manager(config: RustoramaConfig) -> CollectionManager {
    CollectionManager::new(CollectionsConfiguration {})
}
