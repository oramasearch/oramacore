use std::sync::Arc;

use anyhow::{Context, Result};
use config::Config;
use rustorama::collection_manager::{CollectionManager, CollectionsConfiguration};
use rustorama::web_server::{HttpConfig, WebServer};
use serde::Deserialize;
use tracing::info;

#[derive(Debug, Deserialize, Clone)]
struct RustoramaConfig {
    http: HttpConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

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
    let manager = CollectionManager::new(CollectionsConfiguration {});
    let manager = Arc::new(manager);
    let web_server = WebServer::new(manager);

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    web_server.start(config.http).await?;

    Ok(())
}
