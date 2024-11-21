use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use config::Config;
use rustorama::collection_manager::{CollectionManager, CollectionsConfiguration};
use rustorama::embeddings::{EmbeddingConfig, EmbeddingService};
use rustorama::web_server::{HttpConfig, WebServer};
use serde::Deserialize;
use tracing::{info, instrument};

#[derive(Debug, Deserialize, Clone)]
struct RustoramaConfig {
    http: HttpConfig,
    embeddings: EmbeddingConfig,
}

#[instrument(level = "info")]
fn load_config() -> Result<RustoramaConfig> {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.jsonc".to_string());
    
    let config_path = PathBuf::from(config_path);
    let config_path = fs::canonicalize(&config_path)?;
    let config_path: String = config_path.to_string_lossy().into();

    info!("Reading configuration from {:?}", config_path);

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Json5))
        .add_source(config::Environment::with_prefix("RUSTORAMA"))
        .build()
        .context("Failed to load configuration")?;

    info!("Deserializing configuration");

    settings
        .try_deserialize::<RustoramaConfig>()
        .context("Failed to deserialize configuration")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = load_config()?;

    start(config).await?;

    Ok(())
}

async fn start(config: RustoramaConfig) -> Result<()> {
    let embedding_service = EmbeddingService::try_new(config.embeddings)
        .with_context(|| "Failed to initialize the EmbeddingService")?;
    let embedding_service = Arc::new(embedding_service);
    let manager = CollectionManager::new(CollectionsConfiguration {
        embedding_service,
    });
    let manager = Arc::new(manager);
    let web_server = WebServer::new(manager);

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    web_server.start(config.http).await?;

    Ok(())
}
