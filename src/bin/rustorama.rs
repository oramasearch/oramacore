use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use config::Config;
use rustorama::embeddings::{EmbeddingConfig, EmbeddingService};
use rustorama::web_server::{HttpConfig, WebServer};
use rustorama::{build_orama, ReadSideConfig, WriteSideConfig};
use serde::Deserialize;
use tracing::{info, instrument};

#[derive(Debug, Deserialize, Clone)]
struct RustoramaConfig {
    http: HttpConfig,
    embeddings: EmbeddingConfig,
    writer_side: Option<WriteSideConfig>,
    reader_side: Option<ReadSideConfig>,
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
    let (writer, reader, receiver) = build_orama(
        config.embeddings,
        config
            .writer_side
            .expect("Writer side configuration is required"),
        config
            .reader_side
            .expect("Reader side configuration is required"),
    )
    .await?;

    let web_server = WebServer::new(writer, reader);

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    web_server.start(config.http).await?;

    Ok(())
}
