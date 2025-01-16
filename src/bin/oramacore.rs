use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use config::Config;
use oramacore::{start, OramacoreConfig};
use tracing::{info, instrument};

#[instrument(level = "info")]
fn load_config() -> Result<OramacoreConfig> {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.jsonc".to_string());

    let config_path = PathBuf::from(config_path);
    let config_path = fs::canonicalize(&config_path)?;
    let config_path: String = config_path.to_string_lossy().into();

    info!("Reading configuration from {:?}", config_path);

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Json5))
        .add_source(config::Environment::with_prefix("ORAMACORE"))
        .build()
        .context("Failed to load configuration")?;

    info!("Deserializing configuration");

    settings
        .try_deserialize::<OramacoreConfig>()
        .context("Failed to deserialize configuration")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = match load_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load configuration: {:?}", e);
            return Err(anyhow!("Failed to load configuration"));
        }
    };

    start(config).await?;

    Ok(())
}
