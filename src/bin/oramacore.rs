use std::fs::{self, OpenOptions};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use config::Config;
use oramacore::{start, OramacoreConfig};
use tracing::{info, instrument};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{fmt, Registry};

#[instrument(level = "info")]
fn load_config() -> Result<OramacoreConfig> {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.yaml".to_string());

    let config_path = PathBuf::from(config_path);
    let config_path = fs::canonicalize(&config_path)?;
    let config_path: String = config_path.to_string_lossy().into();

    info!("Reading configuration from {:?}", config_path);

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Yaml))
        .add_source(config::Environment::with_prefix("ORAMACORE"))
        .build()
        .context("Failed to load configuration")?;

    info!("Deserializing configuration");

    settings
        .try_deserialize::<OramacoreConfig>()
        .context("Failed to deserialize configuration")
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let debug_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log.log")
        .unwrap();

    let subscriber = Registry::default()
        .with(
            // stdout layer, to view everything in the console
            fmt::layer().compact().with_ansi(true),
        )
        .with(
            // log-debug file, to log the debug
            fmt::layer().json().with_writer(debug_file),
        );

    tracing::subscriber::set_global_default(subscriber).unwrap();

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
