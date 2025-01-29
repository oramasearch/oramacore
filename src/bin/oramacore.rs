use std::fs::{self, OpenOptions};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use config::Config;
use oramacore::{start, OramacoreConfig};
use tracing::{instrument, Subscriber};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{fmt, EnvFilter, Registry};

#[instrument(level = "info")]
fn load_config() -> Result<OramacoreConfig> {
    println!("Loading configuration...");
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.yaml".to_string());

    let config_path = PathBuf::from(config_path);
    let config_path = fs::canonicalize(&config_path)?;
    let config_path: String = config_path.to_string_lossy().into();

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Yaml))
        .add_source(config::Environment::with_prefix("ORAMACORE"))
        .build()
        .context("Failed to load configuration")?;

    let oramacore_config = settings
        .try_deserialize::<OramacoreConfig>()
        .context("Failed to deserialize configuration")?;

    println!("Configuration loaded successfully");
    Ok(oramacore_config)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let oramacore_config = match load_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load configuration: {:?}", e);
            return Err(anyhow!("Failed to load configuration"));
        }
    };

    let subscriber = Registry::default().with(fmt::layer().compact().with_ansi(true));
    let subscriber: Box<dyn Subscriber + Send + Sync + 'static> =
        if let Some(file_path) = &oramacore_config.log.file_path {
            println!("Logging to file: {:?}", file_path);
            let debug_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(file_path)
                .expect("Cannot open log file");
            Box::new(subscriber.with(fmt::layer().json().with_writer(debug_file)))
        } else {
            Box::new(subscriber)
        };
    let subscriber = subscriber.with(EnvFilter::from_default_env());
    tracing::subscriber::set_global_default(subscriber).unwrap();

    start(oramacore_config).await?;

    Ok(())
}
