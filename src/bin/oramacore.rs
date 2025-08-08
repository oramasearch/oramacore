use std::borrow::Cow;
use std::fs::{self, OpenOptions};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use config::Config;
use itertools::Itertools;
use oramacore::build_info::get_build_version;
use oramacore::{start, OramacoreConfig};
use tracing::instrument;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{fmt, EnvFilter, Registry};

#[instrument(level = "info")]
fn load_config() -> Result<OramacoreConfig> {
    println!("Loading configuration...");
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.yaml".to_string());

    let config_path = PathBuf::from(config_path);
    let config_path = fs::canonicalize(&config_path)?;
    let config_path: String = config_path.to_string_lossy().into();

    let builder = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Yaml))
        .add_source(config::Environment::with_prefix("ORAMACORE").separator("_"));
    let settings = builder.build().context("Failed to load configuration")?;

    let oramacore_config = settings
        .try_deserialize::<OramacoreConfig>()
        .context("Failed to deserialize configuration")?;

    println!("Configuration loaded successfully");
    Ok(oramacore_config)
}

fn main() -> anyhow::Result<()> {
    let build_info = oramacore::build_info::get_build_info();
    println!("{build_info}",);

    let oramacore_config = match load_config() {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load configuration: {e:?}");
            return Err(anyhow!("Failed to load configuration"));
        }
    };

    let mut sentry_guard = None;
    if let Some(sentry_dsn) = oramacore_config.log.sentry_dns.clone() {
        let _guard = sentry::init(sentry::ClientOptions {
            // Enable capturing of traces; set this a to lower value in production:
            dsn: Some(sentry_dsn.parse().expect("Invalid Sentry DSN")),
            traces_sample_rate: 1.0,
            release: Some(Cow::Owned(get_build_version())),
            environment: oramacore_config
                .log
                .sentry_environment
                .clone()
                .map(|s| s.into()),
            ..sentry::ClientOptions::default()
        });
        sentry_guard = Some(_guard);
    }

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { run(oramacore_config).await })?;

    drop(sentry_guard);

    Ok(())
}

async fn run(oramacore_config: OramacoreConfig) -> anyhow::Result<()> {
    let subscriber = Registry::default().with(fmt::layer().compact().with_ansi(true));

    let mut levels = oramacore_config.log.levels.clone();
    if !levels.contains_key("oramacore") {
        levels.insert("oramacore".to_string(), LevelFilter::INFO);
    }
    let levels = levels
        .into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .join(",");
    let env = EnvFilter::builder()
        .with_default_directive(LevelFilter::WARN.into())
        .parse(levels)
        .context("Invalid levels")?;

    match (
        oramacore_config.log.sentry_dns.is_some(),
        &oramacore_config.log.file_path,
    ) {
        (false, None) => {
            let subscriber = subscriber.with(env);
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
        (true, None) => {
            let subscriber = subscriber.with(env).with(sentry_tracing::layer());
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
        (false, Some(file_path)) => {
            println!("Logging to file: {file_path:?}");
            let debug_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(file_path)
                .expect("Cannot open log file");
            let subscriber = subscriber
                .with(env)
                .with(fmt::layer().json().with_writer(debug_file));
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
        (true, Some(file_path)) => {
            println!("Logging to file: {file_path:?}");
            let debug_file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(file_path)
                .expect("Cannot open log file");

            let subscriber = subscriber.with(env).with(sentry_tracing::layer());
            let subscriber = subscriber.with(fmt::layer().json().with_writer(debug_file));
            tracing::subscriber::set_global_default(subscriber).unwrap();
        }
    }

    start(oramacore_config).await?;

    Ok(())
}
