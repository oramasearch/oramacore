use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use ai::{
    automatic_embeddings_selector::AutomaticEmbeddingsSelector, gpu::LocalGPUManager,
    llms::LLMService, AIServiceConfig,
};
use anyhow::{Context, Result};
use collection_manager::sides::{
    channel_creator,
    read::{ReadSide, ReadSideConfig},
    write::{WriteSide, WriteSideConfig},
    InputSideChannelType, OutputSideChannelType,
};
use duration_str::deserialize_duration;
use metrics_exporter_prometheus::PrometheusBuilder;
use orama_js_pool::{self, DomainPermission};
use oramacore_lib::nlp;
use serde::{Deserialize, Serialize};
use tracing::level_filters::LevelFilter;
#[allow(unused_imports)]
use tracing::{info, warn};
use web_server::{HttpConfig, WebServer};

pub mod lock;

pub mod types;

pub mod auth;

pub mod code_parser;

pub mod collection_manager;

pub mod web_server;

mod metrics;

mod merger;

pub mod build_info;

pub mod ai;

pub mod python;

#[cfg(test)]
pub mod tests;

#[derive(Debug, Deserialize, Clone, Default)]
pub struct LogConfig {
    pub file_path: Option<PathBuf>,
    pub sentry_dns: Option<String>,
    pub sentry_environment: Option<String>,
    #[serde(deserialize_with = "deserialize_log_levels", default)]
    pub levels: HashMap<String, LevelFilter>,
}

fn deserialize_log_levels<'de, D>(deserializer: D) -> Result<HashMap<String, LevelFilter>, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Wrapper(HashMap<String, String>);

    let wrapper = Wrapper::deserialize(deserializer)?;

    let mut ret = HashMap::new();
    for (k, v) in wrapper.0 {
        let level: LevelFilter = v.parse().map_err(serde::de::Error::custom)?;
        ret.insert(k, level);
    }

    Ok(ret)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HooksConfig {
    /// List of allowed domains for external HTTP calls from hooks
    #[serde(default)]
    pub allowed_domains: Vec<String>,

    /// List of denied domains for external HTTP calls from hooks
    #[serde(default)]
    pub denied_domains: Vec<String>,

    /// Timeout for hook initialization/builder
    #[serde(
        deserialize_with = "deserialize_duration",
        default = "default_hook_evaluation_timeout"
    )]
    pub evaluation_timeout: Duration,

    /// Timeout for hook execution
    #[serde(
        deserialize_with = "deserialize_duration",
        default = "default_hook_execution_timeout"
    )]
    pub execution_timeout: Duration,
}

fn default_hook_evaluation_timeout() -> Duration {
    Duration::from_millis(200)
}

fn default_hook_execution_timeout() -> Duration {
    Duration::from_millis(1000)
}

impl Default for HooksConfig {
    fn default() -> Self {
        Self {
            allowed_domains: vec![],
            denied_domains: vec![],
            evaluation_timeout: Duration::from_millis(200),
            execution_timeout: Duration::from_millis(1000),
        }
    }
}

impl HooksConfig {
    /// Validates that both allowed_domains and denied_domains are not set at the same time
    pub fn validate(&self) -> Result<()> {
        if !self.allowed_domains.is_empty() && !self.denied_domains.is_empty() {
            anyhow::bail!(
                "HooksConfig: Cannot set both allowed_domains and denied_domains. Please configure only one."
            );
        }
        Ok(())
    }

    /// Converts the HooksConfig into a DomainPermission for the JS pool
    pub fn to_domain_permission(&self) -> DomainPermission {
        if !self.allowed_domains.is_empty() {
            DomainPermission::Allow(self.allowed_domains.clone())
        } else if !self.denied_domains.is_empty() {
            DomainPermission::Deny(self.denied_domains.clone())
        } else {
            DomainPermission::AllowAll
        }
    }
}

#[derive(Deserialize, Clone)]
pub struct OramacoreConfig {
    pub log: LogConfig,
    pub http: HttpConfig,
    pub ai_server: AIServiceConfig,
    #[cfg(any(test, feature = "writer"))]
    pub writer_side: WriteSideConfig,
    #[cfg(any(test, feature = "reader"))]
    pub reader_side: ReadSideConfig,
}

pub async fn start(config: OramacoreConfig) -> Result<()> {
    let build_info = build_info::get_build_info();
    info!(build_info = ?build_info, "Starting oramacore");

    let prometheus_hadler = if config.http.with_prometheus {
        Some(
            PrometheusBuilder::new()
                .set_buckets(&[0.1, 0.5, 0.95, 0.999])
                .expect("failed to set buckets")
                .install_recorder()
                .context("failed to install recorder")?,
        )
    } else {
        None
    };

    let (write_side, read_side) = build_orama(config.clone()).await?;

    info!(
        "Starting web server on {}:{}",
        config.http.host, config.http.port
    );

    let web_server = WebServer::new(write_side.clone(), read_side.clone(), prometheus_hadler);
    web_server.start(config.http).await?;

    if let Some(write_side) = write_side {
        write_side.stop().await?;
        write_side.commit().await?;
    }
    if let Some(read_side) = read_side {
        // Force commit during shutdown to ensure all data is persisted
        read_side.commit(true).await?;
    }

    Ok(())
}

pub async fn build_orama(
    config: OramacoreConfig,
) -> Result<(Option<Arc<WriteSide>>, Option<Arc<ReadSide>>)> {
    info!("Installing CryptoProvider");
    use rustls::crypto::{ring::default_provider, CryptoProvider};

    if CryptoProvider::get_default().is_none() {
        let provider = default_provider();
        let _ = provider.install_default();
    }

    info!("Building ai_service");

    let local_gpu_manager = Arc::new(LocalGPUManager::new());

    if !local_gpu_manager.has_nvidia_gpu()? && config.ai_server.remote_llms.clone().is_none() {
        warn!("No local NVIDIA GPU detected. Also, no remote LLMs configured. All inference sessions will be disabled. Expect errors.");
    }

    let llm_service = match LLMService::try_new(
        config.ai_server.llm.clone(),
        config.ai_server.remote_llms.clone(),
        local_gpu_manager.clone(),
    ) {
        Ok(service) => Arc::new(service),
        Err(err) => {
            anyhow::bail!("Failed to create LLMService: {err}. Please check your configuration.");
        }
    };

    // Validate hooks configuration
    #[cfg(feature = "writer")]
    config
        .writer_side
        .hooks
        .validate()
        .context("Invalid writer hooks configuration")?;
    #[cfg(feature = "reader")]
    config
        .reader_side
        .hooks
        .validate()
        .context("Invalid reader hooks configuration")?;

    #[cfg(feature = "writer")]
    let writer_sender_config: Option<OutputSideChannelType> =
        Some(config.writer_side.output.clone());
    #[cfg(not(feature = "writer"))]
    let writer_sender_config = None;
    #[cfg(feature = "reader")]
    let reader_sender_config: Option<InputSideChannelType> = Some(config.reader_side.input.clone());
    #[cfg(not(feature = "reader"))]
    let reader_sender_config = None;

    let (sender_creator, receiver_creator) =
        channel_creator(writer_sender_config, reader_sender_config).await?;

    info!("Building nlp_service");
    let nlp_service = Arc::new(nlp::NLPService::new());

    info!("Building Python service");
    let python_service = Arc::new(python::PythonService::new(config.ai_server.clone())?);

    #[cfg(feature = "writer")]
    let write_side = {
        info!("Building write_side");
        let automatic_embeddings_selector = Arc::new(AutomaticEmbeddingsSelector::new(
            llm_service.clone(),
            config
                .ai_server
                .embeddings
                .and_then(|e| e.automatic_embeddings_selector),
        ));

        let sender_creator = sender_creator.expect("Sender is not created");

        let write_side = WriteSide::try_load(
            sender_creator,
            config.writer_side,
            nlp_service.clone(),
            llm_service.clone(),
            automatic_embeddings_selector,
            python_service.clone(),
        )
        .await
        .context("Cannot create write side")?;

        Some(write_side)
    };
    #[cfg(not(feature = "writer"))]
    let write_side = {
        warn!("Building write_side skipped due to compilation flag");
        None
    };

    #[cfg(feature = "reader")]
    let read_side = {
        info!("Building read_side");

        let receiver_creator = receiver_creator.expect("Receiver is not created");
        let read_side = ReadSide::try_load(
            receiver_creator,
            nlp_service,
            llm_service,
            config.reader_side,
            local_gpu_manager,
            python_service.clone(),
        )
        .await
        .context("Cannot create read side")?;
        Some(read_side)
    };

    #[cfg(not(feature = "reader"))]
    let read_side = {
        warn!("Building read_side skipped due to compilation flag");
        None
    };

    Ok((write_side, read_side))
}
