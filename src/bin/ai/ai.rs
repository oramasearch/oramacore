use anyhow::Result;

use rustorama::ai_client::client::{AIServiceBackend, AIServiceBackendConfig, LlmType};

#[tokio::main]
async fn main() -> Result<()> {
    let config = AIServiceBackendConfig::default();

    let mut backend = AIServiceBackend::try_new(config).await?;

    let health = backend.health_check().await?;

    println!("Health check: {:?}", health);

    Ok(())
}
