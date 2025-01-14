use anyhow::Result;
use tokio_stream::StreamExt;

use rustorama::ai_client::client::{AIServiceBackend, AIServiceBackendConfig, LlmType};

async fn process_stream(mut backend: AIServiceBackend) -> Result<()> {
    let model = LlmType::GoogleQueryTranslator;
    let prompt = "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do? True story here.".to_string();

    let response = backend.call_llm_stream(model, prompt).await?;
    let mut stream = response.into_inner();

    while let Some(response) = stream.next().await {
        let response = response?;
        println!("Received chunk: {}", response.text_chunk);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = AIServiceBackendConfig::default();

    let mut backend = AIServiceBackend::try_new(config).await?;

    let health = backend.health_check().await?;

    dbg!(health);

    let model = LlmType::GoogleQueryTranslator;
    let prompt = "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do? True story here.".to_string();
    let response = backend.call_llm(model, prompt).await?;
    println!("LLM Response: {}", response.into_inner().text);

    process_stream(backend).await?;

    Ok(())
}
