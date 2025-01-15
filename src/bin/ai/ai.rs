use anyhow::Result;
use rustorama::ai::{
    health_check_service_client::HealthCheckServiceClient, llm_service_client::LlmServiceClient,
    HealthCheckRequest, LlmRequest, LlmType,
};
use tokio_stream::StreamExt;
use tonic::Request;

const DEFAULT_HOST: &str = "localhost";
const DEFAULT_PORT: &str = "50051";
async fn create_llm_service_client() -> Result<LlmServiceClient<tonic::transport::Channel>> {
    let addr = format!("http://{}:{}", DEFAULT_HOST, DEFAULT_PORT,);

    let llm_client = LlmServiceClient::connect(addr.clone()).await?;
    let mut health_check_client = HealthCheckServiceClient::connect(addr).await?;

    let health_check_request = Request::new(HealthCheckRequest {
        service: "HealthCheck".to_string(),
    });

    // Test it is alive
    dbg!(
        health_check_client
            .check_health(health_check_request)
            .await?
    );

    Ok(llm_client)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut llm_client = create_llm_service_client().await?;

    let model = LlmType::GoogleQueryTranslator;
    let prompt = "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do? True story here.".to_string();
    let request = Request::new(LlmRequest {
        model: model.into(),
        prompt: prompt.clone(),
    });
    let response = llm_client.call_llm(request).await?;

    println!("LLM Response: {}", response.into_inner().text);

    let request = Request::new(LlmRequest {
        model: model.into(),
        prompt,
    });

    let response = llm_client.call_llm_stream(request).await?;
    let mut stream = response.into_inner();

    while let Some(response) = stream.next().await {
        let response = response?;
        println!("Received chunk: {}", response.text_chunk);
    }

    Ok(())
}
