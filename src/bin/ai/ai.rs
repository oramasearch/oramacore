use anyhow::Result;

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

#[tokio::main]
async fn main() -> Result<()> {
    let config = AIServiceBackendConfig::default();

    let mut backend = AIServiceBackend::try_new(config).await?;

    let health = backend.health_check().await?;

    println!("Health check: {:?}", health);

    Ok(())
}
