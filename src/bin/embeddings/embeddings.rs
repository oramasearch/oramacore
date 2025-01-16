use anyhow::Result;
use rustorama::ai::{
     EmbeddingRequest, HealthCheckRequest,
    OramaIntent, OramaModel,
};
use std::time::Instant;
use tonic::Request;
use rustorama::ai::llm_service_client::LlmServiceClient;

const DEFAULT_HOST: &str = "localhost";
const DEFAULT_PORT: &str = "50051";
async fn create_embeddings_service_client(
) -> Result<LlmServiceClient<tonic::transport::Channel>> {
    let addr = format!("http://{}:{}", DEFAULT_HOST, DEFAULT_PORT,);

    let embeddings_service_client = LlmServiceClient::connect(addr.clone()).await?;

    let health_check_request = Request::new(HealthCheckRequest {
        service: "HealthCheck".to_string(),
    });

    Ok(embeddings_service_client)
}

#[tokio::main]
async fn main() -> Result<()> {
    let start_init = Instant::now();

    let mut embeddings_service_client = create_embeddings_service_client().await?;

    let init_duration = start_init.elapsed();
    println!(
        "Service initialization with health check took: {:?}",
        init_duration
    );

    let input_text = r"
        /**
         * This method is needed to used because of issues like: https://github.com/askorama/orama/issues/301
         * that issue is caused because the array that is pushed is huge (>100k)
         *
         * @example
         * ```ts
         * safeArrayPush(myArray, [1, 2])
         * ```
         */
        export function safeArrayPush<T>(arr: T[], newArr: T[]): void {
          if (newArr.length < MAX_ARGUMENT_FOR_STACK) {
            Array.prototype.push.apply(arr, newArr)
          } else {
            const newArrLength = newArr.length
            for (let i = 0; i < newArrLength; i += MAX_ARGUMENT_FOR_STACK) {
              Array.prototype.push.apply(arr, newArr.slice(i, i + MAX_ARGUMENT_FOR_STACK))
            }
          }
        }
    ".to_string();
    let model = OramaModel::BgeSmall;
    let start_embedding = Instant::now();
    let request = Request::new(EmbeddingRequest {
        input: vec![input_text],
        model: model.into(),
        intent: OramaIntent::Passage.into(),
    });
    let _embedding = embeddings_service_client.get_embedding(request).await?;

    let embedding_duration = start_embedding.elapsed();

    println!("\nPerformance Metrics:");
    println!("-------------------");
    println!("Embedding generation took: {:?}", embedding_duration);

    Ok(())
}
