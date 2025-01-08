use anyhow::Result;
use rustorama::ai_client::client::{AIServiceBackend, AIServiceBackendConfig, Intent, Model};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let start_init = Instant::now();
    let mut service = AIServiceBackend::try_new(AIServiceBackendConfig::default())
        .await
        .unwrap();
    let init_duration = start_init.elapsed();
    println!("Service initialization took: {:?}", init_duration);

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

    let start_embedding = Instant::now();
    let _embedding = service
        .generate_embeddings(vec![input_text], Model::BgeSmall, Intent::Passage)
        .await?;
    let embedding_duration = start_embedding.elapsed();

    println!("\nPerformance Metrics:");
    println!("-------------------");
    println!("Embedding generation took: {:?}", embedding_duration);

    Ok(())
}
