use anyhow::Result;
use rustorama::embeddings::OramaModel;

#[tokio::main]
async fn main() -> Result<()> {
    let model = OramaModel::JinaV2BaseCode.try_new().await?;

    let embedding = model.embed(
        vec![r"
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
        ".to_string()],
        Some(1),
    )?;

    dbg!(embedding.first().unwrap());

    Ok(())
}
