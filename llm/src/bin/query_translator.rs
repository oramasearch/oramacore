use anyhow::Result;
use llm::query_translator::QueryTranslator;

#[tokio::main]
async fn main() -> Result<()> {
    let qt = QueryTranslator::try_new().await?;

    let q1 = "What are the best headphones under $200 for listening to hi-fi music?";
    let schema = r#"{
        "name": "string",
        "description: "string",
        "price": "number",
        "tags": "enum[]"
    }"#;

    let result = qt
        .translate(q1.to_string(), Some(schema.to_string()))
        .await?
        .unwrap();

    dbg!(result);

    Ok(())
}
