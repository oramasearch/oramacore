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

    let result1 = qt
        .translate(q1.to_string(), Some(schema.to_string()))
        .await?
        .unwrap();

    println!("{}", result1);

    let q2 = "What are the best headphones under $200 for listening to hi-fi music?";

    let result2 = qt.translate(q2.to_string(), None).await?.unwrap();

    print!("{}", result2);

    Ok(())
}
