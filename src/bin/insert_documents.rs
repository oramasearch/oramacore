use std::{fs, time::Duration};

use serde_json::json;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    let response = client
        .post("http:/localhost:8080/v0/collections")
        .json(&json!({
            "id": "tommaso-test",
        }))
        .send()
        .await?;
    response.error_for_status()?;

    let content =
        fs::read_to_string("/Users/allevo/repos/rustorama/src/bin/imdb_top_1000_tv_series.json")?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let json = json.as_array().unwrap();

    let mut i = 0;
    loop {
        let b: Vec<serde_json::Value> = json
            .iter()
            .skip(i)
            .take(10)
            .map(|v| {
                json!({
                    "title": v.get("title").unwrap(),
                    "plot": v.get("plot").unwrap(),
                    "rating": v.get("rating").unwrap(),
                    "poster_url": v.get("poster_url").unwrap(),
                })
            })
            .collect();

        let response = client
            .patch("http:/localhost:8080/v0/collections/tommaso-test/documents")
            .json(&b)
            .send()
            .await?;
        response.error_for_status()?;

        i += 1;
        if i >= json.len() {
            i = 0;
        }

        sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}
