use anyhow::{Context, Result};

pub async fn fetch_image(url: String) -> Result<Option<Vec<u8>>> {
    if !is_image(&url).await? {
        return Ok(None);
    }

    let client = reqwest::Client::new();

    let http_resp = client
        .get(url)
        .send()
        .await
        .context("Failed to send request")?;

    let bytes = http_resp
        .bytes()
        .await
        .context("Failed to get response bytes")?;

    Ok(Some(bytes.to_vec()))
}

async fn is_image(url: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    let res = client
        .head(url)
        .send()
        .await
        .context("Failed to send HEAD request")?;

    let content_type = res
        .headers()
        .get("Content-Type")
        .context("No Content-Type header")?
        .to_str()
        .context("Invalid Content-Type header")?;

    Ok(content_type.starts_with("image"))
}
