use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub struct CustomModelFiles {
    pub model_name: String,
    pub config: String,
    pub tokenizer: String,
    pub tokenizer_config: String,
    pub special_tokens_map: String,
    pub onnx_model: String,
}

pub struct ConfigCustomModelFiles {
    pub config: String,
    pub tokenizer: String,
    pub tokenizer_config: String,
    pub special_tokens_map: String,
    pub onnx_model: String,
}

impl CustomModelFiles {
    pub fn new(model_name: String, config: ConfigCustomModelFiles) -> Self {
        Self {
            model_name,
            onnx_model: config.onnx_model,
            special_tokens_map: config.special_tokens_map,
            tokenizer: config.tokenizer,
            tokenizer_config: config.tokenizer_config,
            config: config.config,
        }
    }

    pub fn download(&self) -> Result<()> {
        let onnx_model = self.get_huggingface_absolute_path(&self.onnx_model);
        let special_tokens_map = self.get_huggingface_absolute_path(&self.special_tokens_map);
        let tokenizer = self.get_huggingface_absolute_path(&self.tokenizer);
        let tokenizer_config = self.get_huggingface_absolute_path(&self.tokenizer_config);
        let config = self.get_huggingface_absolute_path(&self.config);

        let client = Self::create_client()?;

        self.download_file(&client, onnx_model, "onnx/model.onnx".to_string())?;
        self.download_file(
            &client,
            special_tokens_map,
            "special_tokens_map.json".to_string(),
        )?;
        self.download_file(&client, tokenizer, "tokenizer.json".to_string())?;
        self.download_file(
            &client,
            tokenizer_config,
            "tokenizer_config.json".to_string(),
        )?;
        self.download_file(&client, config, "config.json".to_string())?;

        Ok(())
    }

    fn get_huggingface_absolute_path(&self, file: &String) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.model_name, file
        )
    }

    fn create_client() -> Result<Client> {
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("Mozilla/5.0 (compatible; RustBot/1.0)"),
        );

        Client::builder()
            .user_agent("Mozilla/5.0 (compatible; RustBot/1.0)")
            .default_headers(headers)
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(3600))
            .build()
            .context("Failed to create HTTP client")
    }

    fn download_file(&self, client: &Client, url: String, file_name: String) -> Result<()> {
        println!("Downloading {} from {}", file_name, url);

        let base_path = Path::new(".custom_models").join(&self.model_name);
        let destination_path = base_path.join(&file_name);

        if let Some(parent) = destination_path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create directories")?;
        }

        let mut response = client
            .get(&url)
            .send()
            .context("Failed to send HTTP request")?;

        let mut final_response = response;
        while final_response.status().is_redirection() {
            let new_url = final_response
                .headers()
                .get("location")
                .and_then(|h| h.to_str().ok())
                .context("Missing or invalid Location header in redirect")?;

            final_response = client
                .get(new_url)
                .send()
                .context("Failed to follow redirect")?;
        }

        if final_response.status().is_success() {
            let mut dest_file =
                File::create(&destination_path).context("Failed to create destination file")?;

            let total_size = final_response
                .headers()
                .get("content-length")
                .and_then(|ct_len| ct_len.to_str().ok())
                .and_then(|ct_len| ct_len.parse::<u64>().ok());

            let mut downloaded = 0u64;
            let mut buffer = vec![0u8; 8192];
            let mut last_print = std::time::Instant::now();

            loop {
                let bytes_read = match final_response.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) => return Err(e.into()),
                };

                dest_file
                    .write_all(&buffer[..bytes_read])
                    .context("Failed to write to file")?;

                downloaded += bytes_read as u64;

                if last_print.elapsed() >= std::time::Duration::from_secs(1) {
                    if let Some(total) = total_size {
                        println!(
                            "Downloaded: {:.1}MB / {:.1}MB ({:.1}%)",
                            downloaded as f64 / 1_000_000.0,
                            total as f64 / 1_000_000.0,
                            (downloaded as f64 / total as f64) * 100.0
                        );
                    } else {
                        println!("Downloaded: {:.1}MB", downloaded as f64 / 1_000_000.0);
                    }
                    last_print = std::time::Instant::now();
                }
            }

            println!("Downloaded {} to {:?}", file_name, destination_path);
        } else {
            return Err(anyhow::anyhow!(
                "Failed to download file: {:?}",
                final_response.status()
            ));
        }

        Ok(())
    }
}
