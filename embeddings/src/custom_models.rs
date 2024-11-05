use anyhow::{Context, Result};
use fastembed::{QuantizationMode, TokenizerFiles, UserDefinedEmbeddingModel};
use reqwest::{
    blocking::Client,
    header::{HeaderMap, HeaderValue, USER_AGENT},
};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const BUFFER_SIZE: usize = 8192;
const USER_AGENT_STRING: &str = "Mozilla/5.0 (compatible; RustBot/1.0)";
const BASE_URL: &str = "https://huggingface.co";
const MODEL_BASE_DIR: &str = ".custom_models";

#[derive(Debug, Clone)]
pub struct CustomModel {
    model_name: String,
    files: ModelFileConfig,
}

#[derive(Debug, Clone)]
pub struct ModelFileConfig {
    pub config: String,
    pub tokenizer: String,
    pub tokenizer_config: String,
    pub special_tokens_map: String,
    pub onnx_model: String,
}

impl CustomModel {
    pub fn try_new(model_name: String, files: ModelFileConfig) -> Result<Self> {
        let model = Self {
            model_name: model_name.clone(),
            files,
        };

        if !model.exists() {
            println!(
                "Cannot find model {} locally. Starting download",
                model_name
            );
            model.download()?;
        };

        Ok(model)
    }

    pub fn exists(&self) -> bool {
        self.get_file_mappings().iter().all(|(_, destination)| {
            self.get_destination_path(destination)
                .unwrap_or_default()
                .exists()
        })
    }

    pub fn load(&self) -> Result<UserDefinedEmbeddingModel> {
        let full_path = |file: &str| format!("{}/{}", MODEL_BASE_DIR, file);
        let onnx_file = fs::read(full_path(&self.files.onnx_model))?;
        let tokenizer_files = TokenizerFiles {
            tokenizer_file: fs::read(full_path(&self.files.tokenizer))?,
            config_file: fs::read(full_path(&self.files.config))?,
            special_tokens_map_file: fs::read(full_path(&self.files.special_tokens_map))?,
            tokenizer_config_file: fs::read(full_path(&self.files.tokenizer_config))?,
        };

        Ok(UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files))
    }

    pub fn missing_files(&self) -> Vec<String> {
        self.get_file_mappings()
            .iter()
            .filter_map(|(_, destination)| {
                let path = self.get_destination_path(destination).ok()?;
                if !path.exists() {
                    Some(destination.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_model_dir(&self) -> PathBuf {
        Path::new(MODEL_BASE_DIR).join(&self.model_name)
    }

    pub fn download(&self) -> Result<()> {
        if self.exists() {
            println!("Model {} already exists locally", self.model_name);
            return Ok(());
        }

        let missing = self.missing_files();
        if !missing.is_empty() {
            println!(
                "Model {} is partially downloaded. Missing files: {:?}",
                self.model_name, missing
            );
        }

        let client = create_client()?;
        let files_to_download = self.get_file_mappings();

        for (url, destination) in files_to_download {
            // Skip if file already exists
            if self
                .get_destination_path(&destination)
                .map(|p| p.exists())
                .unwrap_or(false)
            {
                println!("Skipping existing file: {}", destination);
                continue;
            }

            self.download_file(&client, &url, &destination)
                .with_context(|| format!("Failed to download {}", url))?;
        }

        Ok(())
    }

    fn get_file_mappings(&self) -> Vec<(String, String)> {
        vec![
            (self.files.onnx_model.clone(), "onnx/model.onnx"),
            (
                self.files.special_tokens_map.clone(),
                "special_tokens_map.json",
            ),
            (self.files.tokenizer.clone(), "tokenizer.json"),
            (self.files.tokenizer_config.clone(), "tokenizer_config.json"),
            (self.files.config.clone(), "config.json"),
        ]
        .into_iter()
        .map(|(file, dest)| (self.get_huggingface_url(&file), dest.to_string()))
        .collect()
    }

    fn get_huggingface_url(&self, file: &str) -> String {
        format!("{}/{}/resolve/main/{}", BASE_URL, self.model_name, file)
    }

    fn download_file(&self, client: &Client, url: &str, filename: &str) -> Result<()> {
        println!("Downloading {} from {}", filename, url);

        let dest_path = self.get_destination_path(filename)?;
        let response = follow_redirects(client, url)?;

        if response.status().is_success() {
            self.save_file(response, &dest_path)?;
            println!("Downloaded {} to {:?}", filename, dest_path);
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "Download failed with status: {}",
                response.status()
            ))
        }
    }

    fn get_destination_path(&self, filename: &str) -> Result<PathBuf> {
        let dest_path = self.get_model_dir().join(filename);

        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).context("Failed to create directories")?;
        }

        Ok(dest_path)
    }

    fn save_file(&self, mut response: reqwest::blocking::Response, dest_path: &Path) -> Result<()> {
        let mut file = File::create(dest_path).context("Failed to create destination file")?;
        let total_size = get_content_length(&response);
        let mut downloaded = 0u64;
        let mut buffer = vec![0u8; BUFFER_SIZE];
        let mut last_print = Instant::now();

        loop {
            let bytes_read = match response.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return Err(e).context("Failed to read from response"),
            };

            file.write_all(&buffer[..bytes_read])
                .context("Failed to write to file")?;

            downloaded += bytes_read as u64;
            print_progress(downloaded, total_size, &mut last_print);
        }

        Ok(())
    }
}

fn create_client() -> Result<Client> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static(USER_AGENT_STRING));

    Client::builder()
        .user_agent(USER_AGENT_STRING)
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(3600))
        .build()
        .context("Failed to create HTTP client")
}

fn follow_redirects(client: &Client, initial_url: &str) -> Result<reqwest::blocking::Response> {
    let mut response = client
        .get(initial_url)
        .send()
        .context("Failed to send HTTP request")?;

    while response.status().is_redirection() {
        let new_url = response
            .headers()
            .get("location")
            .and_then(|h| h.to_str().ok())
            .context("Missing or invalid Location header in redirect")?;

        response = client
            .get(new_url)
            .send()
            .context("Failed to follow redirect")?;
    }

    Ok(response)
}

fn get_content_length(response: &reqwest::blocking::Response) -> Option<u64> {
    response
        .headers()
        .get("content-length")
        .and_then(|ct_len| ct_len.to_str().ok())
        .and_then(|ct_len| ct_len.parse::<u64>().ok())
}

fn print_progress(downloaded: u64, total_size: Option<u64>, last_print: &mut Instant) {
    if last_print.elapsed() >= Duration::from_secs(1) {
        match total_size {
            Some(total) => println!(
                "Downloaded: {:.1}MB / {:.1}MB ({:.1}%)",
                downloaded as f64 / 1_000_000.0,
                total as f64 / 1_000_000.0,
                (downloaded as f64 / total as f64) * 100.0
            ),
            None => println!("Downloaded: {:.1}MB", downloaded as f64 / 1_000_000.0),
        }
        *last_print = Instant::now();
    }
}
