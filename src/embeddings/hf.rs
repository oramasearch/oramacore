use std::{
    collections::HashMap,
    fmt::Debug,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use duration_string::DurationString;
use fastembed::{InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use futures::future::try_join_all;
use http::{
    header::{CONTENT_LENGTH, USER_AGENT},
    HeaderMap, HeaderValue,
};
use reqwest::Client;
use serde::Deserialize;
use tracing::{info, trace, trace_span, Instrument};

#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceModelRepoFilesConfig {
    pub onnx_model: PathBuf,
    pub special_tokens_map: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
    pub config: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceModelRepoConfig {
    pub real_model_name: String,
    pub files: HuggingFaceModelRepoFilesConfig,
    pub max_input_tokens: usize,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceRepoConfig {
    pub base_url: String,
    pub user_agent: String,
    #[serde(deserialize_with = "deserialize_duration_string")]
    pub connect_timeout: Duration,
    #[serde(deserialize_with = "deserialize_duration_string")]
    pub timeout: Duration,
    pub cache_path: PathBuf,
}

fn deserialize_duration_string<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    DurationString::try_from(s)
        .map_err(serde::de::Error::custom)
        .map(Into::into)
}

pub struct HuggingFaceModel {
    model_name: String,
    model: TextEmbedding,
    dimensions: usize,
}
impl HuggingFaceModel {
    pub fn model_name(&self) -> String {
        self.model_name.clone()
    }
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn embed_query(&self, input: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        self.model.embed(input, None)
    }
    pub fn embed_passage(&self, input: Vec<&String>) -> Result<Vec<Vec<f32>>> {
        self.model.embed(input, None)
    }
}
impl Debug for HuggingFaceModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HuggingFaceModel({})", self.model_name)
    }
}

#[derive(Debug)]
pub struct HuggingFaceRepo {
    hugging_face_config: HuggingFaceRepoConfig,
    model_configs: HashMap<String, HuggingFaceModelRepoConfig>,
}
impl HuggingFaceRepo {
    pub fn new(
        hugging_face_config: HuggingFaceRepoConfig,
        model_configs: HashMap<String, HuggingFaceModelRepoConfig>,
    ) -> Self {
        Self {
            hugging_face_config,
            model_configs,
        }
    }

    pub async fn load_model(&self, model_name: String) -> Result<HuggingFaceModel> {
        let model_repo_config = match self.model_configs.get(&model_name) {
            Some(config) => config,
            None => {
                return Err(anyhow!("Model not found: {}", model_name));
            }
        };

        let cache_path = &self.hugging_face_config.cache_path;

        let model_cache_root_path = calculate_model_cache_path(model_name.clone(), cache_path);
        // Make sure the directory exists
        std::fs::create_dir_all(&model_cache_root_path)
            .with_context(|| format!("Failed to create directory: {model_cache_root_path:?}"))?;

        // Support partial downloads
        let missing_files =
            calculate_missing_files(&model_repo_config.files, &model_cache_root_path);

        download_missing_files(
            &self.hugging_face_config,
            &model_repo_config.real_model_name,
            missing_files,
        )
        .await?;

        let full_path = |file: &PathBuf| model_cache_root_path.join(file);

        info!("Loading model from '{:?}'", &model_cache_root_path);
        let files = &model_repo_config.files;

        let onnx_file = std::fs::read(full_path(&files.onnx_model))?;
        let tokenizer_files = TokenizerFiles {
            tokenizer_file: std::fs::read(full_path(&files.tokenizer))?,
            config_file: std::fs::read(full_path(&files.config))?,
            special_tokens_map_file: std::fs::read(full_path(&files.special_tokens_map))?,
            tokenizer_config_file: std::fs::read(full_path(&files.tokenizer_config))?,
        };

        let init_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files);

        let model =
            TextEmbedding::try_new_from_user_defined(init_model, InitOptionsUserDefined::default())
                .with_context(|| format!("Failed to initialize the model: {model_name}"))?;

        let model = HuggingFaceModel {
            model_name: model_name.clone(),
            model,
            dimensions: model_repo_config.dimensions,
        };

        Ok(model)
    }
}

struct MissingFile<'p> {
    repo_path: &'p PathBuf,
    filesystem_path: PathBuf,
}

fn calculate_model_cache_path(model_name: String, custom_models_path: &Path) -> PathBuf {
    custom_models_path.join(model_name)
}

fn calculate_missing_files<'p>(
    model_files: &'p HuggingFaceModelRepoFilesConfig,
    model_cache_root_path: &Path,
) -> Vec<MissingFile<'p>> {
    let mut missing_files = Vec::new();

    let fs_path = model_cache_root_path.join(&model_files.onnx_model);
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: &model_files.onnx_model,
            filesystem_path: fs_path,
        });
    }

    let fs_path = model_cache_root_path.join(&model_files.special_tokens_map);
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: &model_files.special_tokens_map,
            filesystem_path: fs_path,
        });
    }

    let fs_path = model_cache_root_path.join(&model_files.tokenizer);
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: &model_files.tokenizer,
            filesystem_path: fs_path,
        });
    }

    let fs_path = model_cache_root_path.join(&model_files.tokenizer_config);
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: &model_files.tokenizer_config,
            filesystem_path: fs_path,
        });
    }

    let fs_path = model_cache_root_path.join(&model_files.config);
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: &model_files.config,
            filesystem_path: fs_path,
        });
    }

    missing_files
}

async fn download_missing_files<'p>(
    hugging_face_config: &HuggingFaceRepoConfig,
    model_name: &String,
    files_to_download: Vec<MissingFile<'p>>,
) -> Result<()> {
    if files_to_download.is_empty() {
        return Ok(());
    }

    let client = create_client(hugging_face_config)?;

    let mut v = Vec::new();
    for MissingFile {
        filesystem_path,
        repo_path,
    } in files_to_download
    {
        let url = format!(
            "{}/{}/resolve/main/{}",
            hugging_face_config.base_url,
            model_name,
            repo_path.to_string_lossy()
        );

        let f = download_file(&client, url, filesystem_path).instrument(
            trace_span!("Downloading", %model_name, repo_path = %repo_path.to_string_lossy()),
        );

        v.push(f);
    }

    try_join_all(v).await?;

    Ok(())
}

async fn follow_redirects(client: &Client, initial_url: &str) -> Result<reqwest::Response> {
    let mut response = client
        .get(initial_url)
        .send()
        .await
        .context("Failed to send HTTP request")?;

    if !response.status().is_success() {
        return Err(anyhow!("Failed to download file: {}", response.status()));
    }

    let mut max_redirect = 10;
    while response.status().is_redirection() {
        max_redirect -= 1;

        if max_redirect == 0 {
            return Err(anyhow!("Too many redirects"));
        }

        let new_url = response
            .headers()
            .get("location")
            .and_then(|h| h.to_str().ok())
            .context("Missing or invalid Location header in redirect")?;

        info!("Following redirect to {}", new_url);

        response = client
            .get(new_url)
            .send()
            .await
            .context("Failed to follow redirect")?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to download file: {}", response.status()));
        }
    }

    Ok(response)
}

async fn download_file(client: &Client, url: String, destination: PathBuf) -> Result<()> {
    info!("Start");

    let mut response = follow_redirects(client, &url).await?;

    info!("Response: {:?}", response.headers().get(CONTENT_LENGTH));

    // Make sure the directory exists
    let parent = Path::new(&destination)
        .parent()
        .with_context(|| format!("Failed to get parent directory: '{:?}'", destination))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create directory: '{parent:?}'"))?;

    let mut file = File::create(&destination)
        .with_context(|| format!("Failed to create file '{:?}'", destination))?;

    trace!(
        "Status code: {:?}, headers {:?}",
        response.status(),
        response.headers()
    );

    // COpy the response body to the file
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)
            .with_context(|| format!("Failed to write to file '{:?}'", destination))?;
    }

    info!("Downloaded");

    Ok(())
}

fn create_client(hugging_face_config: &HuggingFaceRepoConfig) -> Result<Client> {
    let user_agent = HeaderValue::from_str(&hugging_face_config.user_agent).with_context(|| {
        format!(
            "Failed to create user agent header: \"{}\"",
            hugging_face_config.user_agent
        )
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, user_agent.clone());

    Client::builder()
        .user_agent(user_agent)
        .default_headers(headers)
        .connect_timeout(hugging_face_config.connect_timeout)
        .timeout(hugging_face_config.timeout)
        .connection_verbose(true)
        .build()
        .context("Failed to create HTTP client")
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[tokio::test]
    async fn test_embedding_run_hf() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let tmp = tempdir::TempDir::new("test_hf_download_onnx")?;
        let cache_path: PathBuf = tmp.path().into();
        std::fs::remove_dir(cache_path.clone())?;

        let rebranded_name = "my-model".to_string();
        let model_name = "Xenova/gte-small".to_string();

        let hugging_face_config = HuggingFaceRepoConfig {
            base_url: "https://huggingface.co".to_string(),
            user_agent: "my-agent".to_string(),
            cache_path: generate_new_path(),
            connect_timeout: *DurationString::try_from("60s".to_string()).unwrap(),
            timeout: *DurationString::try_from("60s".to_string()).unwrap(),
        };

        let model = HuggingFaceModelRepoConfig {
            real_model_name: model_name,
            files: HuggingFaceModelRepoFilesConfig {
                onnx_model: "onnx/model_quantized.onnx".to_string().into(),
                special_tokens_map: "special_tokens_map.json".to_string().into(),
                tokenizer: "tokenizer.json".to_string().into(),
                tokenizer_config: "tokenizer_config.json".to_string().into(),
                config: "config.json".to_string().into(),
            },
            max_input_tokens: 512,
            dimensions: 384,
        };

        let repo = HuggingFaceRepo::new(
            hugging_face_config,
            HashMap::from_iter([(rebranded_name.clone(), model)]),
        );
        let model = repo
            .load_model(rebranded_name)
            .await
            .expect("Failed to cache model");

        let output = model.embed_query(vec![&"foo".to_string()])?;

        assert_eq!(output[0].len(), 384);

        Ok(())
    }

    #[test]
    fn test_embedding_deserialize_hugging_face_repo_config() -> Result<()> {
        let config = r#"
            {
                "base_url": "https://huggingface.co",
                "user_agent": "my-agent",
                "connect_timeout": "60s",
                "timeout": "60s",
                "cache_path": "/tmp/hf_cache"
            }
        "#;

        let config: HuggingFaceRepoConfig = serde_json::from_str(config)?;

        assert_eq!(config.base_url, "https://huggingface.co");
        assert_eq!(config.user_agent, "my-agent");
        assert_eq!(config.connect_timeout, Duration::from_secs(60));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.cache_path, PathBuf::from("/tmp/hf_cache"));

        Ok(())
    }
}
