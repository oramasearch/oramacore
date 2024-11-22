use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::Path,
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

use super::LoadedModel;

#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceModelRepoConfig {
    real_model_name: String,
    config: String,
    tokenizer: String,
    tokenizer_config: String,
    special_tokens_map: String,
    onnx_model: String,
    max_input_tokens: usize,
    dimensions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceConfiguration {
    pub(crate) base_url: String,

    pub(crate) user_agent: String,
    pub(crate) connect_timeout: String,
    pub(crate) timeout: String,

    pub model_configs: HashMap<String, HuggingFaceModelRepoConfig>,
}

struct MissingFile {
    repo_path: String,
    filesystem_path: String,
}

fn calculate_model_cache_path(model_name: String, custom_models_path: String) -> String {
    format!("{}/{}", custom_models_path, model_name)
}

fn calculate_missing_files(
    hugging_face_model_repo_config: &HuggingFaceModelRepoConfig,
    model_cache_root_path: &str,
) -> Vec<MissingFile> {
    let mut missing_files = Vec::new();

    let fs_path = format!(
        "{}/{}",
        model_cache_root_path, hugging_face_model_repo_config.onnx_model
    );
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: hugging_face_model_repo_config.onnx_model.clone(),
            filesystem_path: fs_path,
        });
    }

    let fs_path = format!(
        "{}/{}",
        model_cache_root_path, hugging_face_model_repo_config.special_tokens_map
    );
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: hugging_face_model_repo_config.special_tokens_map.clone(),
            filesystem_path: fs_path,
        });
    }

    let fs_path = format!(
        "{}/{}",
        model_cache_root_path, hugging_face_model_repo_config.tokenizer
    );
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: hugging_face_model_repo_config.tokenizer.clone(),
            filesystem_path: fs_path,
        });
    }

    let fs_path = format!(
        "{}/{}",
        model_cache_root_path, hugging_face_model_repo_config.tokenizer_config
    );
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: hugging_face_model_repo_config.tokenizer_config.clone(),
            filesystem_path: fs_path,
        });
    }

    let fs_path = format!(
        "{}/{}",
        model_cache_root_path, hugging_face_model_repo_config.config
    );
    if !std::path::Path::new(&fs_path).exists() {
        missing_files.push(MissingFile {
            repo_path: hugging_face_model_repo_config.config.clone(),
            filesystem_path: fs_path,
        });
    }

    missing_files
}

async fn download_missing_files(
    hugging_face_config: &HuggingFaceConfiguration,
    model_name: &String,
    files_to_download: Vec<MissingFile>,
) -> Result<()> {
    let client = create_client(hugging_face_config)?;

    let mut v = Vec::new();
    for MissingFile {
        filesystem_path,
        repo_path,
    } in files_to_download
    {
        let url = format!(
            "{}/{}/resolve/main/{}",
            hugging_face_config.base_url, model_name, repo_path
        );

        let f = download_file(&client, url, filesystem_path)
            .instrument(trace_span!("Downloading", %model_name, %repo_path));

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

async fn download_file(client: &Client, url: String, destination: String) -> Result<()> {
    info!("Start");

    let mut response = follow_redirects(client, &url).await?;

    info!("Response: {:?}", response.headers().get(CONTENT_LENGTH));

    // Make sure the directory exists
    let parent = Path::new(&destination)
        .parent()
        .with_context(|| format!("Failed to get parent directory: {}", destination))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create directory: {parent:?}"))?;

    let mut file = File::create(&destination)
        .with_context(|| format!("Failed to create file {}", destination))?;

    trace!("Status code: {:?}, headers {:?}", response.status(), response.headers());

    // COpy the response body to the file
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk)
            .with_context(|| format!("Failed to write to file {}", destination))?;
    }

    info!("Downloaded");

    Ok(())
}

fn create_client(hugging_face_config: &HuggingFaceConfiguration) -> Result<Client> {
    let user_agent = HeaderValue::from_str(&hugging_face_config.user_agent).with_context(|| {
        format!(
            "Failed to create user agent header: \"{}\"",
            hugging_face_config.user_agent
        )
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, user_agent.clone());

    let connect_timeout: Duration =
        DurationString::try_from(hugging_face_config.connect_timeout.clone())?.into();
    let timeout: Duration = DurationString::try_from(hugging_face_config.timeout.clone())?.into();

    Client::builder()
        .user_agent(user_agent)
        .default_headers(headers)
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .connection_verbose(true)
        .build()
        .context("Failed to create HTTP client")
}

impl LoadedModel {
    pub async fn try_from_hugging_face(
        hugging_face_config: &HuggingFaceConfiguration,
        cache_path: String,
        model_name: String,
    ) -> Result<Self> {
        let hugging_face_model_repo_config = hugging_face_config
            .model_configs
            .get(&model_name)
            .with_context(|| format!("Model not found in configuration: {model_name}"))?;

        let text_embedding = try_build_text_embedding_model(
            hugging_face_config,
            cache_path,
            model_name.clone(),
            hugging_face_model_repo_config,
        )
        .await?;

        Ok(Self {
            text_embedding,
            model_name,
            max_input_tokens: hugging_face_model_repo_config.max_input_tokens,
            dimensions: hugging_face_model_repo_config.dimensions,
        })
    }
}

async fn try_build_text_embedding_model(
    hugging_face_config: &HuggingFaceConfiguration,
    cache_path: String,
    model_name: String,
    hugging_face_model_repo_config: &HuggingFaceModelRepoConfig,
) -> Result<TextEmbedding> {
    let model_cache_root_path = calculate_model_cache_path(model_name.clone(), cache_path);
    // Make sure the directory exists
    std::fs::create_dir_all(model_cache_root_path.clone())
        .with_context(|| format!("Failed to create directory: {model_cache_root_path}"))?;

    // Support partial downloads
    let missing_files =
        calculate_missing_files(hugging_face_model_repo_config, &model_cache_root_path);

    download_missing_files(
        hugging_face_config,
        &hugging_face_model_repo_config.real_model_name,
        missing_files,
    )
    .await?;

    let full_path = |file: &str| format!("{}/{}", &model_cache_root_path, file);

    info!("Loading model from {}", &model_cache_root_path);

    let onnx_file = fs::read(full_path(&hugging_face_model_repo_config.onnx_model))?;
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: fs::read(full_path(&hugging_face_model_repo_config.tokenizer))?,
        config_file: fs::read(full_path(&hugging_face_model_repo_config.config))?,
        special_tokens_map_file: fs::read(full_path(
            &hugging_face_model_repo_config.special_tokens_map,
        ))?,
        tokenizer_config_file: fs::read(full_path(
            &hugging_face_model_repo_config.tokenizer_config,
        ))?,
    };

    let init_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files);

    let model =
        TextEmbedding::try_new_from_user_defined(init_model, InitOptionsUserDefined::default())
            .with_context(|| format!("Failed to initialize the model: {model_name}"))?;

    info!("Model loaded");

    Ok(model)
}

#[cfg(test)]
mod tests {
    use crate::embeddings::LoadedModel;

    use super::*;

    #[tokio::test]
    async fn test_hf_download_onnx() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let tmp = tempdir::TempDir::new("test_hf_download_onnx")?;
        let cache_path = tmp.path().to_str().unwrap().to_string();
        fs::remove_dir(cache_path.clone())?;

        let rebranded_name = "my-model".to_string();
        let model_name = "Xenova/gte-small".to_string();

        let hugging_face_config = HuggingFaceConfiguration {
            base_url: "https://huggingface.co".to_string(),
            user_agent: "my-agent".to_string(),
            connect_timeout: "60s".to_string(),
            timeout: "60s".to_string(),
            model_configs: HashMap::from_iter([(
                rebranded_name.clone(),
                HuggingFaceModelRepoConfig {
                    real_model_name: model_name,
                    onnx_model: "onnx/model_quantized.onnx".to_string(),
                    special_tokens_map: "special_tokens_map.json".to_string(),
                    tokenizer: "tokenizer.json".to_string(),
                    tokenizer_config: "tokenizer_config.json".to_string(),
                    config: "config.json".to_string(),
                    max_input_tokens: 512,
                    dimensions: 768,
                },
            )]),
        };

        LoadedModel::try_from_hugging_face(
            &hugging_face_config,
            cache_path.clone(),
            rebranded_name.clone(),
        )
        .await
        ?;

        LoadedModel::try_from_hugging_face(&hugging_face_config, cache_path, rebranded_name)
            .await
            ?;

        Ok(())
    }
}
