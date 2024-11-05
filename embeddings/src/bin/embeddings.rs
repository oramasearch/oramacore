use anyhow::Result;
use embeddings::custom_models::{ConfigCustomModelFiles, CustomModelFiles};

fn main() -> Result<()> {
    let model_files = CustomModelFiles::new(
        "jinaai/jina-embeddings-v2-base-code".to_string(),
        ConfigCustomModelFiles {
            onnx_model: "onnx/model.onnx".to_string(),
            config: "config.json".to_string(),
            tokenizer_config: "tokenizer_config.json".to_string(),
            tokenizer: "tokenizer.json".to_string(),
            special_tokens_map: "special_tokens_map.json".to_string(),
        },
    );

    model_files.download()
}
