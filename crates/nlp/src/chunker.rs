use anyhow::Result;
use text_splitter::{Characters, ChunkConfig, MarkdownSplitter, TextSplitter};
use tiktoken_rs::*;

pub struct Chunker {
    #[allow(dead_code)]
    max_tokens: usize,
    text_splitter: TextSplitter<CoreBPE>,
    markdown_splitter: MarkdownSplitter<Characters>,
}

pub struct ChunkerConfig {
    pub max_tokens: usize,
    pub overlap: Option<usize>,
}

impl Chunker {
    pub fn try_new(config: ChunkerConfig) -> Result<Self> {
        let tokenizer = cl100k_base()?;

        let overlap = config.overlap.unwrap_or_default();

        let text_tokenizer_config = ChunkConfig::new(config.max_tokens)
            .with_sizer(tokenizer)
            .with_overlap(overlap)?;

        Ok(Chunker {
            max_tokens: config.max_tokens,
            text_splitter: TextSplitter::new(text_tokenizer_config),
            markdown_splitter: MarkdownSplitter::new(config.max_tokens),
        })
    }

    pub fn chunk_text(&self, text: &str) -> Vec<String> {
        self.text_splitter
            .chunks(text)
            .map(|chunk| chunk.to_string())
            .collect()
    }

    pub fn chunk_markdown(&self, text: &str) -> Vec<String> {
        self.markdown_splitter
            .chunks(text)
            .map(|chunk| chunk.to_string())
            .collect()
    }
}
