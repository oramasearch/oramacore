use anyhow::Result;
use dashmap::DashMap;
use text_splitter::{Characters, ChunkConfig, CodeSplitter, MarkdownSplitter, TextSplitter};
use tiktoken_rs::*;

use crate::code_parser::CodeLanguage;

pub struct Chunker {
    max_tokens: usize,
    text_splitter: TextSplitter<CoreBPE>,
    code_splitters: DashMap<CodeLanguage, CodeSplitter<Characters>>,
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
            code_splitters: DashMap::new(),
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

    pub fn chunk_code(&self, text: &str, language: CodeLanguage) -> Vec<String> {
        let entry = self.code_splitters.entry(language);

        let code_splitter = entry.or_insert_with(|| {
            let lang = match language {
                CodeLanguage::JavaScript => tree_sitter_javascript::LANGUAGE,
                CodeLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
                CodeLanguage::TSX => tree_sitter_typescript::LANGUAGE_TSX,
                CodeLanguage::HTML => tree_sitter_html::LANGUAGE,
            };

            CodeSplitter::new(lang, self.max_tokens)
                .expect("Unable to create CodeSplitter instance")
        });

        code_splitter
            .chunks(text)
            .map(|chunk| chunk.to_string())
            .collect()
    }
}
