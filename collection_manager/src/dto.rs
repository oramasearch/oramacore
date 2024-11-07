use std::collections::HashMap;

use nlp::locales::Locale;
use serde::{Deserialize, Serialize};
use types::{CodeLanguage, FieldId};

#[derive(Debug, Serialize, Deserialize)]
pub enum LanguageDTO {
    English,
}

impl From<LanguageDTO> for Locale {
    fn from(language: LanguageDTO) -> Self {
        match language {
            LanguageDTO::English => Locale::EN,
        }
    }
}
impl From<Locale> for LanguageDTO {
    fn from(language: Locale) -> Self {
        match language {
            Locale::EN => LanguageDTO::English,
            _ => LanguageDTO::English,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TypedField {
    Text(LanguageDTO),
    Code(CodeLanguage),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SearchMode {
    FullText,
    Vector,
    Hybrid,
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::FullText
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateCollectionOptionDTO {
    pub id: String,
    pub description: Option<String>,
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    pub typed_fields: HashMap<String, TypedField>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CollectionDTO {
    pub id: String,
    pub description: Option<String>,
    pub language: LanguageDTO,
    pub document_count: usize,
    pub string_fields: HashMap<String, FieldId>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Limit(pub usize);
impl Default for Limit {
    fn default() -> Self {
        Limit(10)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchParams {
    pub term: String,
    #[serde(default)]
    pub limit: Limit,
    #[serde(default)]
    pub boost: HashMap<String, f32>,
    #[serde(default)]
    pub properties: Option<Vec<String>>,
    #[serde(default)]
    pub mode: SearchMode,
}
