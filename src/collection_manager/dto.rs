use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    code_parser::CodeLanguage, embeddings::OramaModel, indexes::number::{Number, NumberFilter}, nlp::locales::Locale, types::Document
};

use super::collection::FieldId;

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
pub struct EmbeddingTypedField {
    pub model_name: OramaModel,
    pub document_fields: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TypedField {
    Text(LanguageDTO),
    Code(CodeLanguage),
    Embedding(EmbeddingTypedField),
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
    pub code_fields: HashMap<String, FieldId>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Limit(pub usize);
impl Default for Limit {
    fn default() -> Self {
        Limit(10)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Filter {
    Number(NumberFilter),
    Bool(bool),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NumberFacetDefinitionRange {
    pub from: Number,
    pub to: Number,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NumberFacetDefinition {
    pub ranges: Vec<NumberFacetDefinitionRange>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum FacetDefinition {
    Number(NumberFacetDefinition),
    Bool,
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
    #[serde(default, rename = "where")]
    pub where_filter: HashMap<String, Filter>,
    #[serde(default)]
    pub facets: HashMap<String, FacetDefinition>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultHit {
    pub id: String,
    pub score: f32,
    pub document: Option<Document>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetResult {
    pub count: usize,
    pub values: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub hits: Vec<SearchResultHit>,
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facets: Option<HashMap<String, FacetResult>>,
}
