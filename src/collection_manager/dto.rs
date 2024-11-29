use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    code_parser::CodeLanguage,
    document_storage::DocumentId,
    embeddings::OramaModel,
    indexes::number::{Number, NumberFilter},
    nlp::locales::Locale,
    types::Document,
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldId(pub u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScore {
    pub document_id: DocumentId,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EmbeddingTypedField {
    pub model_name: OramaModel,
    pub document_fields: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum TypedField {
    Text(LanguageDTO),
    Code(CodeLanguage),
    Embedding(EmbeddingTypedField),
    Number,
    Bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateCollectionOptionDTO {
    pub id: String,
    pub description: Option<String>,
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    pub typed_fields: HashMap<String, TypedField>,
}

impl TryFrom<serde_json::Value> for CreateCollectionOptionDTO {
    type Error = anyhow::Error;

    fn try_from(value: serde_json::Value) -> anyhow::Result<Self> {
        let v = serde_json::from_value(value)?;
        Ok(v)
    }
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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
pub struct FulltextMode {
    pub term: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorMode {
    // In Orama previously we support 2 kind:
    // - "term": "hello"
    // - "vector": [...]
    // For simplicity, we only support "term" for now
    // TODO: support "vector"
    pub term: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HybridMode {}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SearchMode {
    #[serde(rename = "fulltext")]
    FullText(FulltextMode),
    #[serde(rename = "vector")]
    Vector(VectorMode),
    #[serde(rename = "hybrid")]
    Hybrid(HybridMode),
    #[serde(untagged)]
    Default(FulltextMode),
}
impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Default(FulltextMode {
            term: "".to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchParams {
    #[serde(flatten, rename = "type")]
    pub mode: SearchMode,
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

impl TryFrom<serde_json::Value> for SearchParams {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        println!("value: {:?}", value);
        let a = serde_json::from_value(value);

        println!("a: {:?}", a);

        a
    }
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

#[cfg(test)]
mod test {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_search_deserialization() {
        let j = json!({
            "type": "fulltext",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        matches!(p.mode, SearchMode::FullText(_));

        let j = json!({
            "type": "vector",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        matches!(p.mode, SearchMode::Vector(_));

        let j = json!({
            "type": "hybrid",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        matches!(p.mode, SearchMode::Hybrid(_));

        let j = json!({
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        matches!(p.mode, SearchMode::Default(_));

        let j = json!({
            "type": "unknown_value",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        matches!(p.mode, SearchMode::Default(_));
    }
}
