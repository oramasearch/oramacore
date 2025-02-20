use std::collections::HashMap;

use axum_openapi3::utoipa::{self, IntoParams};
use axum_openapi3::utoipa::{PartialSchema, ToSchema};
use redact::Secret;
use serde::{de, Deserialize, Serialize};

use crate::{
    nlp::locales::Locale,
    types::{CollectionId, DocumentId, RawJSONDocument, ValueType},
};

mod bm25;
mod global_info;
mod number;

use super::sides::hooks::HookName;
use super::sides::OramaModelSerializable;
pub use bm25::*;
pub use global_info::*;
pub use number::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldId(pub u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScore {
    pub document_id: DocumentId,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, ToSchema, PartialEq, Eq)]
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

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, PartialEq, Eq)]
#[serde(untagged)]
pub enum DocumentFields {
    Properties(Vec<String>),
    Hook(HookName),
    AllStringProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTypedField {
    pub model: OramaModelSerializable,
    pub document_fields: DocumentFields,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypedField {
    Text(Locale),
    Embedding(EmbeddingTypedField),
    Number,
    Bool,
    ArrayText(Locale),
    ArrayNumber,
    ArrayBoolean,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateCollectionEmbeddings {
    pub model: Option<OramaModelSerializable>,
    pub document_fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiKey(pub Secret<String>);

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D>(deserializer: D) -> Result<ApiKey, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        if s.is_empty() {
            return Err(serde::de::Error::custom("API key cannot be empty"));
        }

        Ok(ApiKey(Secret::new(s)))
    }
}

impl PartialSchema for ApiKey {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        String::schema()
    }
}
impl ToSchema for ApiKey {}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateCollection {
    pub id: CollectionId,
    pub description: Option<String>,

    pub read_api_key: ApiKey,
    pub write_api_key: ApiKey,

    #[schema(inline)]
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    #[schema(inline)]
    pub embeddings: Option<CreateCollectionEmbeddings>,
}

impl TryFrom<serde_json::Value> for CreateCollection {
    type Error = anyhow::Error;

    fn try_from(value: serde_json::Value) -> anyhow::Result<Self> {
        let v = serde_json::from_value(value)?;
        Ok(v)
    }
}

pub type DeleteDocuments = Vec<String>;

#[derive(Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct CollectionDTO {
    #[schema(inline)]
    pub id: CollectionId,
    pub description: Option<String>,
    pub document_count: u64,
    #[schema(inline)]
    pub fields: HashMap<String, ValueType>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Copy, Clone)]
pub struct Limit(#[schema(inline)] pub usize);
impl Default for Limit {
    fn default() -> Self {
        Limit(10)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
#[serde(untagged)]
pub enum Filter {
    Number(#[schema(inline)] NumberFilter),
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct NumberFacetDefinitionRange {
    #[schema(inline)]
    pub from: Number,
    #[schema(inline)]
    pub to: Number,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct NumberFacetDefinition {
    #[schema(inline)]
    pub ranges: Vec<NumberFacetDefinitionRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct BoolFacetDefinition {
    #[serde(rename = "true")]
    pub r#true: bool,
    #[serde(rename = "false")]
    pub r#false: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]

pub enum FacetDefinition {
    #[serde(untagged)]
    Number(#[schema(inline)] NumberFacetDefinition),
    #[serde(untagged)]
    Bool(#[schema(inline)] BoolFacetDefinition),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FulltextMode {
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct VectorMode {
    // In Orama previously we support 2 kind:
    // - "term": "hello"
    // - "vector": [...]
    // For simplicity, we only support "term" for now
    // TODO: support "vector"
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct HybridMode {
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AutoMode {
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "mode")]
pub enum SearchMode {
    #[serde(rename = "fulltext")]
    FullText(#[schema(inline)] FulltextMode),
    #[serde(rename = "vector")]
    Vector(#[schema(inline)] VectorMode),
    #[serde(rename = "hybrid")]
    Hybrid(#[schema(inline)] HybridMode),
    #[serde(rename = "auto")]
    Auto(#[schema(inline)] AutoMode),
    #[serde(untagged)]
    Default(#[schema(inline)] FulltextMode),
}
impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Default(FulltextMode {
            term: "".to_string(),
        })
    }
}
impl SearchMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchMode::FullText(_) => "fulltext",
            SearchMode::Vector(_) => "vector",
            SearchMode::Hybrid(_) => "hybrid",
            SearchMode::Auto(_) => "auto",
            SearchMode::Default(_) => "fulltext",
        }
    }

    pub fn from_str(s: &str, term: String) -> Self {
        match s {
            "fulltext" => SearchMode::FullText(FulltextMode { term }),
            "vector" => SearchMode::Vector(VectorMode { term }),
            "hybrid" => SearchMode::Hybrid(HybridMode { term }),
            "auto" => SearchMode::Auto(AutoMode { term }),
            _ => SearchMode::Default(FulltextMode { term }),
        }
    }
}

#[derive(Debug, Clone, ToSchema, PartialEq)]
pub enum Properties {
    None,
    Star,
    Specified(Vec<String>),
}

impl Default for Properties {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct SearchParams {
    #[serde(flatten)]
    #[schema(inline)]
    pub mode: SearchMode,
    #[serde(default)]
    #[schema(inline)]
    pub limit: Limit,
    #[serde(default)]
    pub boost: HashMap<String, f32>,
    #[serde(default, deserialize_with = "deserialize_json_string")]
    #[schema(inline)]
    pub properties: Properties,
    #[serde(default, rename = "where")]
    #[schema(inline)]
    pub where_filter: HashMap<String, Filter>,
    #[serde(default)]
    #[schema(inline)]
    pub facets: HashMap<String, FacetDefinition>,
}

fn deserialize_json_string<'de, D>(deserializer: D) -> Result<Properties, D::Error>
where
    D: de::Deserializer<'de>,
{
    // define a visitor that deserializes
    // `ActualData` encoded as json within a string
    struct PropertiesVisitor;

    impl<'de> de::Visitor<'de> for PropertiesVisitor {
        type Value = Properties;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("Only '*' is supported or an array of strings")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            match v {
                "*" => Ok(Properties::Star),
                _ => Err(E::custom(
                    "Invalid string. only '*' is supported or an array of strings",
                )),
            }
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut v: Vec<String> = Vec::new();
            while let Some(p) = seq.next_element::<String>()? {
                v.push(p);
            }
            Ok(Properties::Specified(v))
        }
    }

    // use our visitor to deserialize an `ActualValue`
    deserializer.deserialize_any(PropertiesVisitor)
}

impl TryFrom<serde_json::Value> for SearchParams {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SearchResultHit {
    pub id: String,
    pub score: f32,
    pub document: Option<RawJSONDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct FacetResult {
    pub count: usize,
    pub values: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SearchResult {
    pub hits: Vec<SearchResultHit>,
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub facets: Option<HashMap<String, FacetResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum RelatedQueriesFormat {
    Question,
    Query,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RelatedRequest {
    enabled: Option<bool>,
    size: Option<usize>,
    format: Option<RelatedQueriesFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum Role {
    System,
    Assistant,
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InteractionMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Interaction {
    pub interaction_id: String,
    pub query: String,
    pub visitor_id: String,
    pub conversation_id: String,
    pub related: Option<RelatedRequest>,
    pub messages: Vec<InteractionMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct NewHookPostParams {
    #[schema(inline)]
    pub name: HookName,
    pub code: String,
}

#[derive(Deserialize, Clone, Serialize, ToSchema, IntoParams)]
pub struct GetHookQueryParams {
    #[schema(inline)]
    pub name: HookName,
}

#[derive(Deserialize, Clone, Serialize, ToSchema, IntoParams)]
pub struct DeleteHookParams {
    #[schema(inline)]
    pub name: HookName,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
pub struct ExecuteActionPayload {
    pub name: String, // we're not using an enum here since users will be able to define their own actions
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InsertSegmentParams {
    pub id: Option<String>,
    pub name: String,
    pub description: String,
    pub goal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteSegmentParams {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateSegmentParams {
    pub id: String,
    pub name: String,
    pub description: String,
    pub goal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InsertTriggerParams {
    pub id: Option<String>,
    pub name: String,
    pub description: String,
    pub response: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteTriggerParams {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateTriggerParams {
    pub id: String,
    pub name: String,
    pub description: String,
    pub response: String,
}

#[cfg(test)]
mod test {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_search_params_mode_deserialization() {
        let j = json!({
            "mode": "fulltext",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert!(matches!(p.mode, SearchMode::FullText(_)));

        let j = json!({
            "mode": "vector",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert!(matches!(p.mode, SearchMode::Vector(_)));

        let j = json!({
            "mode": "hybrid",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert!(matches!(p.mode, SearchMode::Hybrid(_)));

        let j = json!({
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert!(matches!(p.mode, SearchMode::Default(_)));

        let j = json!({
            "mode": "unknown_value",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert!(matches!(p.mode, SearchMode::Default(_)));
    }

    #[test]
    fn test_create_collection_option_dto_serialization() {
        let _: CreateCollection = json!({
            "id": "foo",
            "typed_fields": {
                "vector": {
                    "mode": "embedding",
                    "model_name": "gte-small",
                    "document_fields": ["text"],
                }
            },
            "read_api_key": "foo",
            "write_api_key": "bar",
        })
        .try_into()
        .unwrap();
    }

    #[test]
    fn test_search_params_properties_deserialization() {
        let j = json!({
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert_eq!(p.properties, Properties::None);

        let j = json!({
            "properties": ["p1", "p2"],
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert_eq!(
            p.properties,
            Properties::Specified(vec!["p1".to_string(), "p2".to_string(),])
        );

        let j = json!({
            "properties": "*",
            "term": "hello",
        });
        let p = serde_json::from_value::<SearchParams>(j).unwrap();
        assert_eq!(p.properties, Properties::Star);
    }
}
