use std::collections::HashMap;

use axum_openapi3::utoipa::{self, IntoParams};
use axum_openapi3::utoipa::{PartialSchema, ToSchema};
use redact::Secret;
use serde::de::{Unexpected, Visitor};
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
    #[serde(rename = "arabic")]
    Arabic,
    #[serde(rename = "bulgarian")]
    Bulgarian,
    #[serde(rename = "danish")]
    Danish,
    #[serde(rename = "german")]
    German,
    #[serde(rename = "greek")]
    Greek,
    #[serde(rename = "english")]
    English,
    #[serde(rename = "estonian")]
    Estonian,
    #[serde(rename = "spanish")]
    Spanish,
    #[serde(rename = "finnish")]
    Finnish,
    #[serde(rename = "french")]
    French,
    #[serde(rename = "irish")]
    Irish,
    #[serde(rename = "hindi")]
    Hindi,
    #[serde(rename = "hungarian")]
    Hungarian,
    #[serde(rename = "armenian")]
    Armenian,
    #[serde(rename = "indonesian")]
    Indonesian,
    #[serde(rename = "italian")]
    Italian,
    #[serde(rename = "japanese")]
    Japanese,
    #[serde(rename = "korean")]
    Korean,
    #[serde(rename = "lithuanian")]
    Lithuanian,
    #[serde(rename = "nepali")]
    Nepali,
    #[serde(rename = "dutch")]
    Dutch,
    #[serde(rename = "norwegian")]
    Norwegian,
    #[serde(rename = "portuguese")]
    Portuguese,
    #[serde(rename = "romanian")]
    Romanian,
    #[serde(rename = "russian")]
    Russian,
    #[serde(rename = "sanskrit")]
    Sanskrit,
    #[serde(rename = "slovenian")]
    Slovenian,
    #[serde(rename = "serbian")]
    Serbian,
    #[serde(rename = "swedish")]
    Swedish,
    #[serde(rename = "tamil")]
    Tamil,
    #[serde(rename = "turkish")]
    Turkish,
    #[serde(rename = "ukrainian")]
    Ukrainian,
    #[serde(rename = "chinese")]
    Chinese,
}

impl From<LanguageDTO> for Locale {
    fn from(language: LanguageDTO) -> Self {
        match language {
            LanguageDTO::English => Locale::EN,
            LanguageDTO::Italian => Locale::IT,
            LanguageDTO::Spanish => Locale::ES,
            LanguageDTO::French => Locale::FR,
            LanguageDTO::German => Locale::DE,
            LanguageDTO::Portuguese => Locale::PT,
            LanguageDTO::Dutch => Locale::NL,
            LanguageDTO::Russian => Locale::RU,
            LanguageDTO::Chinese => Locale::ZH,
            LanguageDTO::Korean => Locale::KO,
            LanguageDTO::Arabic => Locale::AR,
            LanguageDTO::Bulgarian => Locale::BG,
            LanguageDTO::Danish => Locale::DA,
            LanguageDTO::Greek => Locale::EL,
            LanguageDTO::Estonian => Locale::ET,
            LanguageDTO::Finnish => Locale::FI,
            LanguageDTO::Irish => Locale::GA,
            LanguageDTO::Hindi => Locale::HI,
            LanguageDTO::Hungarian => Locale::HU,
            LanguageDTO::Armenian => Locale::HY,
            LanguageDTO::Indonesian => Locale::ID,
            LanguageDTO::Lithuanian => Locale::LT,
            LanguageDTO::Nepali => Locale::NE,
            LanguageDTO::Norwegian => Locale::NO,
            LanguageDTO::Romanian => Locale::RO,
            LanguageDTO::Sanskrit => Locale::SA,
            LanguageDTO::Slovenian => Locale::SL,
            LanguageDTO::Serbian => Locale::SR,
            LanguageDTO::Swedish => Locale::SV,
            LanguageDTO::Tamil => Locale::TA,
            LanguageDTO::Turkish => Locale::TR,
            LanguageDTO::Ukrainian => Locale::UK,
            LanguageDTO::Japanese => Locale::JP,
        }
    }
}
impl From<Locale> for LanguageDTO {
    fn from(language: Locale) -> Self {
        match language {
            Locale::EN => LanguageDTO::English,
            Locale::IT => LanguageDTO::Italian,
            Locale::ES => LanguageDTO::Spanish,
            Locale::FR => LanguageDTO::French,
            Locale::DE => LanguageDTO::German,
            Locale::PT => LanguageDTO::Portuguese,
            Locale::NL => LanguageDTO::Dutch,
            Locale::RU => LanguageDTO::Russian,
            Locale::ZH => LanguageDTO::Chinese,
            Locale::KO => LanguageDTO::Korean,
            Locale::AR => LanguageDTO::Arabic,
            Locale::BG => LanguageDTO::Bulgarian,
            Locale::DA => LanguageDTO::Danish,
            Locale::EL => LanguageDTO::Greek,
            Locale::ET => LanguageDTO::Estonian,
            Locale::FI => LanguageDTO::Finnish,
            Locale::GA => LanguageDTO::Irish,
            Locale::HI => LanguageDTO::Hindi,
            Locale::HU => LanguageDTO::Hungarian,
            Locale::HY => LanguageDTO::Armenian,
            Locale::ID => LanguageDTO::Indonesian,
            Locale::LT => LanguageDTO::Lithuanian,
            Locale::NE => LanguageDTO::Nepali,
            Locale::NO => LanguageDTO::Norwegian,
            Locale::RO => LanguageDTO::Romanian,
            Locale::SA => LanguageDTO::Sanskrit,
            Locale::SL => LanguageDTO::Slovenian,
            Locale::SR => LanguageDTO::Serbian,
            Locale::SV => LanguageDTO::Swedish,
            Locale::TA => LanguageDTO::Tamil,
            Locale::TR => LanguageDTO::Turkish,
            Locale::UK => LanguageDTO::Ukrainian,
            Locale::JP => LanguageDTO::Japanese,
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

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
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

use crate::collection_manager::sides::{deserialize_api_key, serialize_api_key};
#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct CreateCollection {
    pub id: CollectionId,
    pub description: Option<String>,

    #[serde(
        deserialize_with = "deserialize_api_key",
        serialize_with = "serialize_api_key"
    )]
    pub read_api_key: ApiKey,
    #[serde(
        deserialize_with = "deserialize_api_key",
        serialize_with = "serialize_api_key"
    )]
    pub write_api_key: ApiKey,

    #[schema(inline)]
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    #[schema(inline)]
    pub embeddings: Option<CreateCollectionEmbeddings>,
}

#[derive(Debug, Deserialize, ToSchema, Clone)]
pub struct ReindexConfig {
    #[serde(default)]
    pub description: Option<String>,

    #[serde(default)]
    #[schema(inline)]
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    #[schema(inline)]
    pub embeddings: Option<CreateCollectionEmbeddings>,
}

#[derive(Debug, Deserialize, ToSchema, Clone)]
pub struct CreateCollectionFrom {
    pub r#from: CollectionId,

    #[serde(default)]
    #[schema(inline)]
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    #[schema(inline)]
    pub embeddings: Option<CreateCollectionEmbeddings>,
}

#[derive(Debug, Deserialize, ToSchema, Clone)]
pub struct SwapCollections {
    pub from: CollectionId,
    pub to: CollectionId,
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

#[derive(Debug, Clone, ToSchema)]
pub struct FulltextMode {
    pub term: String,
}

#[derive(Debug, Clone, ToSchema)]
pub struct VectorMode {
    // In Orama previously we support 2 kind:
    // - "term": "hello"
    // - "vector": [...]
    // For simplicity, we only support "term" for now
    // TODO: support "vector"
    pub term: String,
    #[schema(inline)]
    pub similarity: Similarity,
}

#[derive(Debug, Clone, ToSchema)]
pub struct Similarity(pub f32);

impl Default for Similarity {
    fn default() -> Self {
        Similarity(0.8)
    }
}

impl<'de> Deserialize<'de> for Similarity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct SimilarityVisitor;

        fn check_similarity<E>(value: f32) -> Result<f32, E>
        where
            E: de::Error,
        {
            if !(0.0..=1.0).contains(&value) {
                return Err(de::Error::custom("the value must be between 0.0 and 1.0"));
            }
            Ok(value)
        }

        impl<'de> Visitor<'de> for SimilarityVisitor {
            type Value = Similarity;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a valid similarity value")
            }

            fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let f = f32::from(value);
                check_similarity(f).map(Similarity)
            }

            fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                check_similarity(value).map(Similarity)
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                check_similarity(value as f32).map(Similarity)
            }

            fn visit_map<V>(self, mut visitor: V) -> Result<Self::Value, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                println!("visit_map.....");
                let k: Option<String> = visitor
                    .next_key()
                    .map_err(|e| de::Error::custom(format!("error: {}", e)))?;
                if k.is_none() {
                    return Err(de::Error::invalid_type(Unexpected::Map, &self));
                }
                let f: String = visitor
                    .next_value()
                    .map_err(|e| de::Error::custom(format!("error: {}", e)))?;
                let f: f32 = f
                    .parse()
                    .map_err(|e| de::Error::custom(format!("error: {}", e)))?;

                check_similarity(f).map(Similarity)
            }
        }

        deserializer.deserialize_any(SimilarityVisitor)
    }
}

#[derive(Debug, Clone, ToSchema)]
pub struct HybridMode {
    pub term: String,
    pub similarity: Similarity,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AutoMode {
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SearchModeResult {
    pub mode: String,
}

#[derive(Debug, Clone, ToSchema)]
pub enum SearchMode {
    FullText(#[schema(inline)] FulltextMode),
    Vector(#[schema(inline)] VectorMode),
    Hybrid(#[schema(inline)] HybridMode),
    Auto(#[schema(inline)] AutoMode),
    Default(#[schema(inline)] FulltextMode),
}

impl<'de> Deserialize<'de> for SearchMode {
    fn deserialize<D>(deserializer: D) -> Result<SearchMode, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        fn default_search_mode() -> String {
            "default".to_string()
        }

        #[derive(Deserialize)]
        struct HiddenSearchMode {
            #[serde(default = "default_search_mode")]
            mode: String,
            term: String,
            similarity: Option<Similarity>,
        }

        let mode = match HiddenSearchMode::deserialize(deserializer) {
            Ok(mode) => mode,
            Err(e) => {
                return Err(e);
            }
        };

        match mode.mode.as_str() {
            "fulltext" => Ok(SearchMode::FullText(FulltextMode { term: mode.term })),
            "vector" => Ok(SearchMode::Vector(VectorMode {
                term: mode.term,
                similarity: mode.similarity.unwrap_or_default(),
            })),
            "hybrid" => Ok(SearchMode::Hybrid(HybridMode {
                term: mode.term,
                similarity: mode.similarity.unwrap_or_default(),
            })),
            "default" => Ok(SearchMode::Default(FulltextMode { term: mode.term })),
            "auto" => Ok(SearchMode::Auto(AutoMode { term: mode.term })),
            m => Err(serde::de::Error::custom(format!(
                "Invalid search mode: {}",
                m
            ))),
        }
    }
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
            "vector" => SearchMode::Vector(VectorMode {
                similarity: Similarity(0.8),
                term,
            }),
            "hybrid" => SearchMode::Hybrid(HybridMode {
                similarity: Similarity(0.8),
                term,
            }),
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
    #[serde(default, deserialize_with = "deserialize_properties")]
    #[schema(inline)]
    pub properties: Properties,
    #[serde(default, rename = "where")]
    #[schema(inline)]
    pub where_filter: HashMap<String, Filter>,
    #[serde(default)]
    #[schema(inline)]
    pub facets: HashMap<String, FacetDefinition>,
}

fn deserialize_properties<'de, D>(deserializer: D) -> Result<Properties, D::Error>
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

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, Eq, PartialEq, Copy)]
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
    pub name: String,
    pub description: String,
    pub response: String,
    pub segment_id: Option<String>,
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
    pub segment_id: Option<String>,
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
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap();
        assert!(matches!(p.mode, SearchMode::FullText(_)));

        let j = json!({
            "mode": "vector",
            "term": "hello",
        });
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap();
        assert!(matches!(p.mode, SearchMode::Vector(_)));

        let j = json!({
            "mode": "hybrid",
            "term": "hello",
        });
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap();
        assert!(matches!(p.mode, SearchMode::Hybrid(_)));

        let j = json!({
            "term": "hello",
        });
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap();
        assert!(matches!(p.mode, SearchMode::Default(_)));

        let j = json!({
            "mode": "unknown_value",
            "term": "hello",
        });
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap_err();
        assert!(format!("{}", p).contains("Invalid search mode: unknown_value"));

        let j = json!({
            "mode": "vector",
            "term": "The feline is napping comfortably indoors.",
            "similarity": 0.6,
        });
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap();
        assert!(matches!(p.mode, SearchMode::Vector(v) if (v.similarity.0 - 0.6).abs() < 0.01));
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
