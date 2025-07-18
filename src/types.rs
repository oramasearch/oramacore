use crate::ai::{OramaModel, RemoteLLMProvider};

use crate::ai::automatic_embeddings_selector::ChosenProperties;

use crate::collection_manager::sides::write::index::{FieldType, GeoPoint};
use crate::collection_manager::sides::write::OramaModelSerializable;
use crate::collection_manager::sides::{deserialize_api_key, serialize_api_key};
use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use axum_openapi3::utoipa::openapi::schema::AnyOfBuilder;
use axum_openapi3::utoipa::{self, PartialSchema, ToSchema};
use nlp::locales::Locale;

use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
};

use chrono::{DateTime, Utc};
use redact::Secret;
use serde::de::{Error, Visitor};
use serde::{de, Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RawJSONDocument {
    pub id: Option<String>,
    pub inner: Box<serde_json::value::RawValue>,
}

#[cfg(test)]
impl PartialEq for RawJSONDocument {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.inner.get() == other.inner.get()
    }
}

impl TryFrom<serde_json::Value> for RawJSONDocument {
    type Error = anyhow::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let inner = serde_json::value::to_raw_value(&value).context("Cannot serialize document")?;
        Ok(RawJSONDocument { inner, id })
    }
}

impl Serialize for RawJSONDocument {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        self.inner.serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for RawJSONDocument {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WithId {
            id: String,
        }

        Box::<serde_json::value::RawValue>::deserialize(deserializer).map(|inner| {
            let parsed = serde_json::from_str::<WithId>(inner.get()).unwrap();
            RawJSONDocument {
                inner,
                id: Some(parsed.id),
            }
        })
    }
}

#[cfg(test)]
impl RawJSONDocument {
    pub fn get(&self, key: &str) -> Option<Value> {
        let value: Value = serde_json::from_str(self.inner.get()).ok()?;
        value.get(key).cloned()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy, Serialize, Deserialize, ToSchema)]
pub struct CollectionId(StackString<128>);
impl CollectionId {
    pub fn try_new<A: AsRef<str>>(key: A) -> Result<Self> {
        StackString::try_new(key)
            .map(CollectionId)
            .context("CollectionId is too long. Max 128 char")
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}
impl Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct DocumentId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    #[serde(flatten)]
    pub inner: Map<String, Value>,
}
impl Document {
    pub fn into_flatten(&self) -> FlattenDocument {
        let inner: Map<String, Value> = self
            .inner
            .iter()
            .flat_map(|(key, value)| match value {
                Value::Object(map) => map
                    .into_iter()
                    .map(|(sub_key, sub_value)| (format!("{key}.{sub_key}"), sub_value.clone()))
                    .collect::<Map<_, _>>(),
                _ => {
                    let mut map = Map::new();
                    map.insert(key.clone(), value.clone());
                    map
                }
            })
            .collect();
        FlattenDocument(inner)
    }

    pub fn into_raw(&self, doc_id_str: String) -> Result<RawJSONDocument> {
        let inner =
            serde_json::value::to_raw_value(&self.inner).context("Cannot serialize document")?;
        Ok(RawJSONDocument {
            inner,
            id: Some(doc_id_str),
        })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.inner.get(key)
    }
}

impl From<Map<String, Value>> for Document {
    fn from(map: Map<String, Value>) -> Self {
        Document { inner: map }
    }
}
impl TryFrom<Value> for Document {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Object(map) => Ok(Document { inner: map }),
            _ => Err(anyhow::anyhow!("expected object")),
        }
    }
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub enum UpdateStrategy {
    #[serde(rename = "merge")]
    Merge,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct UpdateDocumentRequest {
    pub strategy: UpdateStrategy,
    pub documents: DocumentList,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum ScalarType {
    String,
    Number,
    Boolean,
}
impl TryFrom<&Value> for ScalarType {
    type Error = anyhow::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::String(_) => Ok(ScalarType::String),
            Value::Number(_) => Ok(ScalarType::Number),
            Value::Bool(_) => Ok(ScalarType::Boolean),
            _ => Err(anyhow::anyhow!("expected scalar")),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum ComplexType {
    Array(ScalarType),
    Embedding,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub enum ValueType {
    Scalar(ScalarType),
    Complex(ComplexType),
}

impl TryFrom<&Value> for ValueType {
    type Error = anyhow::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Null => Err(anyhow::anyhow!("null value is not mapped")),
            Value::String(_) => Ok(ValueType::Scalar(ScalarType::String)),
            Value::Number(_) => Ok(ValueType::Scalar(ScalarType::Number)),
            Value::Bool(_) => Ok(ValueType::Scalar(ScalarType::Boolean)),
            Value::Array(array) => {
                let first = match array.first() {
                    Some(value) => value,
                    // Empty array: we can't infer the type
                    None => return Err(anyhow::anyhow!("no element in the array")),
                };
                let scalar_type = ScalarType::try_from(first)?;

                for value in array.iter() {
                    let value_type = ScalarType::try_from(value)?;
                    if value_type != scalar_type {
                        return Err(anyhow::anyhow!(
                            "expected array of scalars of the same type"
                        ));
                    }
                }

                Ok(ValueType::Complex(ComplexType::Array(scalar_type)))
            }
            Value::Object(_) => Err(anyhow::anyhow!(
                "object value is not mapped: flat the object before"
            )),
        }
    }
}

#[derive(Debug)]
pub struct Schema(HashMap<String, ValueType>);
impl IntoIterator for Schema {
    type Item = (String, ValueType);
    type IntoIter = std::collections::hash_map::IntoIter<String, ValueType>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Schema {
    pub fn inner(&self) -> &HashMap<String, ValueType> {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FlattenDocument(Map<String, Value>);

impl FlattenDocument {
    pub fn get_field_schema(&self) -> Schema {
        let inner: HashMap<String, ValueType> = self
            .0
            .iter()
            .filter_map(|(key, value)| {
                let key = key.clone();
                let t: ValueType = value.try_into().ok()?;
                Some((key, t))
            })
            .collect();

        Schema(inner)
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.0.remove(key)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    pub fn into_inner(self) -> Map<String, Value> {
        self.0
    }

    pub fn get_inner(&self) -> &Map<String, Value> {
        &self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.0.iter()
    }
}

#[derive(Debug, Deserialize)]
pub struct DocumentList(pub Vec<Document>);
impl DocumentList {
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl IntoIterator for DocumentList {
    type Item = Document;
    type IntoIter = std::vec::IntoIter<Document>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl PartialSchema for DocumentList {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        Value::schema()
    }
}
impl ToSchema for DocumentList {}

impl From<Vec<Document>> for DocumentList {
    fn from(docs: Vec<Document>) -> Self {
        DocumentList(docs)
    }
}

impl TryFrom<Value> for DocumentList {
    type Error = anyhow::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Array(array) => {
                let docs = array
                    .into_iter()
                    .map(Document::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(DocumentList(docs))
            }
            _ => Err(anyhow::anyhow!("expected array")),
        }
    }
}
impl TryFrom<Vec<Value>> for DocumentList {
    type Error = anyhow::Error;

    fn try_from(values: Vec<Value>) -> Result<Self, Self::Error> {
        let docs = values
            .into_iter()
            .map(Document::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DocumentList(docs))
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldId(pub u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScore {
    pub document_id: DocumentId,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, ToSchema)]
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum DocumentFields {
    Properties(Vec<String>),
    AllStringProperties,
    Automatic,
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

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct CreateCollectionEmbeddings {
    pub model: Option<OramaModelSerializable>,
    pub document_fields: Vec<String>,
}

impl PartialSchema for OramaModelSerializable {
    fn schema(
    ) -> axum_openapi3::utoipa::openapi::RefOr<axum_openapi3::utoipa::openapi::schema::Schema> {
        let b = AnyOfBuilder::new()
            .item(OramaModel::BgeSmall.as_str_name())
            .item(OramaModel::BgeBase.as_str_name())
            .item(OramaModel::BgeLarge.as_str_name())
            .item(OramaModel::MultilingualE5Small.as_str_name())
            .item(OramaModel::MultilingualE5Base.as_str_name())
            .item(OramaModel::MultilingualE5Large.as_str_name())
            .item(OramaModel::JinaEmbeddingsV2BaseCode.as_str_name());
        axum_openapi3::utoipa::openapi::RefOr::T(b.into())
    }
}
impl ToSchema for OramaModelSerializable {}

#[derive(Debug, Clone, PartialEq, Eq, Copy, Deserialize)]
pub struct ApiKey(Secret<StackString<64>>);
impl ApiKey {
    pub fn try_new<A: AsRef<str>>(key: A) -> Result<Self> {
        StackString::try_new(key.as_ref())
            .map(|s| ApiKey(Secret::new(s)))
            .context("API key is too long. Max 64 char")
    }

    pub fn expose(&self) -> &str {
        self.0.expose_secret().as_str()
    }
}
impl Serialize for ApiKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.expose())
    }
}
impl PartialSchema for ApiKey {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        StackString::<64>::schema()
    }
}
impl ToSchema for ApiKey {}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct ClaimLimits {
    pub max_doc_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Claims {
    pub sub: CollectionId,
    pub scope: StackString<20>,
    pub iss: StackString<128>,
    pub aud: StackString<128>,
    #[serde(rename = "lmt")]
    pub limits: ClaimLimits,
}

#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum WriteApiKey {
    ApiKey(ApiKey),
    Claims(Claims),
}

impl WriteApiKey {
    pub fn from_api_key(api_key: ApiKey) -> Self {
        Self::ApiKey(api_key)
    }

    pub fn from_claims(claims: Claims) -> Self {
        Self::Claims(claims)
    }
}

impl PartialSchema for WriteApiKey {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        ApiKey::schema()
    }
}

impl ToSchema for WriteApiKey {}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ListDocumentInCollectionRequest {
    pub id: CollectionId,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct DeleteCollection {
    #[serde(rename = "collection_id_to_delete")]
    pub id: CollectionId,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct DeleteIndex {
    #[serde(rename = "index_id_to_delete")]
    pub id: IndexId,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
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

    pub language: Option<LanguageDTO>,
    pub embeddings_model: Option<OramaModelSerializable>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct CollectionCreated {
    pub collection_id: CollectionId,
}

#[derive(Debug, Deserialize, Clone, ToSchema)]
pub struct ReindexConfig {
    pub language: LanguageDTO,
    pub embedding_model: OramaModelSerializable,
    pub reference: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CreateCollectionFrom {
    pub r#from: CollectionId,

    #[serde(default)]
    pub language: Option<LanguageDTO>,
    #[serde(default)]
    pub embeddings: Option<CreateCollectionEmbeddings>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SwapCollections {
    pub from: CollectionId,
    pub to: CollectionId,
    pub reference: Option<String>,
}

impl TryFrom<serde_json::Value> for CreateCollection {
    type Error = anyhow::Error;

    fn try_from(value: serde_json::Value) -> anyhow::Result<Self> {
        let v = serde_json::from_value(value)?;
        Ok(v)
    }
}

pub type DeleteDocuments = Vec<String>;

#[derive(Debug, Serialize)]
pub struct DescribeCollectionResponse {
    pub id: CollectionId,
    pub description: Option<String>,
    pub document_count: usize,
    pub created_at: DateTime<Utc>,

    pub indexes: Vec<DescribeCollectionIndexResponse>,
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub struct Limit(pub usize);
impl Default for Limit {
    fn default() -> Self {
        Self(10)
    }
}

#[derive(Debug, Serialize, Deserialize, Copy, Clone, Default)]
pub struct SearchOffset(pub usize);

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum Filter {
    Date(DateFilter),
    Number(NumberFilter),
    Bool(bool),
    String(String),
    GeoPoint(GeoSearchFilter),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFacetDefinitionRange {
    pub from: Number,

    pub to: Number,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumberFacetDefinition {
    pub ranges: Vec<NumberFacetDefinitionRange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoolFacetDefinition {
    #[serde(rename = "true")]
    pub r#true: bool,
    #[serde(rename = "false")]
    pub r#false: bool,
}

#[derive(Debug, Clone)]
pub struct StringFacetDefinition;
impl<'de> Deserialize<'de> for StringFacetDefinition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        struct StringFacetDefinitionVisitor;

        impl<'de> Visitor<'de> for StringFacetDefinitionVisitor {
            type Value = StringFacetDefinition;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string facet definition")
            }

            fn visit_map<V>(self, _visitor: V) -> Result<Self::Value, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                Ok(StringFacetDefinition)
            }
        }

        deserializer.deserialize_any(StringFacetDefinitionVisitor)
    }
}
impl Serialize for StringFacetDefinition {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeMap;
        let map = serializer.serialize_map(Some(0))?;
        map.end()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FacetDefinition {
    #[serde(untagged)]
    Number(NumberFacetDefinition),
    #[serde(untagged)]
    Bool(BoolFacetDefinition),
    #[serde(untagged)]
    String(StringFacetDefinition),
}

#[derive(Debug, Clone, Serialize)]
pub struct FulltextMode {
    pub term: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<Threshold>,
    pub exact: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorMode {
    // In Orama previously we support 2 kind:
    // - "term": "hello"
    // - "vector": [...]
    // For simplicity, we only support "term" for now
    // TODO: support "vector"
    pub term: String,

    pub similarity: Similarity,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Threshold(pub f32);

impl<'de> Deserialize<'de> for Threshold {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct HiddenThreshold(f32);

        let v = HiddenThreshold::deserialize(deserializer).map(|v| v.0)?;
        if !(0.0..=1.0).contains(&v) {
            return Err(de::Error::custom("the value must be between 0.0 and 1.0"));
        }
        Ok(Threshold(v))
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
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
        #[derive(Deserialize)]
        struct HiddenSimilarity(f32);

        let v = HiddenSimilarity::deserialize(deserializer).map(|v| v.0)?;
        if !(0.0..=1.0).contains(&v) {
            return Err(de::Error::custom("the value must be between 0.0 and 1.0"));
        }
        Ok(Similarity(v))
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HybridMode {
    pub term: String,
    pub similarity: Similarity,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<Threshold>,
    pub exact: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMode {
    pub term: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchModeResult {
    pub mode: String,
}

#[derive(Debug, Clone)]
pub enum SearchMode {
    FullText(FulltextMode),
    Vector(VectorMode),
    Hybrid(HybridMode),
    Auto(AutoMode),
    Default(FulltextMode),
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
            threshold: Option<Threshold>,
            exact: Option<bool>,
            tolerance: Option<u8>,
        }

        let mode = match HiddenSearchMode::deserialize(deserializer) {
            Ok(mode) => mode,
            Err(e) => {
                return Err(e);
            }
        };

        match mode.mode.as_str() {
            "fulltext" => Ok(SearchMode::FullText(FulltextMode {
                term: mode.term,
                threshold: mode.threshold,
                exact: mode.exact.unwrap_or(false),
                tolerance: mode.tolerance,
            })),
            "vector" => Ok(SearchMode::Vector(VectorMode {
                term: mode.term,
                similarity: mode.similarity.unwrap_or_default(),
            })),
            "hybrid" => Ok(SearchMode::Hybrid(HybridMode {
                term: mode.term,
                similarity: mode.similarity.unwrap_or_default(),
                threshold: mode.threshold,
                exact: mode.exact.unwrap_or(false),
                tolerance: mode.tolerance,
            })),
            "default" => Ok(SearchMode::Default(FulltextMode {
                term: mode.term,
                threshold: mode.threshold,
                exact: mode.exact.unwrap_or(false),
                tolerance: mode.tolerance,
            })),
            "auto" => Ok(SearchMode::Auto(AutoMode { term: mode.term })),
            m => Err(serde::de::Error::custom(format!(
                "Invalid search mode: {m}"
            ))),
        }
    }
}

impl Serialize for SearchMode {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeMap;
        let map = match self {
            SearchMode::FullText(FulltextMode {
                term,
                threshold,
                exact,
                tolerance,
            }) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("mode", "fulltext")?;
                map.serialize_entry("term", term)?;
                if let Some(threshold) = threshold {
                    map.serialize_entry("threshold", threshold)?;
                }
                if *exact {
                    map.serialize_entry("exact", exact)?;
                }
                if let Some(tolerance) = tolerance {
                    map.serialize_entry("tolerance", tolerance)?;
                }
                map
            }
            SearchMode::Vector(VectorMode { term, similarity }) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("mode", "vector")?;
                map.serialize_entry("term", term)?;
                if similarity.0 != 0.8 {
                    map.serialize_entry("similarity", similarity)?;
                }
                map
            }
            SearchMode::Hybrid(HybridMode {
                term,
                similarity,
                threshold,
                exact,
                tolerance,
            }) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("mode", "hybrid")?;
                map.serialize_entry("term", term)?;
                if similarity.0 != 0.8 {
                    map.serialize_entry("similarity", similarity)?;
                }
                if let Some(threshold) = threshold {
                    map.serialize_entry("threshold", threshold)?;
                }
                if *exact {
                    map.serialize_entry("exact", exact)?;
                }
                if let Some(tolerance) = tolerance {
                    map.serialize_entry("tolerance", tolerance)?;
                }
                map
            }
            SearchMode::Auto(AutoMode { term }) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("mode", "auto")?;
                map.serialize_entry("term", term)?;
                map
            }
            SearchMode::Default(FulltextMode {
                term,
                threshold,
                exact,
                tolerance,
            }) => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("mode", "default")?;
                map.serialize_entry("term", term)?;
                if let Some(threshold) = threshold {
                    map.serialize_entry("threshold", threshold)?;
                }
                if *exact {
                    map.serialize_entry("exact", exact)?;
                }
                if let Some(tolerance) = tolerance {
                    map.serialize_entry("tolerance", tolerance)?;
                }
                map
            }
        };
        map.end()
    }
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::Default(FulltextMode {
            term: "".to_string(),
            threshold: None,
            exact: false,
            tolerance: None,
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
            "fulltext" => SearchMode::FullText(FulltextMode {
                term,
                threshold: None,
                exact: false,
                tolerance: None,
            }),
            "vector" => SearchMode::Vector(VectorMode {
                similarity: Similarity(0.8),
                term,
            }),
            "hybrid" => SearchMode::Hybrid(HybridMode {
                similarity: Similarity(0.8),
                term,
                threshold: None,
                exact: false,
                tolerance: None,
            }),
            "auto" => SearchMode::Auto(AutoMode { term }),
            _ => SearchMode::Default(FulltextMode {
                term,
                threshold: None,
                exact: false,
                tolerance: None,
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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

impl Properties {
    fn is_none(&self) -> bool {
        matches!(self, Properties::None)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FilterOnField {
    #[serde(rename = "$key$")]
    pub field: String,
    pub filter: Filter,
}

#[derive(Debug, Clone, Default)]
pub struct WhereFilter {
    pub filter_on_fields: Vec<(String, Filter)>,
    pub and: Option<Vec<WhereFilter>>,
    pub or: Option<Vec<WhereFilter>>,
    pub not: Option<Box<WhereFilter>>,
}
impl<'de> Deserialize<'de> for WhereFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        #[derive(Deserialize, Debug)]
        #[serde(untagged)]
        enum Value {
            Filter(Filter),
            WhereFilter(Vec<WhereFilter>),
            NotWhereFilter(WhereFilter),
        }

        struct HiddenWhereFilterVisitor;

        impl<'de> de::Visitor<'de> for HiddenWhereFilterVisitor {
            type Value = WhereFilter;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a where filter")
            }

            fn visit_map<V>(self, mut visitor: V) -> Result<Self::Value, V::Error>
            where
                V: de::MapAccess<'de>,
            {
                let mut filter_on_fields = Vec::new();
                let mut and = None;
                let mut or = None;
                let mut not = None;

                while let Some((key, value)) = visitor.next_entry::<String, Value>()? {
                    match (key.as_str(), value) {
                        ("and", Value::WhereFilter(f)) => and = Some(f),
                        ("or", Value::WhereFilter(f)) => or = Some(f),
                        ("not", Value::NotWhereFilter(f)) => not = Some(Box::new(f)),
                        (_, Value::Filter(f)) => {
                            filter_on_fields.push((key, f));
                        }
                        (_, value) => {
                            return Err(de::Error::custom(format!(
                                "Invalid where filter for key {key}: {value:?}"
                            )))
                        }
                    }
                }

                Ok(WhereFilter {
                    filter_on_fields,
                    and,
                    or,
                    not,
                })
            }
        }

        deserializer.deserialize_map(HiddenWhereFilterVisitor)
    }
}

impl Serialize for WhereFilter {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeMap;
        // Count the number of entries for the map
        let mut num_entries = self.filter_on_fields.len();
        if self.and.as_ref().is_some_and(|v| !v.is_empty()) {
            num_entries += 1;
        }
        if self.or.as_ref().is_some_and(|v| !v.is_empty()) {
            num_entries += 1;
        }
        if self.not.is_some() {
            num_entries += 1;
        }

        let mut map = serializer.serialize_map(Some(num_entries))?;

        // Serialize filter_on_fields
        for (key, filter) in &self.filter_on_fields {
            map.serialize_entry(key, filter)?;
        }

        // Serialize and
        if let Some(and_vec) = &self.and {
            if !and_vec.is_empty() {
                map.serialize_entry("and", and_vec)?;
            }
        }

        // Serialize or
        if let Some(or_vec) = &self.or {
            if !or_vec.is_empty() {
                map.serialize_entry("or", or_vec)?;
            }
        }

        // Serialize not
        if let Some(not_val) = &self.not {
            map.serialize_entry("not", not_val.as_ref())?;
        }

        map.end()
    }
}

impl WhereFilter {
    pub fn is_empty(&self) -> bool {
        self.filter_on_fields.is_empty()
            && self.and.as_ref().is_none_or(|v| v.is_empty())
            && self.or.as_ref().is_none_or(|v| v.is_empty())
            && self.not.is_none()
    }

    pub fn get_all_keys(&self) -> Vec<String> {
        let mut keys = self
            .filter_on_fields
            .iter()
            .map(|(k, _)| k)
            .cloned()
            .collect::<Vec<_>>();
        if let Some(and) = &self.and {
            for filter in and.iter() {
                keys.extend(filter.get_all_keys());
            }
        }
        if let Some(or) = &self.or {
            for filter in or.iter() {
                keys.extend(filter.get_all_keys());
            }
        }
        if let Some(not) = &self.not {
            keys.extend(not.get_all_keys());
        }
        keys
    }

    pub fn get_all_pairs(&self) -> (Vec<(&String, &Filter)>, ()) {
        fn recursive_get_all_pairs<'filter>(
            filter: &'filter WhereFilter,
            pairs: &mut Vec<(&'filter String, &'filter Filter)>,
        ) {
            for (key, value) in filter.filter_on_fields.iter() {
                pairs.push((key, value));
            }
            if let Some(and) = &filter.and {
                for f in and.iter() {
                    recursive_get_all_pairs(f, pairs);
                }
            }
            if let Some(or) = &filter.or {
                for f in or.iter() {
                    recursive_get_all_pairs(f, pairs);
                }
            }
            if let Some(not) = &filter.not {
                recursive_get_all_pairs(not, pairs);
            }
        }

        let mut pairs = vec![];
        recursive_get_all_pairs(self, &mut pairs);

        (pairs, ())
    }
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
pub struct NLPSearchRequest {
    pub query: String,
    pub llm_config: Option<InteractionLLMConfig>,
    #[serde(default, rename = "userID")]
    pub user_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
pub enum SortOrder {
    #[serde(rename = "ASC")]
    #[default]
    Ascending,
    #[serde(rename = "DESC")]
    Descending,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SortBy {
    pub property: String,
    #[serde(default)]
    pub order: SortOrder,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchParams {
    #[serde(flatten)]
    pub mode: SearchMode,
    #[serde(default)]
    pub limit: Limit,
    #[serde(default)]
    pub offset: SearchOffset,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub boost: HashMap<String, f32>,
    #[serde(
        default,
        deserialize_with = "deserialize_properties",
        serialize_with = "serialize_properties",
        skip_serializing_if = "Properties::is_none"
    )]
    pub properties: Properties,
    #[serde(default, rename = "where")]
    pub where_filter: WhereFilter,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub facets: HashMap<String, FacetDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub indexes: Option<Vec<IndexId>>,
    #[serde(default, rename = "sortBy", skip_serializing_if = "Option::is_none")]
    pub sort_by: Option<SortBy>,
    #[serde(default, rename = "userID", skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}
impl PartialSchema for SearchParams {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        let b = AnyOfBuilder::new();
        axum_openapi3::utoipa::openapi::RefOr::T(b.into())
    }
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

fn serialize_properties<S>(props: &Properties, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::ser::Serializer,
{
    match props {
        Properties::None => serializer.serialize_unit(),
        Properties::Star => serializer.serialize_str("*"),
        Properties::Specified(vec) => vec.serialize(serializer),
    }
}

impl TryFrom<serde_json::Value> for SearchParams {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SearchResultHit {
    pub id: String,
    pub score: f32,
    pub document: Option<Arc<RawJSONDocument>>,
}

impl Serialize for SearchResultHit {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("SearchResultHit", 3)?;
        state.serialize_field("id", &self.id)?;
        // NB: `split_once` doesn't allocate a new string. it returns a tuple of slices
        let (index_id, _) = match self.id.split_once(":") {
            Some((index_id, _)) => (index_id, ()),
            None => ("", ()),
        };
        state.serialize_field("index_id", &index_id)?;
        state.serialize_field("score", &self.score)?;
        state.serialize_field("document", &self.document)?;
        state.end()
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelatedQueriesFormat {
    #[serde(rename = "question")]
    Question,
    #[serde(rename = "query")]
    Query,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedRequest {
    pub enabled: Option<bool>,
    pub size: Option<usize>,
    pub format: Option<RelatedQueriesFormat>,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CollectionStatsRequest {
    #[serde(default)]
    pub with_keys: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Copy, ToSchema)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "assistant")]
    Assistant,
    #[serde(rename = "user")]
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InteractionMessage {
    pub role: Role,
    pub content: String,
}

impl InteractionMessage {
    pub fn to_async_openai_message(&self) -> ChatCompletionRequestMessage {
        match &self.role {
            Role::System => ChatCompletionRequestSystemMessageArgs::default()
                .content(self.content.clone())
                .build()
                .unwrap()
                .into(),
            Role::Assistant => ChatCompletionRequestAssistantMessageArgs::default()
                .content(self.content.clone())
                .build()
                .unwrap()
                .into(),
            Role::User => ChatCompletionRequestUserMessageArgs::default()
                .content(self.content.clone())
                .build()
                .unwrap()
                .into(),
        }
    }

    pub fn into_json(self) -> serde_json::Value {
        serde_json::json!({
            "role": self.role,
            "content": self.content,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InteractionLLMConfig {
    pub provider: RemoteLLMProvider,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub interaction_id: String,
    pub system_prompt_id: Option<String>,
    pub query: String,
    pub visitor_id: String,
    pub conversation_id: String,
    pub related: Option<RelatedRequest>,
    pub messages: Vec<InteractionMessage>,
    pub llm_config: Option<InteractionLLMConfig>,
    pub min_similarity: Option<f32>,
    pub max_documents: Option<usize>,
    pub ragat_notation: Option<String>,
    pub search_mode: Option<String>,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
pub enum ExecuteActionPayloadName {
    #[serde(rename = "search")]
    Search,
}

#[derive(Deserialize, Clone, Serialize, ToSchema)]
pub struct ExecuteActionPayload {
    pub name: ExecuteActionPayloadName,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InsertSystemPromptParams {
    pub id: Option<String>,
    pub name: String,
    pub prompt: String,
    pub usage_mode: SystemPromptUsageMode,
    pub llm_config: Option<InteractionLLMConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteSystemPromptParams {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub enum SystemPromptUsageMode {
    #[serde(rename = "automatic")]
    Automatic,
    #[serde(rename = "manual")]
    Manual,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateSystemPromptParams {
    pub id: String,
    pub name: String,
    pub prompt: String,
    pub usage_mode: SystemPromptUsageMode,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertDocumentsResult {
    pub inserted: usize,
    pub replaced: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateDocumentsResult {
    pub inserted: usize,
    pub updated: usize,
    pub failed: usize,
}

#[derive(Debug, ToSchema, PartialEq)]
pub enum IndexEmbeddingsCalculation {
    None,
    Automatic,
    AllProperties,
    Properties(Vec<String>),
}

impl<'de> Deserialize<'de> for IndexEmbeddingsCalculation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        #[derive(Deserialize, Serialize, Debug)]
        #[serde(untagged)]
        enum HiddenIndexEmbeddingsCalculation {
            S(String),
            V(Vec<String>),
        }

        let v = HiddenIndexEmbeddingsCalculation::deserialize(deserializer)?;
        match v {
            HiddenIndexEmbeddingsCalculation::S(s) => {
                match s.as_str() {
                    "none" => Ok(IndexEmbeddingsCalculation::None),
                    "automatic" => Ok(IndexEmbeddingsCalculation::Automatic),
                    "all_properties" => Ok(IndexEmbeddingsCalculation::AllProperties),
                    _ => Err(de::Error::custom(
                        "Invalid value for index embeddings calculation. Expected 'none', 'automatic', 'all_properties', or 'hook' or an array of strings",
                    )),
                }
            },
            HiddenIndexEmbeddingsCalculation::V(v) => Ok(IndexEmbeddingsCalculation::Properties(v)),
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateIndexRequest {
    #[serde(rename = "id")]
    pub index_id: IndexId,
    pub embedding: Option<IndexEmbeddingsCalculation>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ReplaceIndexRequest {
    #[serde(rename = "target_index_id")]
    pub runtime_index_id: IndexId,
    pub temp_index_id: IndexId,
    #[serde(default)]
    pub reference: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InsertToolsParams {
    pub id: String,
    pub system_prompt: Option<String>,
    pub description: String,
    pub parameters: String,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DeleteToolParams {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateToolParams {
    pub id: String,
    pub system_prompt: Option<String>,
    pub name: String,
    pub description: String,
    pub parameters: String,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RunToolsParams {
    pub tool_ids: Option<Vec<String>>,
    pub messages: Vec<InteractionMessage>,
    pub llm_config: Option<InteractionLLMConfig>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Number {
    I32(i32),
    F32(f32),
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Number::I32(value) => write!(f, "{value}"),
            Number::F32(value) => write!(f, "{value}"),
        }
    }
}

impl From<i32> for Number {
    fn from(value: i32) -> Self {
        Number::I32(value)
    }
}
impl From<f32> for Number {
    fn from(value: f32) -> Self {
        Number::F32(value)
    }
}
impl TryFrom<&Value> for Number {
    type Error = serde_json::Error;

    fn try_from(value: &Value) -> Result<Self, Self::Error> {
        match value {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Number::I32(i as i32))
                } else if let Some(f) = n.as_f64() {
                    Ok(Number::F32(f as f32))
                } else {
                    Err(serde_json::Error::custom("Not a number"))
                }
            }
            _ => Err(serde_json::Error::custom("Not a number")),
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::I32(a), Number::I32(b)) => a == b,
            (Number::I32(a), Number::F32(b)) => *a as f32 == *b,
            (Number::F32(a), Number::F32(b)) => {
                // This is against the IEEE 754-2008 standard,
                // But we don't care here.
                if a.is_nan() && b.is_nan() {
                    return true;
                }
                a == b
            }
            (Number::F32(a), Number::I32(b)) => *a == *b as f32,
        }
    }
}
impl Eq for Number {}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // f32 is implemented as "binary32" type defined in IEEE 754-2008
        // So, it means, it can represent also +/- Infinity and NaN
        // Threat NaN as "more" the Infinity
        // See `total_cmp` method in f32
        match (self, other) {
            (Number::I32(a), Number::I32(b)) => a.cmp(b),
            (Number::I32(a), Number::F32(b)) => (*a as f32).total_cmp(b),
            (Number::F32(a), Number::F32(b)) => a.total_cmp(b),
            (Number::F32(a), Number::I32(b)) => a.total_cmp(&(*b as f32)),
        }
    }
}

impl std::ops::Add for Number {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Number::I32(a), Number::I32(b)) => (a + b).into(),
            (Number::I32(a), Number::F32(b)) => (a as f32 + b).into(),
            (Number::F32(a), Number::F32(b)) => (a + b).into(),
            (Number::F32(a), Number::I32(b)) => (a + b as f32).into(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct SerializableNumber(pub Number);

impl Serialize for SerializableNumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeTuple;

        match &self.0 {
            Number::F32(v) => {
                let mut tuple = serializer.serialize_tuple(2)?;

                if v.is_infinite() && v.is_sign_positive() {
                    tuple.serialize_element(&3_u8)?;
                    return tuple.end();
                }
                if v.is_infinite() && v.is_sign_negative() {
                    tuple.serialize_element(&4_u8)?;
                    return tuple.end();
                }

                tuple.serialize_element(&1_u8)?;
                tuple.serialize_element(v)?;
                tuple.end()
            }
            Number::I32(v) => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(&2_u8)?;
                tuple.serialize_element(v)?;
                tuple.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SerializableNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};

        struct SerializableNumberVisitor;

        impl<'de> Visitor<'de> for SerializableNumberVisitor {
            type Value = SerializableNumber;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a tuple of size 2 consisting of a u64 discriminant and a value"
                )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let discriminant: u8 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                match discriminant {
                    1_u8 => {
                        let x = seq
                            .next_element()?
                            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                        Ok(SerializableNumber(Number::F32(x)))
                    }
                    2 => {
                        let y = seq
                            .next_element()?
                            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                        Ok(SerializableNumber(Number::I32(y)))
                    }
                    3 => Ok(SerializableNumber(Number::F32(f32::INFINITY))),
                    4 => Ok(SerializableNumber(Number::F32(f32::NEG_INFINITY))),
                    d => Err(A::Error::invalid_value(
                        serde::de::Unexpected::Unsigned(d.into()),
                        &"1, 2, 3, 4",
                    )),
                }
            }
        }

        deserializer.deserialize_tuple(2, SerializableNumberVisitor)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum NumberFilter {
    #[serde(rename = "eq")]
    Equal(Number),
    #[serde(rename = "gt")]
    GreaterThan(Number),
    #[serde(rename = "gte")]
    GreaterThanOrEqual(Number),
    #[serde(rename = "lt")]
    LessThan(Number),
    #[serde(rename = "lte")]
    LessThanOrEqual(Number),
    #[serde(rename = "between")]
    Between((Number, Number)),
}

#[derive(Debug, Serialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct OramaDate(DateTime<Utc>);

impl OramaDate {
    pub fn try_from_i64(millisec_timestamp: i64) -> Option<Self> {
        chrono::DateTime::from_timestamp_millis(millisec_timestamp).map(OramaDate)
    }
    pub fn as_i64(&self) -> i64 {
        self.0.timestamp_millis()
    }
}

impl<'de> Deserialize<'de> for OramaDate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        OramaDate::try_from(s).map_err(serde::de::Error::custom)
    }
}

impl TryFrom<String> for OramaDate {
    type Error = anyhow::Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        (&value).try_into()
    }
}

impl TryFrom<&String> for OramaDate {
    type Error = anyhow::Error;

    fn try_from(value: &String) -> Result<Self, Self::Error> {
        value.as_str().try_into()
    }
}

impl TryFrom<&str> for OramaDate {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        use dateparser::parse_with_timezone;

        parse_with_timezone(value, &Utc).map(OramaDate)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum DateFilter {
    #[serde(rename = "eq")]
    Equal(OramaDate),
    #[serde(rename = "gt")]
    GreaterThan(OramaDate),
    #[serde(rename = "gte")]
    GreaterThanOrEqual(OramaDate),
    #[serde(rename = "lt")]
    LessThan(OramaDate),
    #[serde(rename = "lte")]
    LessThanOrEqual(OramaDate),
    #[serde(rename = "between")]
    Between((OramaDate, OramaDate)),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeoSearchRadiusValue(f32);

impl PartialEq for GeoSearchRadiusValue {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for GeoSearchRadiusValue {}

impl GeoSearchRadiusValue {
    pub fn to_meter(&self, in_unit: GeoSearchRadiusUnit) -> f32 {
        match in_unit {
            GeoSearchRadiusUnit::CentiMeter => self.0 * 0.01, // 1 cm = 0.01 m
            GeoSearchRadiusUnit::Meter => self.0,             // already in meters
            GeoSearchRadiusUnit::KiloMeter => self.0 * 1000.0, // 1 km = 1000 m
            GeoSearchRadiusUnit::Feet => self.0 * 0.3048,     // 1 ft = 0.3048 m
            GeoSearchRadiusUnit::Yard => self.0 * 0.9144,     // 1 yd = 0.9144 m
            GeoSearchRadiusUnit::Mile => self.0 * 1609.344,   // 1 mi = 1609.344 m
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum GeoSearchRadiusUnit {
    #[serde(rename = "cm")]
    CentiMeter,
    #[serde(rename = "m")]
    Meter,
    #[serde(rename = "km")]
    KiloMeter,
    #[serde(rename = "ft")]
    Feet,
    #[serde(rename = "yd")]
    Yard,
    #[serde(rename = "mi")]
    Mile,
}
impl Default for GeoSearchRadiusUnit {
    fn default() -> Self {
        Self::Meter
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct GeoSearchRadiusFilter {
    pub coordinates: GeoPoint,
    #[serde(default)]
    pub unit: GeoSearchRadiusUnit,
    pub value: GeoSearchRadiusValue,
    #[serde(default = "get_true")]
    pub inside: bool,
}

fn get_true() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub struct GeoSearchPolygonFilter {
    pub coordinates: Vec<GeoPoint>,
    #[serde(default = "get_true")]
    pub inside: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq, Eq))]
pub enum GeoSearchFilter {
    #[serde(rename = "radius")]
    Radius(GeoSearchRadiusFilter),
    #[serde(rename = "polygon")]
    Polygon(GeoSearchPolygonFilter),
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, ToSchema)]
pub struct IndexId(StackString<64>);

impl IndexId {
    pub fn try_new<A: AsRef<str>>(key: A) -> Result<Self> {
        StackString::<64>::try_new(key)
            .map(IndexId)
            .map_err(|e| anyhow::anyhow!("IndexId is too long. Max 64 char. {:?}", e))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}
impl Display for IndexId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Serialize)]
pub struct IndexFieldType {
    pub field_id: FieldId,
    pub field_path: String,
    pub is_array: bool,
    pub field_type: FieldType,
}

#[derive(Debug, Serialize)]
pub struct DescribeCollectionIndexResponse {
    pub id: IndexId,
    pub document_count: usize,
    pub fields: Vec<IndexFieldType>,
    pub automatically_chosen_properties: Option<HashMap<String, ChosenProperties>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct StackString<const N: usize>(ArrayString<N>);
impl<const N: usize> PartialSchema for StackString<N> {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        // TODO: put a max length
        String::schema()
    }
}
impl<const N: usize> ToSchema for StackString<N> {}
impl<const N: usize> Serialize for StackString<N> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        String::from_utf8_lossy(self.0.as_bytes()).serialize(serializer)
    }
}
impl<'de, const N: usize> Deserialize<'de> for StackString<N> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(StackString(ArrayString::<N>::try_from(s.as_str()).unwrap()))
    }
}
impl<const N: usize> StackString<N> {
    pub fn try_new<A: AsRef<str>>(key: A) -> Result<Self> {
        let key = key.as_ref();
        if key.is_empty() {
            bail!("StackString cannot be empty");
        }

        let mut s = ArrayString::<N>::new();
        let r = s.try_push_str(key);
        if let Err(e) = r {
            bail!("Parameter is too long. Max {} char. {:?}", N, e);
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}
impl<const N: usize> Display for StackString<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::SecondsFormat;

    use serde_json::json;
    use std::cmp::Ordering;

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
            "mode": "fulltext",
            "term": "hello",
            "threshold": 0.5,
        });
        let j = serde_json::to_string(&j).unwrap();
        let p = serde_json::from_str::<SearchParams>(&j).unwrap();
        assert!(matches!(p.mode, SearchMode::FullText(_)));
        assert!(
            matches!(p.mode, SearchMode::FullText(v) if (v.threshold.unwrap().0 - 0.5).abs() < 0.01)
        );

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
        assert!(format!("{p}").contains("Invalid search mode: unknown_value"));

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
            "embeddings_model": "BGESmall",
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

    #[test]
    fn test_search_hit() {
        let hit = SearchResultHit {
            id: "foo".to_string(),
            score: 0.5,
            document: None,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let deserialized_hit: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized_hit["id"], "foo");
        assert_eq!(deserialized_hit["score"], 0.5);
        assert_eq!(deserialized_hit["document"], Value::Null);
        assert_eq!(deserialized_hit["index_id"], "");

        let hit = SearchResultHit {
            id: "my-index-id:55".to_string(),
            score: 0.5,
            document: None,
        };
        let json = serde_json::to_string(&hit).unwrap();
        let deserialized_hit: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized_hit["id"], "my-index-id:55");
        assert_eq!(deserialized_hit["score"], 0.5);
        assert_eq!(deserialized_hit["document"], Value::Null);
        assert_eq!(deserialized_hit["index_id"], "my-index-id");
    }

    #[test]
    fn test_threshold() {
        #[derive(Deserialize)]
        struct Foo {
            threshold: Threshold,
        }
        let j = json!({
            "threshold": 1.0,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.threshold.0, 1.0);

        let j = json!({
            "threshold": 1,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.threshold.0, 1.0);

        let j = json!({
            "threshold": 0.5,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.threshold.0, 0.5);

        let j = json!({
            "threshold": 0,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.threshold.0, 0.0);

        let j = json!({
            "threshold": 0.0,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.threshold.0, 0.0);
    }

    #[test]
    fn test_similarity() {
        #[derive(Deserialize)]
        struct Foo {
            similarity: Similarity,
        }
        let j = json!({
            "similarity": 1.0,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.similarity.0, 1.0);

        let j = json!({
            "similarity": 1,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.similarity.0, 1.0);

        let j = json!({
            "similarity": 0.5,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.similarity.0, 0.5);

        let j = json!({
            "similarity": 0,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.similarity.0, 0.0);

        let j = json!({
            "similarity": 0.0,
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.similarity.0, 0.0);
    }

    #[test]
    fn test_index_embedding_calculation() {
        #[derive(Deserialize, Debug)]
        struct Foo {
            embedding: IndexEmbeddingsCalculation,
        }
        let j = json!({
            "embedding": "automatic",
        });
        let p = serde_json::from_value::<Foo>(j).unwrap();
        assert_eq!(p.embedding, IndexEmbeddingsCalculation::Automatic);
    }

    #[test]
    fn test_create_index() {
        let j = json!({
            "id": "foo",
            "embedding": "automatic",
        });
        let p = serde_json::from_value::<CreateIndexRequest>(j).unwrap();
        assert_eq!(p.index_id, IndexId::try_new("foo").unwrap());
        assert_eq!(p.embedding, Some(IndexEmbeddingsCalculation::Automatic));

        let j = json!({
            "id": "foo",
            "embedding": "all_properties",
        });
        let p = serde_json::from_value::<CreateIndexRequest>(j).unwrap();
        assert_eq!(p.index_id, IndexId::try_new("foo").unwrap());
        assert_eq!(p.embedding, Some(IndexEmbeddingsCalculation::AllProperties));

        let j = json!({
            "id": "foo",
            "embedding": ["a", "b"],
        });
        let p = serde_json::from_value::<CreateIndexRequest>(j).unwrap();
        assert_eq!(p.index_id, IndexId::try_new("foo").unwrap());
        assert_eq!(
            p.embedding,
            Some(IndexEmbeddingsCalculation::Properties(vec![
                "a".to_string(),
                "b".to_string()
            ]))
        );
    }

    #[test]
    fn test_filter_deserialize() {
        let j = json!({
            "foo": { "eq": 1 },
            "bar": true,
            "baz": "hello",
            "and": [
                {
                    "foo": { "eq": 2 },
                },
                {
                    "bar": false,
                }
            ],
            "or": [
                {
                    "foo": { "eq": 3 },
                },
            ],
            "not": {
                "foo": { "eq": 4 },
            }
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 3);
        assert_eq!(
            p.filter_on_fields[0],
            (
                "foo".to_string(),
                Filter::Number(NumberFilter::Equal(Number::I32(1)))
            )
        );
        assert_eq!(
            p.filter_on_fields[1],
            ("bar".to_string(), Filter::Bool(true))
        );
        assert_eq!(
            p.filter_on_fields[2],
            ("baz".to_string(), Filter::String("hello".to_string()))
        );
        assert_eq!(p.and.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.and.as_ref().unwrap()[0].filter_on_fields[0],
            (
                "foo".to_string(),
                Filter::Number(NumberFilter::Equal(Number::I32(2)))
            )
        );
        assert_eq!(
            p.and.as_ref().unwrap()[1].filter_on_fields[0],
            ("bar".to_string(), Filter::Bool(false))
        );
        assert_eq!(p.or.as_ref().unwrap().len(), 1);
        assert_eq!(
            p.or.as_ref().unwrap()[0].filter_on_fields[0],
            (
                "foo".to_string(),
                Filter::Number(NumberFilter::Equal(Number::I32(3)))
            )
        );
        assert_eq!(
            p.not.as_ref().unwrap().filter_on_fields[0],
            (
                "foo".to_string(),
                Filter::Number(NumberFilter::Equal(Number::I32(4)))
            )
        );

        let j = json!({
            "and": [
                { "color": "black" },
                { "price.amount": { "lte": 1000 } }
            ]
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 0);
        assert_eq!(p.and.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.and.as_ref().unwrap()[0].filter_on_fields[0],
            ("color".to_string(), Filter::String("black".to_string()))
        );
        assert_eq!(
            p.and.as_ref().unwrap()[1].filter_on_fields[0],
            (
                "price.amount".to_string(),
                Filter::Number(NumberFilter::LessThanOrEqual(Number::I32(1000)))
            )
        );
        assert!(p.or.is_none());
        assert!(p.not.is_none());

        let j = json!({
            "and": [
                { "color": "black" },
                { "price.amount": { "lte": 1000 } }
            ],
            "not": { "price.discount.amount": { "gt": 0 } }
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 0);
        assert_eq!(p.and.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.and.as_ref().unwrap()[0].filter_on_fields[0],
            ("color".to_string(), Filter::String("black".to_string()))
        );
        assert_eq!(
            p.and.as_ref().unwrap()[1].filter_on_fields[0],
            (
                "price.amount".to_string(),
                Filter::Number(NumberFilter::LessThanOrEqual(Number::I32(1000)))
            )
        );
        assert_eq!(
            p.not.as_ref().unwrap().filter_on_fields[0],
            (
                "price.discount.amount".to_string(),
                Filter::Number(NumberFilter::GreaterThan(Number::I32(0)))
            )
        );
        assert!(p.or.is_none());

        let j = json!({
            "or": [
                { "productName": "my-phone-1" },
                { "productName": "my-phone-2" }
            ]
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 0);
        assert_eq!(p.or.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.or.as_ref().unwrap()[0].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-1".to_string())
            )
        );
        assert_eq!(
            p.or.as_ref().unwrap()[1].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-2".to_string())
            )
        );
        assert!(p.and.is_none());
        assert!(p.not.is_none());

        let j = json!({
            "or": [
            { "productName": "my-phone-1" },
            { "productName": "my-phone-2" }
            ],
            "not": { "price.discount.amount": { "gt": 0 } }
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();
        assert_eq!(p.filter_on_fields.len(), 0);
        assert_eq!(p.or.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.or.as_ref().unwrap()[0].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-1".to_string())
            )
        );
        assert_eq!(
            p.or.as_ref().unwrap()[1].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-2".to_string())
            )
        );
        assert_eq!(
            p.not.as_ref().unwrap().filter_on_fields[0],
            (
                "price.discount.amount".to_string(),
                Filter::Number(NumberFilter::GreaterThan(Number::I32(0)))
            )
        );
        assert!(p.and.is_none());

        let j = json!({
            "or": [
                { "productName": "my-phone-1" },
                { "productName": "my-phone-2" }
            ],
            "and": [
                { "color": "black" }
            ]
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 0);
        assert_eq!(p.or.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.or.as_ref().unwrap()[0].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-1".to_string())
            )
        );
        assert_eq!(
            p.or.as_ref().unwrap()[1].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-2".to_string())
            )
        );
        assert_eq!(p.and.as_ref().unwrap().len(), 1);
        assert_eq!(
            p.and.as_ref().unwrap()[0].filter_on_fields[0],
            ("color".to_string(), Filter::String("black".to_string()))
        );
        assert!(p.not.is_none());

        let j = json!({
            "or": [
                { "productName": "my-phone-1" },
                { "productName": "my-phone-2" }
            ],
            "and": [
                { "color": "black" }
            ],
            "not": { "manufacturer.name": "my-phone-3" }
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 0);
        assert_eq!(p.or.as_ref().unwrap().len(), 2);
        assert_eq!(
            p.or.as_ref().unwrap()[0].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-1".to_string())
            )
        );
        assert_eq!(
            p.or.as_ref().unwrap()[1].filter_on_fields[0],
            (
                "productName".to_string(),
                Filter::String("my-phone-2".to_string())
            )
        );
        assert_eq!(p.and.as_ref().unwrap().len(), 1);
        assert_eq!(
            p.and.as_ref().unwrap()[0].filter_on_fields[0],
            ("color".to_string(), Filter::String("black".to_string()))
        );
        assert_eq!(
            p.not.as_ref().unwrap().filter_on_fields[0],
            (
                "manufacturer.name".to_string(),
                Filter::String("my-phone-3".to_string())
            )
        );

        let j = json!({
            "date": { "eq": "2023-01-01T00:00:00Z" },
        });
        let p = serde_json::from_value::<WhereFilter>(j);
        assert!(p.is_ok());

        let j = json!({
            "date": { "eq": "2023-01-01" },
        });
        let p = serde_json::from_value::<WhereFilter>(j);
        assert!(p.is_ok());

        let j = json!({
            "position": {
                "polygon": {
                    "coordinates": [
                        { "lat": 45.46472, "lon": 9.1886  },
                        { "lat": 45.46352, "lon": 9.19177 },
                        { "lat": 45.46278, "lon": 9.19176 },
                        { "lat": 45.4628,  "lon": 9.18857 },
                        { "lat": 45.46472, "lon": 9.1886  },
                    ],
                    "inside": true,
                }
            }
        });
        let p = serde_json::from_value::<WhereFilter>(j);
        assert!(p.is_ok());

        let j = json!({
            "position": {
                "radius": {
                  "coordinates": {
                    "lat": 45.4648,
                    "lon": 9.18998
                  },
                  "unit": "m",
                  "value": 1000,
                  "inside": true
                }
              }
        });
        let p = serde_json::from_value::<WhereFilter>(j);
        assert!(p.is_ok());

        let j = json!({
            "location": {
                "radius": {
                    "coordinates": {
                        "lat": 45.4648,
                        "lon": 9.18998
                    },
                    // "unit": 'm',        // The unit of measurement. The default is "m" (meters)
                    "value": 1000,      // The radius length. In that case, 1km
                    // "inside": true      // Whether we want to return the documents inside or outside the radius. The default is "true"
                }
            }
        });
        let p = serde_json::from_value::<WhereFilter>(j);
        assert!(p.is_ok());
    }

    #[test]
    fn test_filter_deserialize_with_bad_names() {
        // if "and", "or", "not" are used as field names, they should be treated as normal fields
        // and not as logical operators
        let j = json!({
            "and": "hello",
            "or": { "eq": 3 },
            "not": true,
        });
        let p = serde_json::from_value::<WhereFilter>(j).unwrap();

        assert_eq!(p.filter_on_fields.len(), 3);
        assert_eq!(
            p.filter_on_fields[0],
            ("and".to_string(), Filter::String("hello".to_string()))
        );
        assert_eq!(
            p.filter_on_fields[1],
            (
                "or".to_string(),
                Filter::Number(NumberFilter::Equal(Number::I32(3)))
            )
        );
        assert_eq!(
            p.filter_on_fields[2],
            ("not".to_string(), Filter::Bool(true))
        );
        assert!(p.and.is_none());
        assert!(p.or.is_none());
        assert!(p.not.is_none());
    }

    #[test]
    fn test_number_eq() {
        let a = Number::from(1);
        let b = Number::from(1.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(-1);
        let b = Number::from(-1.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(2);
        let b = Number::from(2.0);
        assert_eq!(a, b);
        assert!(a == b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(f32::INFINITY);
        let b = Number::from(f32::NEG_INFINITY);
        assert_ne!(a, b);
        assert!(a != b);
        assert!(a != Number::from(f32::NAN));

        let a = Number::from(f32::NAN);
        assert_eq!(a, Number::from(f32::NAN));
        assert!(a == Number::from(f32::NAN));
        assert!(a != Number::from(f32::INFINITY));
        assert!(a != Number::from(f32::NEG_INFINITY));
        assert!(a != Number::from(0));
    }

    #[test]
    fn test_number_ord() {
        let a = Number::from(1);

        let b = Number::from(1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(-1);

        let b = Number::from(-1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(1.0);

        let b = Number::from(1.0);
        assert_eq!(a.cmp(&b), Ordering::Equal);

        let b = Number::from(2.0);
        assert_eq!(a.cmp(&b), Ordering::Less);
        let b = Number::from(2);
        assert_eq!(a.cmp(&b), Ordering::Less);

        let b = Number::from(-2.0);
        assert_eq!(a.cmp(&b), Ordering::Greater);
        let b = Number::from(-2);
        assert_eq!(a.cmp(&b), Ordering::Greater);

        let a = Number::from(1);
        assert!(a < Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a < Number::from(f32::NAN));

        let a = Number::from(1.0);
        assert!(a < Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a < Number::from(f32::NAN));

        let a = Number::from(f32::NAN);
        assert!(a > Number::from(f32::INFINITY));
        assert!(a > Number::from(f32::NEG_INFINITY));
        assert!(a == Number::from(f32::NAN));

        let v = [
            Number::from(1),
            Number::from(1.0),
            Number::from(2),
            Number::from(2.0),
            Number::from(-1),
            Number::from(-1.0),
            Number::from(-2),
            Number::from(-2.0),
            Number::from(f32::INFINITY),
            Number::from(f32::NEG_INFINITY),
            Number::from(f32::NAN),
        ];

        for i in 0..v.len() {
            for j in 0..v.len() {
                let way = v[i].cmp(&v[j]);
                let other_way = v[j].cmp(&v[i]);

                assert_eq!(way.reverse(), other_way);
            }
        }
    }

    #[test]
    fn test_serialize_search_params() {
        let data: OramaDate = "2025-07-11T00:00:00Z".try_into().unwrap();

        let search_params = SearchParams {
            boost: HashMap::from([("field_id".to_string(), 1.0)]),
            facets: HashMap::from([
                (
                    "field_id1".to_string(),
                    FacetDefinition::Bool(BoolFacetDefinition {
                        r#true: true,
                        r#false: false,
                    }),
                ),
                (
                    "field_id2".to_string(),
                    FacetDefinition::Number(NumberFacetDefinition {
                        ranges: vec![NumberFacetDefinitionRange {
                            from: Number::I32(2),
                            to: Number::F32(42.0),
                        }],
                    }),
                ),
                (
                    "field_id3".to_string(),
                    FacetDefinition::String(StringFacetDefinition),
                ),
            ]),
            user_id: Some("user-id".to_string()),
            indexes: Some(vec![IndexId::try_new("my-index-id").unwrap()]),
            limit: Limit(42),
            offset: SearchOffset(42),
            properties: Properties::None,
            where_filter: WhereFilter {
                filter_on_fields: vec![
                    ("field_id1".to_string(), Filter::Bool(true)),
                    (
                        "field_id2".to_string(),
                        Filter::Number(NumberFilter::Between((Number::I32(4), Number::I32(4)))),
                    ),
                    ("field_id3".to_string(), Filter::String("wow".to_string())),
                    (
                        "field_id4".to_string(),
                        Filter::Date(DateFilter::Equal(data.clone())),
                    ),
                ],
                and: None,
                or: None,
                not: None,
            },
            mode: SearchMode::Vector(VectorMode {
                term: "the-term".to_string(),
                similarity: Similarity(1.0),
            }),
            sort_by: None,
        };
        let search_params_value = serde_json::to_value(&search_params).unwrap();

        let expected = json!({
          "mode": "vector",
          "term": "the-term",
          "similarity": 1.0,
          "limit": 42,
          "offset": 42,
          "boost": {
            "field_id": 1.0
          },
          "where": {
            "field_id1": true,
            "field_id2": {
              "between": [
                4,
                4
              ]
            },
            "field_id3": "wow",
            "field_id4": {
              "eq": data.0.to_rfc3339_opts(SecondsFormat::Secs, true),
            }
          },
          "facets": {
            "field_id2": {
              "ranges": [
                {
                  "from": 2,
                  "to": 42.0
                }
              ]
            },
            "field_id3": {},
            "field_id1": {
              "true": true,
              "false": false
            }
          },
          "indexes": [
            "my-index-id"
          ],
          "userID": "user-id",
        });

        assert_eq!(search_params_value, expected);
    }

    #[test]
    fn test_serialize_sort_by() {
        let params = json!({
            "term": "hello",
        });
        let search_params: SearchParams = serde_json::from_value(params).unwrap();
        assert_eq!(search_params.sort_by, None);

        let params = json!({
            "term": "hello",
            "sortBy": {
                "property": "name",
            },
        });
        let search_params: SearchParams = serde_json::from_value(params).unwrap();
        assert_eq!(
            search_params.sort_by,
            Some(SortBy {
                property: "name".to_string(),
                order: SortOrder::Ascending,
            })
        );

        let params = json!({
            "term": "hello",
            "sortBy": {
                "property": "name",
                "order": "ASC",
            },
        });
        let search_params: SearchParams = serde_json::from_value(params).unwrap();
        assert_eq!(
            search_params.sort_by,
            Some(SortBy {
                property: "name".to_string(),
                order: SortOrder::Ascending,
            })
        );

        let params = json!({
            "term": "hello",
            "sortBy": {
                "property": "name",
                "order": "DESC",
            },
        });
        let search_params: SearchParams = serde_json::from_value(params).unwrap();
        assert_eq!(
            search_params.sort_by,
            Some(SortBy {
                property: "name".to_string(),
                order: SortOrder::Descending,
            })
        );
    }
}
