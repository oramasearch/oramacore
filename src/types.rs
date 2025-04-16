use crate::ai::automatic_embeddings_selector::ChosenProperties;
use crate::ai::RemoteLLMProvider;
use crate::collection_manager::sides::hooks::HookName;
use crate::collection_manager::sides::{
    deserialize_api_key, serialize_api_key, OramaModelSerializable,
};
use crate::nlp::locales::Locale;
use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use axum_openapi3::utoipa::{self, IntoParams};
use axum_openapi3::utoipa::{PartialSchema, ToSchema};
use redact::Secret;
use serde::de::{Error, Unexpected, Visitor};
use serde::{de, Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
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

#[derive(Debug, Clone, Hash, PartialEq, Eq, Copy)]
pub struct CollectionId(ArrayString<128>);

impl CollectionId {
    pub fn try_new(key: String) -> Result<Self> {
        if key.is_empty() {
            bail!("CollectionId cannot be empty");
        }

        let mut s = ArrayString::<128>::new();
        let r = s.try_push_str(&key);
        if let Err(e) = r {
            bail!("CollectionId is too long. Max 128 char. {:?}", e);
        }
        Ok(Self(s))
    }

    pub fn try_from(key: &str) -> Result<Self> {
        if key.is_empty() {
            bail!("CollectionId cannot be empty");
        }

        let mut s = ArrayString::<128>::new();
        let r = s.try_push_str(key);
        if let Err(e) = r {
            bail!("CollectionId is too long. Max 128 char. {:?}", e);
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

// Implement serialize for CollectionId
impl Serialize for CollectionId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        String::from_utf8_lossy(self.0.as_bytes()).serialize(serializer)
    }
}
// Implement deserialize for CollectionId
impl<'de> Deserialize<'de> for CollectionId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(CollectionId(ArrayString::try_from(s.as_str()).unwrap()))
    }
}
impl CollectionId {
    pub fn from(s: String) -> Self {
        CollectionId(ArrayString::try_from(s.as_str()).unwrap())
    }
}
impl Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(self.0.as_bytes()))
    }
}
impl AsRef<Path> for CollectionId {
    fn as_ref(&self) -> &Path {
        self.0.as_ref()
    }
}

impl PartialSchema for CollectionId {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        String::schema()
    }
}
impl ToSchema for CollectionId {}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct DocumentId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Document {
    #[serde(flatten)]
    #[schema(inline)]
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
                    .map(|(sub_key, sub_value)| (format!("{}.{}", key, sub_key), sub_value.clone()))
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

    pub fn into_raw(&self) -> Result<RawJSONDocument> {
        let id = self
            .inner
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let inner =
            serde_json::value::to_raw_value(&self.inner).context("Cannot serialize document")?;
        Ok(RawJSONDocument { inner, id })
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

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize, ToSchema)]
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

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize, ToSchema)]
pub enum ComplexType {
    Array(#[schema(inline)] ScalarType),
    Embedding,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize, ToSchema)]
pub enum ValueType {
    Scalar(#[schema(inline)] ScalarType),
    Complex(#[schema(inline)] ComplexType),
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

#[derive(Debug, Deserialize, ToSchema)]
pub struct DocumentList(#[schema(inline)] Vec<Document>);
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

pub trait StringParser: Send + Sync {
    fn tokenize_str_and_stem(&self, input: &str) -> Result<Vec<(String, Vec<String>)>>;
}

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

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct ApiKey(Secret<ArrayString<64>>);
impl ApiKey {
    pub fn try_new(key: String) -> Result<Self> {
        if key.is_empty() {
            bail!("API key cannot be empty");
        }

        let mut s = ArrayString::<64>::new();
        let r = s.try_push_str(&key);
        if let Err(e) = r {
            bail!("API key is too long. Max 64 char. {:?}", e);
        }
        let s = Secret::new(s);
        Ok(Self(s))
    }

    pub fn try_from(key: &str) -> Result<Self> {
        if key.is_empty() {
            bail!("API key cannot be empty");
        }

        let mut s = ArrayString::<64>::new();
        let r = s.try_push_str(key);
        if let Err(e) = r {
            bail!("API key is too long. Max 64 char. {:?}", e);
        }
        let s = Secret::new(s);
        Ok(Self(s))
    }

    pub fn expose(&self) -> &str {
        self.0.expose_secret()
    }
}

impl<'de> Deserialize<'de> for ApiKey {
    fn deserialize<D>(deserializer: D) -> Result<ApiKey, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::try_new(s).map_err(|e| de::Error::custom(format!("error: {}", e)))
    }
}

impl PartialSchema for ApiKey {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        // TODO put the min and max size here
        String::schema()
    }
}
impl ToSchema for ApiKey {}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct DeleteCollection {
    pub id: CollectionId,
}

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

    pub reference: Option<String>,
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

#[derive(Debug, Serialize, Deserialize, ToSchema, PartialEq, Eq)]
pub struct CollectionDTO {
    #[schema(inline)]
    pub id: CollectionId,
    pub description: Option<String>,
    pub document_count: u64,
    #[schema(inline)]
    pub fields: HashMap<String, ValueType>,
    pub automatically_chosen_properties: Option<HashMap<String, ChosenProperties>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Copy, Clone)]
pub struct Limit(#[schema(inline)] pub usize);
impl Default for Limit {
    fn default() -> Self {
        Self(10)
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Copy, Clone, Default)]
pub struct Offset(#[schema(inline)] pub usize);

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
#[serde(untagged)]
pub enum Filter {
    Number(#[schema(inline)] NumberFilter),
    Bool(bool),
    String(String),
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

#[derive(Debug, Clone, Serialize, ToSchema)]
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

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]

pub enum FacetDefinition {
    #[serde(untagged)]
    Number(#[schema(inline)] NumberFacetDefinition),
    #[serde(untagged)]
    Bool(#[schema(inline)] BoolFacetDefinition),
    #[serde(untagged)]
    String(#[schema(inline)] StringFacetDefinition),
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
    #[schema(inline)]
    pub offset: Offset,
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
    pub document: Option<Arc<RawJSONDocument>>,
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
    #[serde(rename = "question")]
    Question,
    #[serde(rename = "query")]
    Query,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RelatedRequest {
    pub enabled: Option<bool>,
    pub size: Option<usize>,
    pub format: Option<RelatedQueriesFormat>,
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
pub struct InteractionLLMConfig {
    pub provider: RemoteLLMProvider,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Interaction {
    pub interaction_id: String,
    pub system_prompt_id: Option<String>,
    pub query: String,
    pub visitor_id: String,
    pub conversation_id: String,
    pub related: Option<RelatedRequest>,
    pub messages: Vec<InteractionMessage>,
    pub llm_config: Option<InteractionLLMConfig>,
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

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct InsertDocumentsResult {
    pub inserted: usize,
    pub replaced: usize,
    pub failed: usize,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Number {
    I32(#[schema(inline)] i32),
    F32(#[schema(inline)] f32),
}

impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Number::I32(value) => write!(f, "{}", value),
            Number::F32(value) => write!(f, "{}", value),
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

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub enum NumberFilter {
    #[serde(rename = "eq")]
    Equal(#[schema(inline)] Number),
    #[serde(rename = "gt")]
    GreaterThan(#[schema(inline)] Number),
    #[serde(rename = "gte")]
    GreaterThanOrEqual(#[schema(inline)] Number),
    #[serde(rename = "lt")]
    LessThan(#[schema(inline)] Number),
    #[serde(rename = "lte")]
    LessThanOrEqual(#[schema(inline)] Number),
    #[serde(rename = "between")]
    Between(#[schema(inline)] (Number, Number)),
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

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
}
