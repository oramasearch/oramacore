use anyhow::{Context, Result};
use axum_openapi3::utoipa;
use axum_openapi3::utoipa::ToSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RawJSONDocument {
    pub id: Option<String>,
    pub inner: Box<serde_json::value::RawValue>
}

impl TryFrom<serde_json::Value> for RawJSONDocument {
    type Error = anyhow::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        let id = value
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let inner = serde_json::value::to_raw_value(&value)
            .context("Cannot serialize document")?;
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


#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct CollectionId(pub String);

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
        let id = self.inner.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let inner = serde_json::value::to_raw_value(&self.inner)
            .context("Cannot serialize document")?;
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

#[derive(Debug, Clone)]
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
