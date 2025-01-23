use std::{
    collections::HashMap,
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use serde::{de::Visitor, Deserialize, Deserializer, Serialize};

use crate::collection_manager::dto::FieldId;

pub struct FieldIdHashMap<T>(HashMap<FieldId, T>);

impl<T> FieldIdHashMap<T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn into_inner(self) -> HashMap<FieldId, T> {
        self.0
    }
}

impl<T> Deref for FieldIdHashMap<T> {
    type Target = HashMap<FieldId, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for FieldIdHashMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> FromIterator<(FieldId, T)> for FieldIdHashMap<T> {
    fn from_iter<I: IntoIterator<Item = (FieldId, T)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> From<HashMap<FieldId, T>> for FieldIdHashMap<T> {
    fn from(map: HashMap<FieldId, T>) -> Self {
        Self(map)
    }
}

impl<T: Debug> Debug for FieldIdHashMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_map().entries(self.0.iter()).finish()
    }
}

impl<T: Serialize> Serialize for FieldIdHashMap<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_seq(self.0.iter())
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for FieldIdHashMap<T> {
    fn deserialize<D>(deserializer: D) -> Result<FieldIdHashMap<T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(FieldIdHashMapVisitor(PhantomData))
    }
}

struct FieldIdHashMapVisitor<T>(PhantomData<T>);

impl<'de, T: Deserialize<'de>> Visitor<'de> for FieldIdHashMapVisitor<T> {
    type Value = FieldIdHashMap<T>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("Expect a sequence of (FieldId, T) pairs")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut map = HashMap::new();
        while let Some((k, v)) = seq.next_element()? {
            map.insert(k, v);
        }
        Ok(FieldIdHashMap(map))
    }
}
