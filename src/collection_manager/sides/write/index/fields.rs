use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    sync::Arc,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::mpsc::Sender;

use crate::{
    ai::OramaModel,
    collection_manager::sides::{
        hooks::{HooksRuntime, SelectEmbeddingPropertiesReturnType},
        write::embedding::MultiEmbeddingCalculationRequest,
        OperationSender, OramaModelSerializable, Term, TermStringField,
    },
    nlp::{
        chunker::{Chunker, ChunkerConfig},
        locales::Locale,
        TextParser,
    },
    types::{CollectionId, DocumentId, FieldId, IndexId, Number, SerializableNumber},
};

use super::{get_value, CreateIndexEmbeddingFieldDefintionRequest, EmbeddingStringCalculation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldType {
    Filter(FilterFieldType),
    Score(ScoreFieldType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum FilterFieldType {
    #[serde(rename = "number")]
    Number,
    #[serde(rename = "bool")]
    Bool,
    #[serde(rename = "filter_string")]
    String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ScoreFieldType {
    #[serde(rename = "string")]
    String,
    #[serde(rename = "embedding")]
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedFilterFieldIndexer {
    field_id: FieldId,
    field_path: Box<[String]>,
    is_array: bool,
    field_type: FilterFieldType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializedFilterFieldType {
    Number(SerializedFilterFieldIndexer),
    Bool(SerializedFilterFieldIndexer),
    String(SerializedFilterFieldIndexer),
}

#[derive(Debug, Serialize, Deserialize)]
enum SerializedEmbeddingStringCalculation {
    AllProperties,
    Properties(Box<[Box<[String]>]>),
    Hook,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SerializedScoreFieldType {
    String(SerializedFilterFieldIndexer, Locale),
    Embedding(
        SerializedFilterFieldIndexer,
        OramaModelSerializable,
        SerializedEmbeddingStringCalculation,
    ),
}

pub trait GenericField {
    fn field_id(&self) -> FieldId;
    fn field_path(&self) -> &[String];
    fn is_array(&self) -> bool;
    fn field_type(&self) -> FieldType;

    async fn index_value(
        &self,
        doc_id: DocumentId,
        doc: &Map<String, Value>,
    ) -> Result<Vec<IndexedValue>>;
}

pub enum IndexFilterField {
    Number(NumberFilterField),
    Bool(BoolFilterField),
    String(StringFilterField),
}

impl GenericField for IndexFilterField {
    fn field_id(&self) -> FieldId {
        match self {
            IndexFilterField::Number(field) => field.field_id,
            IndexFilterField::Bool(field) => field.field_id,
            IndexFilterField::String(field) => field.field_id,
        }
    }

    fn field_path(&self) -> &[String] {
        match self {
            IndexFilterField::Number(field) => &field.field_path,
            IndexFilterField::Bool(field) => &field.field_path,
            IndexFilterField::String(field) => &field.field_path,
        }
    }

    fn is_array(&self) -> bool {
        match self {
            IndexFilterField::Number(field) => field.is_array,
            IndexFilterField::Bool(field) => field.is_array,
            IndexFilterField::String(field) => field.is_array,
        }
    }

    fn field_type(&self) -> FieldType {
        match self {
            IndexFilterField::Number(_) => FieldType::Filter(FilterFieldType::Number),
            IndexFilterField::Bool(_) => FieldType::Filter(FilterFieldType::Bool),
            IndexFilterField::String(_) => FieldType::Filter(FilterFieldType::String),
        }
    }

    async fn index_value(
        &self,
        _doc_id: DocumentId,
        doc: &Map<String, Value>,
    ) -> Result<Vec<IndexedValue>> {
        let Some(value) = get_value(doc, self.field_path()) else {
            return Ok(vec![]);
        };
        match self {
            IndexFilterField::Number(field) => field.index_value(value),
            IndexFilterField::Bool(field) => field.index_value(value),
            IndexFilterField::String(field) => field.index_value(value),
        }
    }
}

impl IndexFilterField {
    pub fn new_number(field_id: FieldId, field_path: Box<[String]>) -> Self {
        IndexFilterField::Number(NumberFilterField::new(field_id, field_path, false))
    }
    pub fn new_number_arr(field_id: FieldId, field_path: Box<[String]>) -> Self {
        IndexFilterField::Number(NumberFilterField::new(field_id, field_path, true))
    }
    pub fn new_bool(field_id: FieldId, field_path: Box<[String]>) -> Self {
        IndexFilterField::Bool(BoolFilterField::new(field_id, field_path, false))
    }
    pub fn new_bool_arr(field_id: FieldId, field_path: Box<[String]>) -> Self {
        IndexFilterField::Bool(BoolFilterField::new(field_id, field_path, true))
    }
    pub fn new_string(field_id: FieldId, field_path: Box<[String]>) -> Self {
        IndexFilterField::String(StringFilterField::new(field_id, field_path, false))
    }
    pub fn new_string_arr(field_id: FieldId, field_path: Box<[String]>) -> Self {
        IndexFilterField::String(StringFilterField::new(field_id, field_path, true))
    }

    pub fn serialize(&self) -> SerializedFilterFieldType {
        match self {
            IndexFilterField::Number(_) => {
                SerializedFilterFieldType::Number(SerializedFilterFieldIndexer {
                    field_id: self.field_id(),
                    field_path: self.field_path().to_vec().into_boxed_slice(),
                    is_array: self.is_array(),
                    field_type: FilterFieldType::Number,
                })
            }
            IndexFilterField::Bool(_) => {
                SerializedFilterFieldType::Bool(SerializedFilterFieldIndexer {
                    field_id: self.field_id(),
                    field_path: self.field_path().to_vec().into_boxed_slice(),
                    is_array: self.is_array(),
                    field_type: FilterFieldType::Bool,
                })
            }
            IndexFilterField::String(_) => {
                SerializedFilterFieldType::String(SerializedFilterFieldIndexer {
                    field_id: self.field_id(),
                    field_path: self.field_path().to_vec().into_boxed_slice(),
                    is_array: self.is_array(),
                    field_type: FilterFieldType::String,
                })
            }
        }
    }

    pub fn load_from(dump: SerializedFilterFieldType) -> Self {
        match dump {
            SerializedFilterFieldType::Number(field) => IndexFilterField::Number(
                NumberFilterField::new(field.field_id, field.field_path, field.is_array),
            ),
            SerializedFilterFieldType::Bool(field) => IndexFilterField::Bool(BoolFilterField::new(
                field.field_id,
                field.field_path,
                field.is_array,
            )),
            SerializedFilterFieldType::String(field) => IndexFilterField::String(
                StringFilterField::new(field.field_id, field.field_path, field.is_array),
            ),
        }
    }
}

struct NumberFilterField {
    field_id: FieldId,
    field_path: Box<[String]>,
    is_array: bool,
}

impl NumberFilterField {
    pub fn new(field_id: FieldId, field_path: Box<[String]>, is_array: bool) -> Self {
        Self {
            field_id,
            field_path,
            is_array,
        }
    }

    pub fn index_value(&self, value: &Value) -> Result<Vec<IndexedValue>> {
        let data: Vec<Number> = match value {
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    vec![Number::I32(i as i32)]
                } else if let Some(f) = n.as_f64() {
                    vec![Number::F32(f as f32)]
                } else {
                    vec![]
                }
            }
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| {
                    if let Value::Number(n) = v {
                        if let Some(i) = n.as_i64() {
                            Some(Number::I32(i as i32))
                        } else {
                            n.as_f64().map(|f| Number::F32(f as f32))
                        }
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            _ => vec![],
        };

        let data = data
            .into_iter()
            .map(|num| IndexedValue::FilterNumber(self.field_id, SerializableNumber(num)))
            .collect();

        Ok(data)
    }
}

struct BoolFilterField {
    field_id: FieldId,
    field_path: Box<[String]>,
    is_array: bool,
}

impl BoolFilterField {
    pub fn new(field_id: FieldId, field_path: Box<[String]>, is_array: bool) -> Self {
        Self {
            field_id,
            field_path,
            is_array,
        }
    }

    pub fn index_value(&self, value: &Value) -> Result<Vec<IndexedValue>> {
        let data: Vec<bool> = match value {
            Value::Bool(b) => vec![*b],
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| {
                    if let Value::Bool(b) = v {
                        Some(*b)
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        };

        let data = data
            .into_iter()
            .map(|b| IndexedValue::FilterBool(self.field_id, b))
            .collect();

        Ok(data)
    }
}

struct StringFilterField {
    field_id: FieldId,
    field_path: Box<[String]>,
    is_array: bool,
}

impl StringFilterField {
    pub fn new(field_id: FieldId, field_path: Box<[String]>, is_array: bool) -> Self {
        Self {
            field_id,
            field_path,
            is_array,
        }
    }

    pub fn index_value(&self, value: &Value) -> Result<Vec<IndexedValue>> {
        let data: Vec<String> = match value {
            Value::String(s) => vec![s.clone()],
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| {
                    if let Value::String(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        };

        let data = data
            .into_iter()
            // TODO: put this "25" in the collection config
            .filter(|s| !s.is_empty() && s.len() < 25)
            .map(|s| IndexedValue::FilterString(self.field_id, s))
            .collect();

        Ok(data)
    }
}

pub enum IndexScoreField {
    String(StringScoreField),
    Embedding(EmbeddingField),
}

impl GenericField for IndexScoreField {
    fn field_id(&self) -> FieldId {
        match self {
            IndexScoreField::String(field) => field.field_id,
            IndexScoreField::Embedding(field) => field.field_id,
        }
    }

    fn field_path(&self) -> &[String] {
        match self {
            IndexScoreField::String(field) => &field.field_path,
            IndexScoreField::Embedding(field) => &field.field_path,
        }
    }

    fn is_array(&self) -> bool {
        match self {
            IndexScoreField::String(field) => field.is_array,
            IndexScoreField::Embedding(_) => false,
        }
    }

    fn field_type(&self) -> FieldType {
        match self {
            IndexScoreField::String(_) => FieldType::Score(ScoreFieldType::String),
            IndexScoreField::Embedding(_) => FieldType::Score(ScoreFieldType::Embedding),
        }
    }

    async fn index_value(
        &self,
        doc_id: DocumentId,
        doc: &Map<String, Value>,
    ) -> Result<Vec<IndexedValue>> {
        match self {
            IndexScoreField::String(field) => {
                let Some(value) = get_value(doc, &field.field_path) else {
                    return Ok(vec![]);
                };
                field.index_value(value).await
            }
            IndexScoreField::Embedding(field) => field.index_value(doc_id, doc).await,
        }
    }
}

impl IndexScoreField {
    pub fn new_string(
        field_id: FieldId,
        field_path: Box<[String]>,
        parser: Arc<TextParser>,
    ) -> Self {
        IndexScoreField::String(StringScoreField::new(field_id, field_path, false, parser))
    }
    pub fn new_string_arr(
        field_id: FieldId,
        field_path: Box<[String]>,
        parser: Arc<TextParser>,
    ) -> Self {
        IndexScoreField::String(StringScoreField::new(field_id, field_path, true, parser))
    }
    pub fn new_embedding(
        collection_id: CollectionId,
        index_id: IndexId,
        field_id: FieldId,
        field_path: Box<[String]>,
        model: OramaModel,
        calculation: EmbeddingStringCalculation,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        op_sender: OperationSender,
    ) -> Self {
        IndexScoreField::Embedding(EmbeddingField::new(
            collection_id,
            index_id,
            field_id,
            field_path,
            model,
            calculation,
            embedding_sender,
            op_sender,
        ))
    }

    pub fn switch_to_embedding_hook(&mut self, hooks_runtime: Arc<HooksRuntime>) {
        if let IndexScoreField::Embedding(field) = self {
            field.switch_to_embedding_hook(hooks_runtime);
        }
    }

    pub fn get_embedding_field_definition(
        &self,
    ) -> Option<CreateIndexEmbeddingFieldDefintionRequest> {
        match self {
            IndexScoreField::String(_) => None,
            IndexScoreField::Embedding(field) => {
                let r = CreateIndexEmbeddingFieldDefintionRequest {
                    field_path: field.field_path.clone(),
                    model: field.model,
                    string_calculation: field.calculation.clone(),
                };

                Some(r)
            }
        }
    }

    pub fn serialize(&self) -> SerializedScoreFieldType {
        match self {
            IndexScoreField::String(f) => SerializedScoreFieldType::String(
                SerializedFilterFieldIndexer {
                    field_id: self.field_id(),
                    field_path: self.field_path().to_vec().into_boxed_slice(),
                    is_array: self.is_array(),
                    field_type: FilterFieldType::Number,
                },
                f.parser.locale(),
            ),
            IndexScoreField::Embedding(f) => SerializedScoreFieldType::Embedding(
                SerializedFilterFieldIndexer {
                    field_id: self.field_id(),
                    field_path: self.field_path().to_vec().into_boxed_slice(),
                    is_array: self.is_array(),
                    field_type: FilterFieldType::Number,
                },
                OramaModelSerializable(f.model),
                match &f.calculation {
                    EmbeddingStringCalculation::AllProperties => {
                        SerializedEmbeddingStringCalculation::AllProperties
                    }
                    EmbeddingStringCalculation::Properties(v) => {
                        SerializedEmbeddingStringCalculation::Properties(v.clone())
                    }
                    EmbeddingStringCalculation::Hook(_) => {
                        SerializedEmbeddingStringCalculation::Hook
                    }
                },
            ),
        }
    }

    pub fn load_from(
        dump: SerializedScoreFieldType,
        collection_id: CollectionId,
        index_id: IndexId,
        hooks_runtime: Arc<HooksRuntime>,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        op_sender: OperationSender,
    ) -> Self {
        match dump {
            SerializedScoreFieldType::String(field, locale) => {
                if field.is_array {
                    Self::new_string_arr(
                        field.field_id,
                        field.field_path,
                        Arc::new(TextParser::from_locale(locale)),
                    )
                } else {
                    // TextParser is high cost and here we create everytime the same parser and put it into an Arc
                    // This is not good
                    // TODO: think about a better way to do this
                    Self::new_string(
                        field.field_id,
                        field.field_path,
                        Arc::new(TextParser::from_locale(locale)),
                    )
                }
            }
            SerializedScoreFieldType::Embedding(field, model, calc) => Self::new_embedding(
                collection_id,
                index_id,
                field.field_id,
                field.field_path,
                model.0,
                match calc {
                    SerializedEmbeddingStringCalculation::AllProperties => {
                        EmbeddingStringCalculation::AllProperties
                    }
                    SerializedEmbeddingStringCalculation::Properties(v) => {
                        EmbeddingStringCalculation::Properties(v)
                    }
                    SerializedEmbeddingStringCalculation::Hook => {
                        EmbeddingStringCalculation::Hook(hooks_runtime)
                    }
                },
                embedding_sender,
                op_sender,
            ),
        }
    }
}

struct StringScoreField {
    field_id: FieldId,
    field_path: Box<[String]>,
    is_array: bool,
    parser: Arc<TextParser>,
}
impl StringScoreField {
    pub fn new(
        field_id: FieldId,
        field_path: Box<[String]>,
        is_array: bool,
        parser: Arc<TextParser>,
    ) -> Self {
        Self {
            field_id,
            field_path,
            is_array,
            parser,
        }
    }

    pub async fn index_value(&self, value: &Value) -> Result<Vec<IndexedValue>> {
        let data = match value {
            Value::String(s) => self.parser.tokenize_and_stem(s),
            Value::Array(arr) => {
                let all_string_field = arr.iter().filter_map(|v| {
                    if let Value::String(s) = v {
                        Some(s)
                    } else {
                        None
                    }
                });

                let mut data = Vec::new();
                for value in all_string_field {
                    data.extend(self.parser.tokenize_and_stem(value));
                }
                data
            }
            _ => return Ok(vec![]),
        };

        let field_length = data.len().min(u16::MAX as usize - 1) as u16;
        let mut terms: HashMap<Term, TermStringField> = Default::default();
        for (position, (original, stemmeds)) in data.into_iter().enumerate() {
            // This `for` loop wants to build the `terms` hashmap
            // it is a `HashMap<String, (u32, HashMap<(DocumentId, FieldId), Posting>)>`
            // that means we:
            // term as string -> (term count, HashMap<(DocumentId, FieldId), Posting>)
            // Here we don't want to store Posting into PostingListStorage,
            // that is business of the IndexReader.
            // Instead, here we want to extrapolate data from the document.
            // The real storage leaves in the IndexReader.
            // `original` & `stemmeds` appears in the `terms` hashmap with the "same value"
            // ie: the position of the origin and stemmed term are the same.

            let original = Term(original);
            match terms.entry(original) {
                Entry::Occupied(mut entry) => {
                    let p: &mut TermStringField = entry.get_mut();

                    p.positions.push(position);
                }
                Entry::Vacant(entry) => {
                    let p = TermStringField {
                        positions: vec![position],
                    };
                    entry.insert(p);
                }
            };

            for stemmed in stemmeds {
                let stemmed = Term(stemmed);
                match terms.entry(stemmed) {
                    Entry::Occupied(mut entry) => {
                        let p: &mut TermStringField = entry.get_mut();
                        p.positions.push(position);
                    }
                    Entry::Vacant(entry) => {
                        let p = TermStringField {
                            positions: vec![position],
                        };
                        entry.insert(p);
                    }
                };
            }
        }

        Ok(vec![IndexedValue::ScoreString(
            self.field_id,
            field_length,
            terms,
        )])
    }
}

struct EmbeddingField {
    field_id: FieldId,
    collection_id: CollectionId,
    index_id: IndexId,
    field_path: Box<[String]>,
    model: OramaModel,
    chunker: Chunker,
    calculation: EmbeddingStringCalculation,
    embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
    op_sender: OperationSender,
}
impl EmbeddingField {
    pub fn new(
        collection_id: CollectionId,
        index_id: IndexId,
        field_id: FieldId,
        field_path: Box<[String]>,
        model: OramaModel,
        calculation: EmbeddingStringCalculation,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        op_sender: OperationSender,
    ) -> Self {
        let max_tokens = model.senquence_length();
        let overlap = model.overlap();

        let chunker = Chunker::try_new(ChunkerConfig {
            max_tokens,
            overlap: Some(overlap),
        })
        .expect("Hardcoded 'senquence_length' and 'overlap' value have to be valid");

        Self {
            collection_id,
            index_id,
            field_id,
            field_path,
            model,
            chunker,
            calculation,
            embedding_sender,
            op_sender,
        }
    }

    fn switch_to_embedding_hook(&mut self, hooks_runtime: Arc<HooksRuntime>) {
        self.calculation = EmbeddingStringCalculation::Hook(hooks_runtime);
    }

    async fn index_value(
        &self,
        doc_id: DocumentId,
        doc: &Map<String, Value>,
    ) -> Result<Vec<IndexedValue>> {
        let input: String = match &self.calculation {
            EmbeddingStringCalculation::AllProperties => {
                fn recursive_object_inspection<'doc>(
                    obj: &'doc Map<String, Value>,
                    result: &mut Vec<&'doc String>,
                ) {
                    for (_, value) in obj.iter() {
                        match value {
                            Value::String(s) => {
                                result.push(s);
                            }
                            Value::Object(inner_obj) => {
                                recursive_object_inspection(inner_obj, result);
                            }
                            Value::Array(arr) => {
                                result.extend(arr.iter().filter_map(|v| {
                                    if let Value::String(s) = v {
                                        Some(s)
                                    } else {
                                        None
                                    }
                                }));
                            }
                            _ => {}
                        }
                    }
                }

                let mut result = vec![];
                recursive_object_inspection(doc, &mut result);

                join_vec_strings(&result)
            }
            EmbeddingStringCalculation::Properties(v) => {
                fn recursive_object_inspection<'doc>(
                    obj: &'doc Map<String, Value>,
                    path: &[String],
                    path_index: usize,
                    result: &mut Vec<&'doc String>,
                ) {
                    let key = &path[path_index];
                    let Some(v) = obj.get(key) else {
                        // Nothig to add to result
                        return;
                    };

                    if path.len() == path_index {
                        if let Value::String(s) = v {
                            result.push(s);
                        }
                        if let Value::Array(arr) = v {
                            result.extend(arr.iter().filter_map(|v| {
                                if let Value::String(s) = v {
                                    Some(s)
                                } else {
                                    None
                                }
                            }));
                        }
                        return;
                    }

                    if let Value::Object(obj) = v {
                        recursive_object_inspection(obj, path, path_index + 1, result);
                    }
                }
                let mut result = vec![];
                for path in v {
                    recursive_object_inspection(doc, path, 0, &mut result);
                }

                join_vec_strings(&result)
            }
            EmbeddingStringCalculation::Hook(hooks_runtime) => {
                let a = hooks_runtime
                    .calculate_text_for_embedding(self.collection_id, self.index_id, doc.clone())
                    .await;
                match a {
                    Some(Ok(input)) => match input {
                        SelectEmbeddingPropertiesReturnType::Properties(v) => v
                            .iter()
                            .filter_map(|field_name| {
                                let value = doc.get(field_name).and_then(|v| v.as_str());
                                value
                            })
                            .collect(),
                        SelectEmbeddingPropertiesReturnType::Text(v) => v,
                    },
                    Some(Err(e)) => {
                        tracing::error!(error = ?e, "Error calculating text for embedding. Ignored");
                        return Ok(vec![]);
                    }
                    None => {
                        // Nothing to calculate
                        return Ok(vec![]);
                    }
                }
            }
        };

        // The input could be:
        // - empty: we should skip this
        // - "normal": it is ok
        // - "too long": we should chunk it in a smart way

        if input.trim().is_empty() {
            return Ok(vec![]);
        }

        let texts = self.chunker.chunk_text(&input);

        self.embedding_sender
            .send(MultiEmbeddingCalculationRequest {
                model: self.model,
                coll_id: self.collection_id,
                field_id: self.field_id,
                index_id: self.index_id,
                op_sender: self.op_sender.clone(),
                doc_id,
                text: texts,
            })
            .await
            .context("Error sending embedding calculation request")?;

        Ok(vec![])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexedValue {
    FilterNumber(FieldId, SerializableNumber),
    FilterBool(FieldId, bool),
    FilterString(FieldId, String),
    ScoreString(FieldId, u16, HashMap<Term, TermStringField>),
}

fn join_vec_strings(v: &[&String]) -> String {
    let total_capacity = v.iter().map(|s| s.len()).sum::<usize>() + v.len() - 1;

    let mut final_str = String::with_capacity(total_capacity);
    for (i, s) in v.iter().enumerate() {
        if i > 0 {
            final_str.push(' ');
        }
        final_str.push_str(s);
    }
    final_str
}
