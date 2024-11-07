use crate::dto::{CollectionDTO, SearchParams};
use crate::embeddings_management::get_embeddable_string;
use anyhow::{anyhow, Context};
use dashmap::DashMap;
use document_storage::DocumentStorage;
use embeddings::OramaModels;
use fastembed::TextEmbedding;
use nlp::locales::Locale;
use nlp::TextParser;
use ordered_float::NotNan;
use serde_json::Value;
use std::any::Any;
use std::sync::RwLock;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    sync::{atomic::AtomicU16, Arc},
};
use storage::Storage;
use string_index::{
    scorer::{bm25::BM25Score, counter::CounterScore},
    StringIndex,
};
use types::{
    CollectionId, DocumentId, DocumentList, FieldId, ScalarType, SearchResult, SearchResultHit,
    StringParser, TokenScore, ValueType,
};
use vector_index::{VectorIndex, VectorIndexConfig};

pub struct Collection {
    pub(crate) id: CollectionId,
    description: Option<String>,
    string_index: StringIndex,
    language: Locale,
    field_id_generator: AtomicU16,
    string_fields: DashMap<String, FieldId>,
    document_storage: Arc<DocumentStorage>,
    vector_index: RwLock<VectorIndex>,
    embedding_model: Arc<TextEmbedding>,
    parsers: HashMap<FieldId, Box<dyn StringParser>>,
    default_parser: Box<dyn StringParser>,
}

impl Collection {
    pub fn new(
        storage: Arc<Storage>,
        id: CollectionId,
        description: Option<String>,
        language: Locale,
        document_storage: Arc<DocumentStorage>,
        parsers: HashMap<String, Box<dyn StringParser>>,
    ) -> Self {
        // @todo: make the language configurable
        let default_parser = TextParser::from_language(Locale::EN);

        let vector_index = VectorIndex::new(VectorIndexConfig {
            // @todo: make the embeddings model configurable
            embeddings_model: OramaModels::JinaV2BaseCode,
        });

        let embedding_model = vector_index
            .embeddings_model
            .try_new()
            .with_context(|| "Unable to initialize embedding model when creating the collection")
            .unwrap();

        let mut collection = Collection {
            id,
            description,
            language,
            string_fields: Default::default(),
            field_id_generator: AtomicU16::new(0),
            document_storage,
            string_index: StringIndex::new(storage.clone()),
            parsers: Default::default(),
            default_parser: Box::new(default_parser),
            vector_index: RwLock::new(vector_index),
            embedding_model,
        };

        for (field_name, parser) in parsers {
            let field_id = collection.get_field_id(field_name.clone());
            collection.parsers.insert(field_id, parser);
        }

        collection
    }

    pub fn as_dto(&self) -> CollectionDTO {
        CollectionDTO {
            id: self.id.0.clone(),
            description: self.description.clone(),
            language: self.language.into(),
            document_count: self.string_index.get_total_documents(),
            string_fields: self
                .string_fields
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
        }
    }

    pub fn insert_batch(&self, document_list: DocumentList) -> Result<(), anyhow::Error> {
        let mut strings: HashMap<DocumentId, Vec<(FieldId, Vec<(String, Vec<String>)>)>> =
            HashMap::with_capacity(self.string_fields.len());
        let mut documents = Vec::with_capacity(document_list.len());
        let mut vector_index = self.vector_index.write().unwrap();

        for doc in document_list {
            let mut flatten = doc.into_flatten();
            let schema = flatten.get_field_schema();

            let internal_document_id = self.generate_document_id();

            let embeddable_string = get_embeddable_string(&doc);
            let embedding = self
                .embedding_model
                .embed(vec![embeddable_string], Some(1))
                .with_context(|| {
                    "An error occurred while generating the embeddings for a document"
                })?;

            vector_index
                .insert(internal_document_id, embedding[0].to_vec().as_slice())
                .unwrap();

            for (key, field_type) in schema {
                if field_type == ValueType::Scalar(ScalarType::String) {
                    // flatten is a copy of the document, so we can remove the key
                    let value = match flatten.remove(&key) {
                        Some(Value::String(value)) => value,
                        _ => Err(anyhow!("value is not string. This should never happen"))?,
                    };
                    let field_id = self.get_field_id(key.clone());

                    let parser = self.parsers.get(&field_id).unwrap_or(&self.default_parser);

                    let tokens = parser.tokenize_str_and_stem(&value)?;

                    strings
                        .entry(internal_document_id)
                        .or_default()
                        .push((field_id, tokens));
                }
            }

            documents.push((internal_document_id, doc));
        }

        // TODO: if the insert_multiple fails, should we rollback the `add_documents`?
        self.document_storage.add_documents(documents)?;

        self.string_index.insert_multiple(strings)?;

        Ok(())
    }

    pub fn vector_search(
        &self,
        search_params: SearchParams,
    ) -> Result<SearchResult, anyhow::Error> {
        let query_as_embedding = self
            .embedding_model
            .embed(vec![search_params.term], Some(1))?;
        let mut vector_index = self
            .vector_index
            .write()
            .unwrap();

        let knn = vector_index.search(query_as_embedding[0].as_slice(), search_params.limit.0);
        let hits = knn
            .iter()
            .map(|id| {
                let full_document = self
                    .document_storage
                    .get_all(vec![id.clone()])
                    .with_context(|| "Unable to fetch full document")
                    .unwrap();

                SearchResultHit {
                    id: "".to_string(),
                    score: 0.0,
                    document: full_document[0].clone(),
                }
            })
            .collect();

        Ok(SearchResult { count: 0, hits })
    }

    pub fn search(&self, search_params: SearchParams) -> Result<SearchResult, anyhow::Error> {
        // TODO: handle search_params.properties

        let boost: HashMap<_, _> = search_params
            .boost
            .into_iter()
            .map(|(field_name, boost)| {
                let field_id = self.get_field_id(field_name);
                (field_id, boost)
            })
            .collect();
        let properties: Vec<_> = match search_params.properties {
            Some(properties) => properties
                .into_iter()
                .map(|p| self.get_field_id(p))
                .collect(),
            None => self.string_fields.iter().map(|e| *e.value()).collect(),
        };

        let tokens: Vec<_> = self
            .default_parser
            .tokenize_str_and_stem(&search_params.term)?
            .into_iter()
            .flat_map(|(token, stemmed)| {
                let mut terms = vec![token];
                terms.extend(stemmed);
                terms
            })
            .collect();

        let fields_on_search_with_default_parser: Vec<_> = self
            .string_fields
            .iter()
            .filter(|field_id| !self.parsers.contains_key(field_id.value()))
            .filter(|field_id| properties.contains(field_id.value()))
            .map(|field_id| *field_id.value())
            .collect();
        println!(
            "searching for tokens {:?}",
            fields_on_search_with_default_parser,
        );
        let mut token_scores = self.string_index.search(
            tokens,
            Some(fields_on_search_with_default_parser),
            boost.clone(),
            BM25Score,
        )?;
        println!(
            "Element found with default parser: {:?}",
            token_scores.len()
        );

        // Depends on the self.parsers size, this loop can be optimized, parallelizing the search.
        // But for now, we will keep it simple.
        // TODO: think about how to parallelize this search
        for (field_id, parser) in &self.parsers {
            if !properties.contains(field_id) {
                continue;
            }
            let tokens: Vec<_> = parser
                .tokenize_str_and_stem(&search_params.term)?
                .into_iter()
                .flat_map(|(token, stemmed)| {
                    let mut terms = vec![token];
                    terms.extend(stemmed);
                    terms
                })
                .collect();

            let field_token_scores = self.string_index.search(
                tokens,
                Some(vec![*field_id]),
                boost.clone(),
                CounterScore,
            )?;

            let field_name = self
                .string_fields
                .iter()
                .find(|v| v.value() == field_id)
                .unwrap();
            let field_name = field_name.key();
            println!(
                "Element found with parser: {field_id:?} ({:?}) {:?}",
                field_name,
                field_token_scores.len()
            );

            // Merging scores that come from different parsers are hard.
            // Because we are focused on PoC with tanstack, this case happens only with "code" field.
            // We use just a simple counter to merge the scores.
            // Anyway, this it not a good solution for a real-world application.
            // TODO: think about how to merge scores from different parsers
            for (document_id, score) in field_token_scores {
                if let Some(existing_score) = token_scores.get(&document_id) {
                    token_scores.insert(document_id, existing_score + score);
                } else {
                    token_scores.insert(document_id, score);
                }
            }
        }

        let count = token_scores.len();

        let token_scores = top_n(token_scores, search_params.limit.0);

        let docs = self
            .document_storage
            .get_all(token_scores.iter().map(|m| m.document_id).collect())?;

        let hits: Vec<_> = token_scores
            .into_iter()
            .zip(docs)
            .map(|(token_score, document)| {
                let id = document
                    .as_ref()
                    .and_then(|d| d.get("id"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                SearchResultHit {
                    id,
                    score: token_score.score,
                    document,
                }
            })
            .collect();

        Ok(SearchResult { count, hits })
    }

    fn generate_document_id(&self) -> DocumentId {
        self.document_storage.generate_document_id()
    }

    fn get_field_id(&self, field_name: String) -> FieldId {
        if let Some(field_id) = self.string_fields.get(&field_name) {
            return *field_id;
        }

        let field_id = self.string_fields.entry(field_name).or_insert_with(|| {
            let field_id = self
                .field_id_generator
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            FieldId(field_id)
        });

        *field_id
    }
}

fn top_n(map: HashMap<DocumentId, f32>, n: usize) -> Vec<TokenScore> {
    // A min-heap of size `n` to keep track of the top N elements
    let mut heap: BinaryHeap<Reverse<(NotNan<f32>, DocumentId)>> = BinaryHeap::with_capacity(n);

    for (key, value) in map {
        // Insert into the heap if it's not full, or replace the smallest element if the current one is larger
        if heap.len() < n {
            heap.push(Reverse((NotNan::new(value).unwrap(), key)));
        } else if let Some(Reverse((min_value, _))) = heap.peek() {
            if value > *min_value.as_ref() {
                heap.pop();
                heap.push(Reverse((NotNan::new(value).unwrap(), key)));
            }
        }
    }

    // Collect results into a sorted Vec (optional sorting based on descending values)
    let result: Vec<TokenScore> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse((value, key))| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}
