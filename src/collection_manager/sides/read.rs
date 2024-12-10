use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    fmt::Debug,
    ops::Deref,
    path::PathBuf,
    sync::{atomic::AtomicU32, Arc},
};

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use futures::join;
use ordered_float::NotNan;
use serde::Deserialize;
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{debug, error, info, instrument};

use crate::{
    collection_manager::{
        dto::{
            FacetDefinition, FacetResult, FieldId, Filter, SearchMode, SearchParams, SearchResult,
            SearchResultHit, TokenScore, TypedField,
        },
        CollectionId,
    },
    document_storage::DocumentId,
    embeddings::{EmbeddingService, LoadedModel},
    indexes::{
        bool::BoolIndex,
        number::{NumberFilter, NumberIndex},
        string::{scorer::bm25::BM25Score, StringIndex},
        vector::{VectorIndex, VectorIndexConfig},
    },
    nlp::TextParser,
    types::Document,
};

use super::{
    document_storage::DocumentStorage,
    write::{CollectionWriteOperation, GenericWriteOperation, InsertStringTerms, WriteOperation},
};

#[derive(Debug, Deserialize, Clone)]
pub struct DataConfig {
    pub data_dir: PathBuf,
    pub max_size_per_chunk: usize,
}

pub struct CollectionsReader {
    embedding_service: Arc<EmbeddingService>,
    collections: RwLock<HashMap<CollectionId, CollectionReader>>,
    document_storage: Arc<dyn DocumentStorage>,
    posting_id_generator: Arc<AtomicU32>,
    data_config: DataConfig,
}
impl CollectionsReader {
    pub fn new(
        embedding_service: Arc<EmbeddingService>,
        document_storage: Arc<dyn DocumentStorage>,
        data_config: DataConfig,
    ) -> Self {
        Self {
            embedding_service,
            collections: Default::default(),
            document_storage,
            posting_id_generator: Arc::new(AtomicU32::new(0)),
            data_config,
        }
    }

    pub async fn update(&self, op: WriteOperation) -> Result<()> {
        match op {
            WriteOperation::Generic(GenericWriteOperation::CreateCollection { id }) => {
                info!("CreateCollection {:?}", id);
                let collection_reader = CollectionReader {
                    id: id.clone(),
                    embedding_service: self.embedding_service.clone(),

                    document_storage: Arc::clone(&self.document_storage),

                    // The unwrap here is bad even if it is safe because it never fails
                    // TODO: remove this unwrap
                    vector_index: VectorIndex::try_new(VectorIndexConfig {})
                        .context("Cannot create vector index during collection creation")?,
                    fields_per_model: Default::default(),

                    string_index: StringIndex::new(self.posting_id_generator.clone()),
                    number_index: NumberIndex::new(
                        self.data_config.data_dir.join("numbers"),
                        self.data_config.max_size_per_chunk,
                    )?,
                    bool_index: BoolIndex::new(),

                    fields: Default::default(),
                };

                self.collections.write().await.insert(id, collection_reader);
            }
            WriteOperation::Collection(collection_id, coll_op) => {
                let collections = self.collections.read().await;

                let collection_reader = match collections.get(&collection_id) {
                    Some(collection_reader) => collection_reader,
                    None => {
                        error!(target: "Collection not found", ?collection_id);
                        return Err(anyhow::anyhow!("Collection not found"));
                    }
                };

                match coll_op {
                    CollectionWriteOperation::CreateField {
                        field_id,
                        field_name,
                        field,
                    } => {
                        collection_reader
                            .create_field(field_id, field_name, field)
                            .await
                            .context("Cannot create field")?;
                    }
                    CollectionWriteOperation::IndexEmbedding {
                        doc_id,
                        field_id,
                        value,
                    } => {
                        collection_reader
                            .index_embedding(doc_id, field_id, value)
                            .context("cannot index embedding")?;
                    }
                    CollectionWriteOperation::IndexString {
                        doc_id,
                        field_id,
                        terms,
                    } => {
                        collection_reader
                            .index_string(doc_id, field_id, terms)
                            .await
                            .context("cannot index string")?;
                    }
                    CollectionWriteOperation::InsertDocument { doc_id, doc } => {
                        collection_reader
                            .insert_document(doc_id, doc)
                            .await
                            .context("cannot insert document")?;
                    }
                }
            }
        };

        Ok(())
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionReadLock<'coll>>
    where
        's: 'coll,
    {
        let r = self.collections.read().await;
        CollectionReadLock::try_new(r, id)
    }
}

pub struct CollectionReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
    id: CollectionId,
}

impl<'guard> CollectionReadLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionReader>>,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionReadLock { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for CollectionReadLock<'_> {
    type Target = CollectionReader;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}

pub struct CollectionReader {
    id: CollectionId,
    embedding_service: Arc<EmbeddingService>,

    document_storage: Arc<dyn DocumentStorage>,

    fields: DashMap<String, (FieldId, TypedField)>,

    // indexes
    vector_index: VectorIndex,
    fields_per_model: DashMap<Arc<LoadedModel>, Vec<FieldId>>,

    string_index: StringIndex,
    number_index: NumberIndex,
    bool_index: BoolIndex,
    // TODO: textparser -> vec<field_id>
}

impl CollectionReader {
    async fn create_field(
        &self,
        field_id: FieldId,
        field_name: String,
        field: TypedField,
    ) -> Result<()> {
        self.fields
            .insert(field_name.clone(), (field_id, field.clone()));

        if let TypedField::Embedding(embedding) = field {
            let orama_model = self
                .embedding_service
                .get_model(embedding.model_name)
                .await?;

            self.vector_index
                .add_field(field_id, orama_model.dimensions())?;

            self.fields_per_model
                .entry(orama_model)
                .or_default()
                .push(field_id);
        };

        Ok(())
    }

    fn index_embedding(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        value: Vec<f32>,
    ) -> Result<()> {
        // `insert_batch` is designed to process multiple values at once
        // We are inserting only one value, and this is not good for performance
        // We should add an API to accept a single value and avoid the rebuild step
        // Instead, we could move the "rebuild" logic to the `VectorIndex`
        // TODO: do it.
        self.vector_index
            .insert_batch(vec![(doc_id, field_id, vec![value])])
    }

    #[instrument(skip(self, terms), level="debug", fields(self.id = ?self.id))]
    async fn index_string(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        terms: InsertStringTerms,
    ) -> Result<()> {
        self.string_index.insert(doc_id, field_id, terms).await?;
        Ok(())
    }

    #[instrument(skip(self), level="debug", fields(self.id = ?self.id))]
    async fn insert_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        self.string_index.new_document_inserted().await;
        self.document_storage.add_document(doc_id, doc).await
    }

    fn get_field_id(&self, field_name: String) -> Result<FieldId> {
        let field_id = self.fields.get(&field_name);

        match field_id {
            Some(field_id) => Ok(field_id.0),
            None => Err(anyhow!("Field not found")),
        }
    }

    fn get_field_id_with_type(&self, field_name: &str) -> Result<(FieldId, TypedField)> {
        self.fields
            .get(field_name)
            .map(|v| v.clone())
            .ok_or_else(|| anyhow!("Field not found"))
    }

    fn calculate_boost(&self, boost: HashMap<String, f32>) -> HashMap<FieldId, f32> {
        boost
            .into_iter()
            .filter_map(|(field_name, boost)| {
                let field_id = self.get_field_id(field_name).ok()?;
                Some((field_id, boost))
            })
            .collect()
    }

    fn calculate_filtered_doc_ids(
        &self,
        where_filter: HashMap<String, Filter>,
    ) -> Result<Option<HashSet<DocumentId>>> {
        if where_filter.is_empty() {
            Ok(None)
        } else {
            let filters: Result<Vec<_>> = where_filter
                .into_iter()
                .map(|(field_name, value)| {
                    self.get_field_id_with_type(&field_name)
                        .with_context(|| format!("Unknown field \"{}\"", &field_name))
                        .map(|(field_id, field_type)| (field_name, field_id, field_type, value))
                })
                .collect();
            let mut filters = filters?;
            let last = filters.pop();

            let (field_name, field_id, field_type, filter) = match last {
                Some(v) => v,
                None => return Err(anyhow!("No filter provided")),
            };

            let mut doc_ids = match (&field_type, filter) {
                (TypedField::Number, Filter::Number(filter_number)) => {
                    self.number_index.filter(field_id, filter_number)?
                }
                (TypedField::Bool, Filter::Bool(filter_bool)) => {
                    self.bool_index.filter(field_id, filter_bool)?
                }
                _ => {
                    error!(
                        "Filter on field {:?}({:?}) not supported",
                        field_name, field_type
                    );
                    anyhow::bail!(
                        "Filter on field {:?}({:?}) not supported",
                        field_name,
                        field_type
                    )
                }
            };
            for (field_name, field_id, field_type, filter) in filters {
                let doc_ids_ = match (&field_type, filter) {
                    (TypedField::Number, Filter::Number(filter_number)) => {
                        self.number_index.filter(field_id, filter_number)?
                    }
                    (TypedField::Bool, Filter::Bool(filter_bool)) => {
                        self.bool_index.filter(field_id, filter_bool)?
                    }
                    _ => {
                        error!(
                            "Filter on field {:?}({:?}) not supported",
                            field_name, field_type
                        );
                        anyhow::bail!(
                            "Filter on field {:?}({:?}) not supported",
                            field_name,
                            field_type
                        )
                    }
                };
                doc_ids = doc_ids.intersection(&doc_ids_).copied().collect();
            }

            info!("Matching doc from filters: {:?}", doc_ids);

            Ok(Some(doc_ids))
        }
    }

    fn calculate_properties(&self, properties: Option<Vec<String>>) -> Result<Vec<FieldId>> {
        let properties: Result<Vec<_>> = match properties {
            Some(properties) => properties
                .into_iter()
                .map(|p| self.get_field_id(p))
                .collect(),
            None => self.fields.iter().map(|e| Ok(e.value().0)).collect(),
        };

        properties
    }

    #[instrument(skip(self), level="debug", fields(self.id = ?self.id))]
    pub async fn search<S: TryInto<SearchParams> + Debug>(
        &self,
        search_params: S,
    ) -> Result<SearchResult, anyhow::Error>
    where
        anyhow::Error: From<S::Error>,
        S::Error: std::fmt::Display,
    {
        let search_params = search_params
            .try_into()
            .map_err(|e| anyhow!("Cannot convert search params: {}", e))?;

        let SearchParams {
            mode,
            properties,
            boost,
            facets,
            limit,
            where_filter,
        } = search_params;

        let filtered_doc_ids = self.calculate_filtered_doc_ids(where_filter)?;
        let boost = self.calculate_boost(boost);
        let properties = self.calculate_properties(properties)?;

        let token_scores = match mode {
            SearchMode::Default(search_params) | SearchMode::FullText(search_params) => {
                self.search_full_text(&search_params.term, properties, boost, filtered_doc_ids)
                    .await?
            }
            SearchMode::Vector(search_params) => {
                self.search_vector(&search_params.term, filtered_doc_ids)
                    .await?
            }
            SearchMode::Hybrid(search_params) => {
                let (vector, fulltext) = join!(
                    self.search_vector(&search_params.term, filtered_doc_ids.clone()),
                    self.search_full_text(&search_params.term, properties, boost, filtered_doc_ids)
                );
                let vector = vector?;
                let fulltext = fulltext?;

                // min-max normalization
                let max = vector.values().copied().fold(0.0, f32::max);
                let max = max.max(fulltext.values().copied().fold(0.0, f32::max));
                let min = vector.values().copied().fold(0.0, f32::min);
                let min = min.min(fulltext.values().copied().fold(0.0, f32::min));

                let vector: HashMap<_, _> = vector
                    .into_iter()
                    .map(|(k, v)| (k, (v - min) / (max - min)))
                    .collect();

                let mut fulltext: HashMap<_, _> = fulltext
                    .into_iter()
                    .map(|(k, v)| (k, (v - min) / (max - min)))
                    .collect();

                for (k, v) in vector {
                    let e = fulltext.entry(k).or_default();
                    *e += v;
                }
                fulltext
            }
        };

        info!("token_scores len: {:?}", token_scores.len());

        debug!("token_scores: {:?}", token_scores);

        let facets = self.calculate_facets(&token_scores, facets)?;

        let count = token_scores.len();

        let top_results = top_n(token_scores, limit.0);

        let docs = self
            .document_storage
            .get_documents_by_ids(top_results.iter().map(|m| m.document_id).collect())
            .await?;

        let hits: Vec<_> = top_results
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

        Ok(SearchResult {
            count,
            hits,
            facets,
        })
    }

    async fn search_full_text(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let text_parser = TextParser::from_language(crate::nlp::locales::Locale::EN);
        let tokens = text_parser.tokenize(term);

        self.string_index
            .search(
                tokens,
                // This option is not required.
                // It was introduced because for test purposes we
                // could avoid to pass every properties
                // Anyway the production code should always pass the properties
                // So we could avoid this option
                // TODO: remove this option
                Some(properties),
                boost,
                BM25Score::default(),
                filtered_doc_ids.as_ref(),
            )
            .await
    }

    async fn search_vector(
        &self,
        term: &str,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut ret: HashMap<DocumentId, f32> = HashMap::new();

        for e in &self.fields_per_model {
            let model = e.key();
            let fields = e.value();

            let e = model.embed(vec![term.to_string()], None)?;

            for k in e {
                let r = self.vector_index.search(fields, &k, 1)?;

                for (doc_id, score) in r {
                    if !filtered_doc_ids
                        .as_ref()
                        .map(|f| f.contains(&doc_id))
                        .unwrap_or(true)
                    {
                        continue;
                    }

                    let v = ret.entry(doc_id).or_default();
                    *v += score;
                }
            }
        }

        Ok(ret)
    }

    fn calculate_facets(
        &self,
        token_scores: &HashMap<DocumentId, f32>,
        facets: HashMap<String, FacetDefinition>,
    ) -> Result<Option<HashMap<String, FacetResult>>> {
        if facets.is_empty() {
            Ok(None)
        } else {
            info!("Computing facets on {:?}", facets.keys());

            let mut res_facets: HashMap<String, FacetResult> = HashMap::new();
            for (field_name, facet) in facets {
                let field_id = self.get_field_id(field_name.clone())?;

                // This calculation is inneficient
                // we have the doc_ids that matches:
                // - filters
                // - search
                // We should use them to calculate the facets
                // Instead here we are building an hashset and
                // iter again on it to filter the doc_ids.
                // We could create a dedicated method in the indexes that
                // accepts the matching doc_ids + facet definition and returns the count
                // TODO: do it
                match facet {
                    FacetDefinition::Number(facet) => {
                        let mut values = HashMap::new();

                        for range in facet.ranges {
                            let facet: HashSet<_> = self
                                .number_index
                                .filter(field_id, NumberFilter::Between((range.from, range.to)))?
                                .into_iter()
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect();

                            values.insert(format!("{}-{}", range.from, range.to), facet.len());
                        }

                        res_facets.insert(
                            field_name,
                            FacetResult {
                                count: values.len(),
                                values,
                            },
                        );
                    }
                    FacetDefinition::Bool => {
                        let true_facet: HashSet<DocumentId> = self
                            .bool_index
                            .filter(field_id, true)?
                            .into_iter()
                            .filter(|doc_id| token_scores.contains_key(doc_id))
                            .collect();
                        let false_facet: HashSet<DocumentId> = self
                            .bool_index
                            .filter(field_id, false)?
                            .into_iter()
                            .filter(|doc_id| token_scores.contains_key(doc_id))
                            .collect();

                        res_facets.insert(
                            field_name,
                            FacetResult {
                                count: 2,
                                values: HashMap::from_iter([
                                    ("true".to_string(), true_facet.len()),
                                    ("false".to_string(), false_facet.len()),
                                ]),
                            },
                        );
                    }
                }
            }
            Ok(Some(res_facets))
        }
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
