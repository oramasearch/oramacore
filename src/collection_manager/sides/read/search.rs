use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Context, Result};
use ordered_float::NotNan;
use tokio::join;
use tracing::{debug, error, info, instrument};

use super::CollectionReader;
use crate::collection_manager::dto::Limit;
use crate::nlp::locales::Locale;
use crate::{
    capped_heap::CappedHeap,
    collection_manager::dto::{
        FacetDefinition, FacetResult, FieldId, Filter, SearchMode, SearchParams, SearchResult,
        SearchResultHit, TokenScore, TypedField,
    },
    indexes::{number::NumberFilter, string::BM25Scorer},
    metrics::{
        SearchFilterLabels, SearchLabels, SEARCH_FILTER_HISTOGRAM, SEARCH_FILTER_METRIC,
        SEARCH_METRIC,
    },
    types::DocumentId,
};

impl CollectionReader {
    #[instrument(skip(self), level="debug", fields(self.id = ?self.id))]
    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult, anyhow::Error> {
        let metric = SEARCH_METRIC.create(SearchLabels {
            collection: self.id.0.to_string(),
        });
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
                self.search_vector(&search_params.term, filtered_doc_ids, &limit)
                    .await?
            }
            SearchMode::Hybrid(search_params) => {
                let (vector, fulltext) = join!(
                    self.search_vector(&search_params.term, filtered_doc_ids.clone(), &limit),
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

        drop(metric);

        Ok(SearchResult {
            count,
            hits,
            facets,
        })
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
            return Ok(None);
        }

        let metric = SEARCH_FILTER_METRIC.create(SearchFilterLabels {
            collection: self.id.0.to_string(),
        });

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

        info!(
            "Filtering on field {:?}({:?}): {:?}",
            field_name, field_type, filter
        );

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

        drop(metric);

        SEARCH_FILTER_HISTOGRAM
            .create(SearchFilterLabels {
                collection: self.id.0.to_string(),
            })
            .record_usize(doc_ids.len());
        info!("Matching doc from filters: {:?}", doc_ids.len());

        Ok(Some(doc_ids))
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

    async fn search_full_text(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::new();

        let mut tokens_cache: HashMap<Locale, Vec<String>> = Default::default();

        for field_id in properties {
            let text_parser = self.text_parser_per_field.get(&field_id);
            let (locale, text_parser) = match text_parser.as_ref() {
                None => return Err(anyhow!("No text parser for this field")),
                Some(text_parser) => (text_parser.0, &text_parser.1),
            };

            let tokens = tokens_cache
                .entry(locale)
                .or_insert_with(|| text_parser.tokenize(term));

            self.string_index
                .search(
                    tokens,
                    // This option is not required.
                    // It was introduced because for test purposes we
                    // could avoid to pass every properties
                    // Anyway the production code should always pass the properties
                    // So we could avoid this option
                    // TODO: remove this option
                    Some(&[field_id]),
                    &boost,
                    &mut scorer,
                    filtered_doc_ids.as_ref(),
                )
                .await?;
        }

        Ok(scorer.get_scores())
    }

    async fn search_vector(
        &self,
        term: &str,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
        limit: &Limit,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut ret: HashMap<DocumentId, f32> = HashMap::new();

        for e in &self.fields_per_model {
            let model = e.key();
            let fields = e.value();

            let e = model.embed_query(vec![&term.to_string()]).await?;

            for k in e {
                let r = self.vector_index.search(fields, &k, limit.0)?;

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
                    FacetDefinition::Bool(facets) => {
                        let mut values = HashMap::new();

                        if facets.r#true {
                            let true_facet: HashSet<DocumentId> = self
                                .bool_index
                                .filter(field_id, true)?
                                .into_iter()
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect();
                            values.insert("true".to_string(), true_facet.len());
                        }
                        if facets.r#false {
                            let false_facet: HashSet<DocumentId> = self
                                .bool_index
                                .filter(field_id, false)?
                                .into_iter()
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect();
                            values.insert("false".to_string(), false_facet.len());
                        }

                        res_facets.insert(
                            field_name,
                            FacetResult {
                                count: values.len(),
                                values,
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
    let mut capped_heap = CappedHeap::new(n);

    for (key, value) in map {
        let k = match NotNan::new(value) {
            Ok(k) => k,
            Err(_) => continue,
        };
        let v = key;
        capped_heap.insert(k, v);
    }

    let result: Vec<TokenScore> = capped_heap
        .into_top()
        .map(|(value, key)| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}
