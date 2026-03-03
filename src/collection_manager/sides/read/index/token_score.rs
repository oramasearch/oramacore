use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context as _, Result};
use oramacore_fields::string::{Bm25Params, SearchParams};
use oramacore_lib::filters::FilterResult;
use tracing::debug;

use crate::{
    ai::llms::{self, LLMService},
    collection_manager::{
        bm25::BM25Scorer,
        sides::read::index::committed_field::VectorSearchParams,
    },
    python::embeddings::Intent,
    types::{
        DocumentId, FieldId, FulltextMode, HybridMode, IndexId, Limit, Properties, ScoreMode,
        SearchMode, SearchModeResult, Similarity, Threshold, VectorMode,
    },
};

use super::{
    embedding_field::EmbeddingFieldStorage,
    path_to_index_id_map::PathToIndexId,
    string_field::{DocIdSetFilter, StringFieldStorage},
    FieldType, ReadSideContext, TextParser,
};

/// Parameters required for token score calculation.
///
/// This structure extracts only the necessary fields from SearchParams,
/// using lifetimes to avoid cloning data.
pub struct TokenScoreParams<'search> {
    /// The resolved search mode (fulltext, vector, or hybrid, Auto is already resolved)
    pub mode: &'search ScoreMode,
    /// Field-specific boost values
    pub boost: &'search HashMap<String, f32>,
    /// Properties (fields) to search on
    pub properties: &'search Properties,
    /// Maximum number of results
    pub limit_hint: Limit,
    pub filtered_doc_ids: Option<&'search FilterResult<DocumentId>>,
}

/// Context for executing token score calculations.
///
/// Encapsulates all data needed for search scoring operations,
/// maintaining separation between Index state management and scoring logic.
/// Index is responsible for gathering the data and passing it to the
/// TokenScoreContext constructor, maintaining proper encapsulation.
pub struct TokenScoreContext<'index> {
    index_id: IndexId,
    document_count: u64,
    embedding_fields: &'index HashMap<FieldId, EmbeddingFieldStorage>,
    string_fields: &'index HashMap<FieldId, StringFieldStorage>,
    text_parser: &'index TextParser,
    context: &'index ReadSideContext,
    path_to_field_id_map: &'index PathToIndexId,
}

impl<'index> TokenScoreContext<'index> {
    /// Creates a new TokenScoreContext with provided dependencies.
    ///
    /// Index gathers all data and passes it in, maintaining encapsulation.
    /// This constructor accepts all necessary data as parameters, ensuring
    /// that Index internals are not directly accessed. The caller (Index)
    /// is responsible for gathering the data and passing it in.
    pub fn new(
        index_id: IndexId,
        document_count: u64,
        embedding_fields: &'index HashMap<FieldId, EmbeddingFieldStorage>,
        string_fields: &'index HashMap<FieldId, StringFieldStorage>,
        text_parser: &'index TextParser,
        context: &'index ReadSideContext,
        path_to_field_id_map: &'index PathToIndexId,
    ) -> Self {
        Self {
            index_id,
            document_count,
            embedding_fields,
            string_fields,
            text_parser,
            context,
            path_to_field_id_map,
        }
    }

    /// Resolves a SearchMode into a ScoreMode, handling auto-mode via LLM.
    ///
    /// When the mode is Auto, this method calls the LLM service to determine
    /// whether to use fulltext, vector, or hybrid search based on the query.
    /// All other modes are converted directly into their ScoreMode equivalents.
    pub async fn resolve_score_mode(
        llm_service: &LLMService,
        mode: &SearchMode,
    ) -> Result<ScoreMode> {
        match mode {
            SearchMode::Auto(mode_result) => {
                let final_mode: String = llm_service
                    .run_known_prompt(
                        llms::KnownPrompts::Autoquery,
                        vec![],
                        vec![("query".to_string(), mode_result.term.clone())],
                        None,
                        None,
                    )
                    .await?;

                let serialized_final_mode: SearchModeResult = serde_json::from_str(&final_mode)?;

                match serialized_final_mode.mode.as_str() {
                    "fulltext" => Ok(ScoreMode::FullText(FulltextMode {
                        term: mode_result.term.clone(),
                        threshold: None,
                        exact: false,
                        tolerance: None,
                    })),
                    "hybrid" => Ok(ScoreMode::Hybrid(HybridMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity::default(),
                        threshold: None,
                        exact: false,
                        tolerance: None,
                    })),
                    "vector" => Ok(ScoreMode::Vector(VectorMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity::default(),
                    })),
                    _ => bail!("Invalid search mode"),
                }
            }
            SearchMode::FullText(m) => Ok(ScoreMode::FullText(m.clone())),
            SearchMode::Vector(m) => Ok(ScoreMode::Vector(m.clone())),
            SearchMode::Hybrid(m) => Ok(ScoreMode::Hybrid(m.clone())),
            SearchMode::Default(m) => Ok(ScoreMode::Default(m.clone())),
        }
    }

    /// Calculates boost values for fields, mapping field names to field IDs.
    fn calculate_boost(&self, boost: &HashMap<String, f32>) -> HashMap<FieldId, f32> {
        boost
            .iter()
            .filter_map(|(field_name, boost)| {
                let a = self.path_to_field_id_map.get(field_name);
                let field_id = a?.0;
                Some((field_id, *boost))
            })
            .collect()
    }

    /// Calculates vector field properties by collecting all embedding field IDs.
    fn calculate_vector_properties(&self) -> Result<Vec<FieldId>> {
        Ok(self.embedding_fields.keys().copied().collect())
    }

    /// Calculates string field properties based on the Properties specification.
    ///
    /// Handles three cases:
    /// - Specified: Uses the provided field names (validates they exist and are strings)
    /// - None/Star: Uses all string fields from the string_fields map
    fn calculate_string_properties(&self, properties: &Properties) -> Result<Vec<FieldId>> {
        let properties: HashSet<_> = match properties {
            Properties::Specified(properties) => {
                let mut field_ids = HashSet::new();
                for field in properties {
                    let Some((field_id, field_type)) = self.path_to_field_id_map.get(field) else {
                        continue;
                    };
                    if !matches!(field_type, FieldType::String) {
                        continue;
                    }
                    field_ids.insert(field_id);
                }
                field_ids
            }
            Properties::None | Properties::Star => self
                .string_fields
                .keys()
                .copied()
                .collect(),
        };

        Ok(properties.into_iter().collect())
    }

    /// Performs full-text search using BM25F scoring via `collect_contributions()`.
    ///
    /// For each field, calls `collect_contributions()` which returns raw per-token
    /// normalized TF values (already including field boost, length normalization,
    /// and exact_match_boost). These are fed into the BM25Scorer which computes
    /// cross-field IDF saturation.
    fn search_full_text(
        &self,
        term: &str,
        threshold: Option<Threshold>,
        exact: bool,
        tolerance: Option<u8>,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let tokens = self.text_parser.tokenize_and_stem(term);
        let mut tokens: Vec<_> = if exact {
            tokens.into_iter().map(|e| e.0).collect()
        } else {
            tokens
                .into_iter()
                .flat_map(|e| std::iter::once(e.0).chain(e.1))
                .collect()
        };

        if tokens.is_empty() {
            // Ensure we have at least one token otherwise the result will be empty.
            tokens.push(String::from(""));
        }

        let mut scorer: BM25Scorer<DocumentId> = match threshold {
            Some(Threshold(threshold)) => {
                let perc = tokens.len() as f32 * threshold;
                let threshold = perc.floor() as u32;
                BM25Scorer::with_threshold(threshold)
            }
            None => BM25Scorer::plain(),
        };

        // Use the collection's total document count for canonical BM25F
        let total_documents = self.document_count as f32;

        // Build a filter adapter if we have filtered doc IDs
        let filter_adapter = filtered_doc_ids.map(DocIdSetFilter::new);

        // Collect contributions from all fields at once.
        // Each call returns per-token normalized TF values that already include
        // field boost, length normalization, and exact_match_boost.
        let mut field_contributions = Vec::new();
        for field_id in &properties {
            let Some(string_field) = self.string_fields.get(field_id) else {
                continue;
            };
            let field_boost = boost.get(field_id).copied().unwrap_or(1.0);
            let search_params = SearchParams {
                tokens: &tokens,
                exact_match: exact,
                boost: field_boost,
                bm25_params: Bm25Params::default(),
                tolerance: if exact { Some(0) } else { tolerance },
                phrase_boost: None,
            };
            let contributions = match &filter_adapter {
                Some(adapter) => string_field
                    .collect_contributions_with_filter(&search_params, adapter)
                    .context("Cannot collect string field contributions with filter")?,
                None => string_field
                    .collect_contributions(&search_params)
                    .context("Cannot collect string field contributions")?,
            };
            field_contributions.push((*field_id, contributions));
        }

        // Process token by token for cross-field BM25F scoring.
        // For each token, aggregate per_doc_ntf across fields, compute corpus-level
        // document frequency, and finalize with IDF saturation.
        for (term_index, _token) in tokens.iter().enumerate() {
            scorer.reset_term();

            // Collect all unique document IDs across all fields for this token
            // to compute corpus-level document frequency.
            let mut corpus_docs: HashSet<u64> = HashSet::new();

            for (_field_id, contributions) in &field_contributions {
                if let Some(tc) = contributions.token_contributions.get(term_index) {
                    for &(doc_id, ntf) in &tc.per_doc_ntf {
                        corpus_docs.insert(doc_id);
                        // ntf already includes boost + length normalization + exact_match_boost.
                        // Pass weight=1.0 since the boost is already baked in.
                        scorer.add_precomputed_field(
                            DocumentId(doc_id),
                            ntf,
                            1.0,
                        );
                    }
                }
            }

            let corpus_df = corpus_docs.len().max(1);

            // Finalize with corpus-level IDF
            match &scorer {
                BM25Scorer::Plain(_) => {
                    scorer.finalize_term_plain(
                        corpus_df,
                        total_documents,
                        1.2, // k parameter
                        1.0, // phrase boost
                    );
                }
                BM25Scorer::WithThreshold(_) => {
                    scorer.finalize_term(
                        corpus_df,
                        total_documents,
                        1.2,             // k parameter
                        1.0,             // phrase boost
                        1 << term_index, // token_indexes: bit mask indicating which token this is
                    );
                }
            }

            // Move to next term
            scorer.next_term();
        }

        Ok(scorer.get_scores())
    }

    /// Performs vector similarity search using embeddings.
    ///
    /// This method calculates embeddings for the search term and performs similarity
    /// search across all embedding fields via EmbeddingFieldStorage.
    fn search_vector(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        similarity: Similarity,
        limit: Limit,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut output: HashMap<DocumentId, f32> = HashMap::new();

        for field_id in properties {
            let Some(embedding_field) = self.embedding_fields.get(&field_id) else {
                bail!("Cannot search on field {field_id:?}: unknown embedding field");
            };

            let model = embedding_field.model();

            // We don't cache the embedding.
            // We can do that because, for now, an index and a collection has only one embedding field.
            // Anyway, if the user searches in different indexes, we are re-calculating the embedding.
            // We should put a sort of cache here.
            // TODO: think about this.
            let targets = self
                .context
                .python_service
                .embeddings_service
                .calculate_embeddings(vec![term.to_string()], Intent::Query, model)
                .context("Cannot calculate embeddings")?;

            for target in targets {
                let params = VectorSearchParams {
                    target: &target,
                    limit: limit.0,
                    similarity: similarity.0,
                    filtered_doc_ids,
                };

                embedding_field.search(&params, &mut output)?;
            }
        }

        Ok(output)
    }

    /// Performs hybrid search combining full-text and vector search.
    ///
    /// This method executes both vector and full-text searches,
    /// normalizes their scores using min-max normalization, and combines them.
    fn search_hybrid(
        &self,
        mode: &HybridMode,
        properties: &Properties,
        boost: HashMap<FieldId, f32>,
        limit: Limit,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let vector_properties = self.calculate_vector_properties()?;
        let string_properties = self.calculate_string_properties(properties)?;

        let vector_results = self.search_vector(
            &mode.term,
            vector_properties,
            mode.similarity,
            limit,
            filtered_doc_ids,
        )?;
        let fulltext_results = self.search_full_text(
            &mode.term,
            mode.threshold,
            mode.exact,
            mode.tolerance,
            string_properties,
            boost,
            filtered_doc_ids,
        )?;

        // Normalize and combine the results
        Self::normalize_and_combine(vector_results, fulltext_results)
    }

    /// Normalizes and combines vector and fulltext search results.
    ///
    /// Uses min-max normalization to scale both result sets to [0, 1],
    /// then combines them by summing scores for documents that appear in both sets.
    fn normalize_and_combine(
        vector: HashMap<DocumentId, f32>,
        fulltext: HashMap<DocumentId, f32>,
    ) -> Result<HashMap<DocumentId, f32>> {
        // Find min and max across both result sets for normalization
        let max = vector.values().copied().fold(0.0, f32::max);
        let max = max.max(fulltext.values().copied().fold(0.0, f32::max));
        let min = vector.values().copied().fold(0.0, f32::min);
        let min = min.min(fulltext.values().copied().fold(0.0, f32::min));

        // Normalize vector scores
        let vector: HashMap<_, _> = vector
            .into_iter()
            .map(|(k, v)| (k, (v - min) / (max - min)))
            .collect();

        // Normalize fulltext scores
        let mut fulltext: HashMap<_, _> = fulltext
            .into_iter()
            .map(|(k, v)| (k, (v - min) / (max - min)))
            .collect();

        // Combine by adding vector scores to fulltext results
        for (k, v) in vector {
            let e = fulltext.entry(k).or_default();
            *e += v;
        }

        Ok(fulltext)
    }

    /// Tracks metrics for the token score calculation.
    ///
    /// Records the number of matching documents and the percentage of documents matched.
    fn track_metrics(&self, result_count: usize) {
        use crate::metrics::{
            search::{MATCHING_COUNT_CALCULTATION_COUNT, MATCHING_PERC_CALCULATION_COUNT},
            CollectionLabels,
        };

        let index_id = self.index_id.to_string();
        MATCHING_COUNT_CALCULTATION_COUNT.track_usize(
            CollectionLabels {
                collection: index_id.clone(),
            },
            result_count,
        );
        MATCHING_PERC_CALCULATION_COUNT.track(
            CollectionLabels {
                collection: index_id,
            },
            result_count as f64 / self.document_count as f64,
        );
    }

    /// Executes token score calculation for the given search parameters.
    ///
    /// This is the main entry point that orchestrates all search operations.
    ///
    /// # Arguments
    ///
    /// * `params` - The token score parameters (mode, properties, boost, limit)
    /// * `results` - Mutable map to populate with document IDs and their scores
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if any search operation fails.
    pub fn execute(
        self,
        params: &TokenScoreParams<'_>,
        results: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        // Calculate boost values
        let boost = self.calculate_boost(params.boost);

        // Dispatch to appropriate search strategy based on the resolved score mode
        match params.mode {
            ScoreMode::Default(mode) | ScoreMode::FullText(mode) => {
                let properties = self.calculate_string_properties(params.properties)?;
                results.extend(self.search_full_text(
                    &mode.term,
                    mode.threshold,
                    mode.exact,
                    mode.tolerance,
                    properties,
                    boost,
                    params.filtered_doc_ids,
                )?)
            }
            ScoreMode::Vector(mode) => {
                let vector_properties = self.calculate_vector_properties()?;
                results.extend(self.search_vector(
                    &mode.term,
                    vector_properties,
                    mode.similarity,
                    params.limit_hint,
                    params.filtered_doc_ids,
                )?)
            }
            ScoreMode::Hybrid(mode) => {
                results.extend(self.search_hybrid(
                    mode,
                    params.properties,
                    boost,
                    params.limit_hint,
                    params.filtered_doc_ids,
                )?);
            }
        };

        // Track metrics
        self.track_metrics(results.len());

        debug!("token_scores: {:?}", results);

        Ok(())
    }
}
