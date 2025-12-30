use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context as _, Result};
use oramacore_lib::filters::FilterResult;
use tokio::join;
use tracing::debug;

use crate::{
    ai::llms,
    collection_manager::{
        bm25::BM25Scorer,
        sides::read::index::committed_field::{StringSearchParams, VectorSearchParams},
    },
    python::embeddings::Intent,
    types::{
        DocumentId, FieldId, FulltextMode, HybridMode, IndexId, Limit, Properties, SearchMode,
        SearchModeResult, Similarity, Threshold, VectorMode,
    },
};

use super::{
    merge::Field, path_to_index_id_map::PathToIndexId, CommittedFields, FieldType, ReadSideContext,
    TextParser, UncommittedFields,
};

/// Parameters required for token score calculation.
///
/// This structure extracts only the necessary fields from SearchParams,
/// using lifetimes to avoid cloning data.
pub struct TokenScoreParams<'search> {
    /// The search mode (fulltext, vector, hybrid, or auto)
    pub mode: &'search SearchMode,
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
    uncommitted_fields: &'index UncommittedFields,
    committed_fields: &'index CommittedFields,
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
        uncommitted_fields: &'index UncommittedFields,
        committed_fields: &'index CommittedFields,
        text_parser: &'index TextParser,
        context: &'index ReadSideContext,
        path_to_field_id_map: &'index PathToIndexId,
    ) -> Self {
        Self {
            index_id,
            document_count,
            uncommitted_fields,
            committed_fields,
            text_parser,
            context,
            path_to_field_id_map,
        }
    }

    /// Determines the actual search mode to use, handling auto-mode via LLM.
    ///
    /// When the mode is Auto, this method calls the LLM service to determine
    /// whether to use fulltext, vector, or hybrid search based on the query.
    async fn determine_search_mode(&self, mode: &SearchMode) -> Result<SearchMode> {
        match mode {
            SearchMode::Auto(mode_result) => {
                let final_mode: String = self
                    .context
                    .llm_service
                    .run_known_prompt(
                        llms::KnownPrompts::Autoquery,
                        vec![],
                        vec![("query".to_string(), mode_result.term.clone())],
                        None,
                        None,
                    )
                    .await?;

                let serilized_final_mode: SearchModeResult = serde_json::from_str(&final_mode)?;

                match serilized_final_mode.mode.as_str() {
                    "fulltext" => Ok(SearchMode::FullText(FulltextMode {
                        term: mode_result.term.clone(),
                        threshold: None,
                        exact: false,
                        tolerance: None,
                    })),
                    "hybrid" => Ok(SearchMode::Hybrid(HybridMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity::default(),
                        threshold: None,
                        exact: false,
                        tolerance: None,
                    })),
                    "vector" => Ok(SearchMode::Vector(VectorMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity::default(),
                    })),
                    _ => bail!("Invalid search mode"),
                }
            }
            _ => Ok(mode.clone()),
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

    /// Calculates vector field properties by collecting all vector fields from
    /// uncommitted and committed field collections.
    async fn calculate_vector_properties(&self) -> Result<Vec<FieldId>> {
        let properties: HashSet<_> = self
            .uncommitted_fields
            .vector_fields
            .keys()
            .chain(self.committed_fields.vector_fields.keys())
            .copied()
            .collect();

        Ok(properties.into_iter().collect())
    }

    /// Calculates string field properties based on the Properties specification.
    ///
    /// Handles three cases:
    /// - Specified: Uses the provided field names (validates they exist and are strings)
    /// - None/Star: Uses all string fields from uncommitted and committed collections
    async fn calculate_string_properties(&self, properties: &Properties) -> Result<Vec<FieldId>> {
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
                .uncommitted_fields
                .string_fields
                .keys()
                .chain(self.committed_fields.string_fields.keys())
                .copied()
                .collect(),
        };

        Ok(properties.into_iter().collect())
    }

    /// Performs full-text search using BM25 scoring algorithm.
    ///
    /// This method tokenizes the search term, searches across all specified properties,
    /// and computes BM25 scores for matching documents. It handles both uncommitted
    /// and committed field data.
    async fn search_full_text(
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
                .flat_map(|e| std::iter::once(e.0).chain(e.1.into_iter()))
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

        let mut field_global_info: HashMap<FieldId, _> = HashMap::new();
        let mut corpus_dfs: HashMap<String, usize> = HashMap::new();

        // Single pass to compute both global info and corpus document frequencies
        for token in &tokens {
            let mut total_corpus_docs = HashSet::new();

            for field_id in &properties {
                let Some(field) = self.uncommitted_fields.string_fields.get(field_id) else {
                    continue;
                };
                let committed = self.committed_fields.string_fields.get(field_id);

                // Compute global info only once per field
                if !field_global_info.contains_key(field_id) {
                    let global_info = if let Some(committed) = committed {
                        committed.stats().global_info.clone() + field.global_info()
                    } else {
                        field.global_info()
                    };
                    field_global_info.insert(*field_id, global_info);
                }

                // Collect unique document IDs for this term across all fields for corpus DF calculation
                let mut corpus_scorer = BM25Scorer::plain();
                let single_token_slice = std::slice::from_ref(token);
                let corpus_context = StringSearchParams {
                    tokens: single_token_slice,
                    exact_match: exact,
                    boost: 1.0,
                    field_id: *field_id,
                    global_info: field_global_info[field_id].clone(),
                    filtered_doc_ids,
                    tolerance,
                };

                field
                    .search(&corpus_context, &mut corpus_scorer)
                    .context("Cannot perform corpus search")?;
                if let Some(committed_field) = committed {
                    committed_field
                        .search(&corpus_context, &mut corpus_scorer)
                        .context("Cannot perform corpus search")?;
                }

                // Add documents found in this field to the corpus count
                total_corpus_docs.extend(corpus_scorer.get_scores().keys().cloned());
            }

            corpus_dfs.insert(token.clone(), total_corpus_docs.len().max(1));
        }

        // Now perform the actual scoring with pre-computed corpus frequencies
        for (term_index, token) in tokens.iter().enumerate() {
            scorer.reset_term();
            let corpus_df = corpus_dfs[token];

            // Then, add field contributions for this term with filtering applied
            for field_id in &properties {
                let Some(field) = self.uncommitted_fields.string_fields.get(field_id) else {
                    continue;
                };
                let committed = self.committed_fields.string_fields.get(field_id);

                let global_info = field_global_info[field_id].clone();

                let boost = boost.get(field_id).copied().unwrap_or(1.0);

                // Reuse the same token slice to avoid allocation
                let single_token_slice = std::slice::from_ref(token);

                let context = StringSearchParams {
                    tokens: single_token_slice,
                    exact_match: exact,
                    boost,
                    field_id: *field_id,
                    global_info,
                    filtered_doc_ids,
                    tolerance,
                };

                // Search this field for this specific term (with filters applied)
                field
                    .search(&context, &mut scorer)
                    .context("Cannot perform search")?;

                if let Some(committed_field) = committed {
                    committed_field
                        .search(&context, &mut scorer)
                        .context("Cannot perform search")?;
                }
            }

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
    /// search across all specified vector fields. It handles both uncommitted and
    /// committed field data.
    async fn search_vector(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        similarity: Similarity,
        limit: Limit,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut output: HashMap<DocumentId, f32> = HashMap::new();

        for field_id in properties {
            let Some(uncommitted) = self.uncommitted_fields.vector_fields.get(&field_id) else {
                bail!("Cannot search on field {field_id:?}: unknown field");
            };
            let committed = self.committed_fields.vector_fields.get(&field_id);

            let model = uncommitted.get_model();

            // We don't cache the embedding.
            // We can do that because, for now, an index and an collection has only one embedding field.
            // Anyway, if the user seach in different index, we are re-calculating the embedding.
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

                uncommitted.search(&params, &mut output)?;
                if let Some(committed) = committed {
                    committed.search(&params, &mut output)?;
                }
            }
        }

        Ok(output)
    }

    /// Performs hybrid search combining full-text and vector search.
    ///
    /// This method executes both vector and full-text searches in parallel,
    /// normalizes their scores using min-max normalization, and combines them.
    async fn search_hybrid(
        &self,
        mode: &HybridMode,
        properties: &Properties,
        boost: HashMap<FieldId, f32>,
        limit: Limit,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let vector_properties = self.calculate_vector_properties().await?;
        let string_properties = self.calculate_string_properties(properties).await?;

        // Execute both searches in parallel
        let (vector_res, fulltext_res) = join!(
            self.search_vector(
                &mode.term,
                vector_properties,
                mode.similarity,
                limit,
                filtered_doc_ids,
            ),
            self.search_full_text(
                &mode.term,
                mode.threshold,
                mode.exact,
                mode.tolerance,
                string_properties,
                boost,
                filtered_doc_ids,
            )
        );

        let vector_results = vector_res?;
        let fulltext_results = fulltext_res?;

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
    /// * `search_document_context` - Context containing filtered/deleted document information
    /// * `results` - Mutable map to populate with document IDs and their scores
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if any search operation fails.
    pub async fn execute(
        self,
        params: &TokenScoreParams<'_>,
        results: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        // Determine the actual search mode (handle auto-mode with LLM)
        let search_mode = self.determine_search_mode(params.mode).await?;

        // Calculate boost values
        let boost = self.calculate_boost(params.boost);

        // Dispatch to appropriate search strategy based on mode
        match search_mode {
            SearchMode::Default(mode) | SearchMode::FullText(mode) => {
                let properties = self.calculate_string_properties(params.properties).await?;
                results.extend(
                    self.search_full_text(
                        &mode.term,
                        mode.threshold,
                        mode.exact,
                        mode.tolerance,
                        properties,
                        boost,
                        params.filtered_doc_ids,
                    )
                    .await?,
                )
            }
            SearchMode::Vector(mode) => {
                let vector_properties = self.calculate_vector_properties().await?;
                results.extend(
                    self.search_vector(
                        &mode.term,
                        vector_properties,
                        mode.similarity,
                        params.limit_hint,
                        params.filtered_doc_ids,
                    )
                    .await?,
                )
            }
            SearchMode::Hybrid(mode) => {
                results.extend(
                    self.search_hybrid(
                        &mode,
                        params.properties,
                        boost,
                        params.limit_hint,
                        params.filtered_doc_ids,
                    )
                    .await?,
                );
            }
            SearchMode::Auto(_) => unreachable!(),
        };

        // Track metrics
        self.track_metrics(results.len());

        debug!("token_scores: {:?}", results);

        Ok(())
    }
}
