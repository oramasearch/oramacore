use anyhow::{Context, Result};
use axum::extract::State;
use futures::{future::join_all, Stream, StreamExt};
use llm_json::repair_json;
use orama_js_pool::{ExecOptions, OutputChannel};
use regex::Regex;
use serde::{self, Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use super::constraint_extractor::{
    extract_constraints, format_constraints_for_prompt, has_shared_budget, inject_constraints,
    validate_search_params, BudgetAllocation, BudgetPlannerResponse, ExtractedConstraint,
    FieldInfo,
};
use super::llms::{KnownPrompts, LLMService};
use crate::ai::run_hooks::run_before_retrieval;
use crate::collection_manager::sides::read::{SearchAnalyticEventOrigin, SearchRequest};
use crate::{
    collection_manager::sides::read::{CollectionStats, ReadSide},
    types::{
        CollectionId, IndexId, InteractionLLMConfig, InteractionMessage, ReadApiKey, SearchParams,
        SearchResult,
    },
};

// ==== Macro Rules ====

// Macro to send a step result through the channel
macro_rules! send_step {
    ($tx:expr, $step:expr) => {
        $tx.send(Ok($step))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send step: {}", e))?
    };
}

// ===== Modified Data Models =====

/// Enhanced search result that includes the originating query information
#[derive(Serialize, Debug, Clone)]
pub struct QueryMappedSearchResult {
    pub original_query: String,
    pub generated_query: String,
    pub search_params: SearchParams,
    pub results: Vec<SearchResult>,
    pub query_index: usize,
}

impl QueryMappedSearchResult {
    /// Extract just the search results for legacy compatibility
    pub fn into_search_results(self) -> Vec<SearchResult> {
        self.results
    }

    /// Get a reference to the search results
    pub fn search_results(&self) -> &[SearchResult] {
        &self.results
    }

    /// Get the number of results
    pub fn result_count(&self) -> usize {
        self.results.len()
    }
}

/// Statistics for a single field in an index
#[derive(Serialize, Debug)]
struct FieldStats {
    field_path: String,
    field_type: String,
}

/// Statistics for a single index
#[derive(Serialize)]
struct IndexStats {
    pub id: String,
    fields_stats: Vec<FieldStats>,
}

/// Input structure for property selection
#[derive(Serialize)]
struct PropertiesSelectorInput {
    pub indexes_stats: Vec<IndexStats>,
}

/// A selected property with its metadata
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SelectedProperty {
    pub property: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub get: bool,
}

/// Query with its selected properties
#[derive(Serialize, Deserialize, Debug)]
pub struct SelectedProperties {
    pub query: String,
    pub properties: Vec<SelectedProperty>,
}

/// Properties for a specific collection
#[derive(Serialize, Debug, Clone)]
pub struct CollectionSelectedProperties {
    pub selected_properties: Vec<SelectedProperty>,
}

/// Combined query and properties information
#[derive(Serialize, Debug, Clone)]
pub struct QueryAndProperties {
    pub query: String,
    pub properties: HashMap<String, CollectionSelectedProperties>,
    pub filter_properties: HashMap<String, Vec<String>>,
}

/// Enhanced structure that tracks the query generation pipeline
#[derive(Serialize, Debug, Clone)]
pub struct TrackedQuery {
    pub index: usize,
    pub original_query: String,
    pub generated_query_text: String,
    pub search_params: SearchParams,
}

// ===== LLM Response Types =====

#[derive(Deserialize, Debug)]
struct LLMPropertiesResponse {
    #[serde(flatten)]
    collections: HashMap<String, CollectionPropertiesWrapper>,
}

#[derive(Deserialize, Debug)]
struct CollectionPropertiesWrapper {
    selected_properties: Vec<SelectedProperty>,
}

// ===== Statistics Types =====

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}

/// Unified representation of field statistics
#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum FieldStatType {
    #[serde(rename = "uncommitted_bool")]
    UncommittedBoolean {
        false_count: usize,
        true_count: usize,
    },
    #[serde(rename = "committed_bool")]
    CommittedBoolean {
        false_count: usize,
        true_count: usize,
    },
    #[serde(rename = "uncommitted_number")]
    UncommittedNumber {
        #[serde(default)]
        _min: Option<f64>,
        #[serde(default)]
        _max: Option<f64>,
        #[serde(default)]
        count: usize,
    },
    #[serde(rename = "committed_number")]
    CommittedNumber {
        #[serde(default)]
        _min: Option<f64>,
        #[serde(default)]
        _max: Option<f64>,
        #[serde(default)]
        document_count: usize,
    },
    #[serde(rename = "uncommitted_string_filter")]
    UncommittedStringFilter {
        #[allow(dead_code)]
        key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },
    #[serde(rename = "committed_string_filter")]
    CommittedStringFilter {
        #[allow(dead_code)]
        key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },
    #[serde(rename = "uncommitted_string")]
    UncommittedString {
        #[allow(dead_code)]
        key_count: usize,
        global_info: GlobalInfo,
    },
    #[serde(rename = "committed_string")]
    CommittedString {
        #[allow(dead_code)]
        key_count: usize,
        global_info: GlobalInfo,
    },
    #[serde(rename = "uncommitted_vector")]
    UncommittedVector {
        document_count: usize,
        #[allow(dead_code)]
        vector_count: usize,
    },
    #[serde(rename = "committed_vector")]
    CommittedVector {
        #[allow(dead_code)]
        dimensions: usize,
        #[allow(dead_code)]
        vector_count: usize,
    },
}

impl FieldStatType {
    /// Returns the document count for this field statistic
    fn document_count(&self) -> usize {
        use FieldStatType::*;
        match self {
            UncommittedBoolean {
                false_count,
                true_count,
            }
            | CommittedBoolean {
                false_count,
                true_count,
            } => false_count + true_count,
            UncommittedNumber { count, .. } => *count,
            CommittedNumber { document_count, .. } => *document_count,
            UncommittedStringFilter { document_count, .. }
            | CommittedStringFilter { document_count, .. } => *document_count,
            UncommittedString { global_info, .. } | CommittedString { global_info, .. } => {
                global_info.total_documents
            }
            UncommittedVector { document_count, .. } => *document_count,
            CommittedVector { .. } => 1,
        }
    }

    /// Returns the field type as a string
    fn field_type_name(&self) -> &'static str {
        use FieldStatType::*;
        match self {
            UncommittedBoolean { .. } | CommittedBoolean { .. } => "boolean",
            UncommittedNumber { .. } | CommittedNumber { .. } => "number",
            UncommittedStringFilter { .. } | CommittedStringFilter { .. } => "string_filter",
            UncommittedString { .. } | CommittedString { .. } => "string",
            UncommittedVector { .. } | CommittedVector { .. } => "vector",
        }
    }

    /// Checks if this is an unfilterable string field
    fn is_unfilterable_string(&self) -> bool {
        matches!(
            self,
            FieldStatType::UncommittedString { .. } | FieldStatType::CommittedString { .. }
        )
    }

    /// Extracts keys if this is a string filter field
    fn extract_keys(&self) -> Option<&[String]> {
        match self {
            FieldStatType::CommittedStringFilter { keys: Some(k), .. }
            | FieldStatType::UncommittedStringFilter { keys: Some(k), .. } => Some(k),
            _ => None,
        }
    }
}

// ===== Query Processing Steps =====

#[derive(Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AdvancedAutoQuerySteps {
    Init,
    OptimizingQuery,
    QueryOptimized(Vec<String>),
    SelectingProps,
    SelectedProps(Vec<HashMap<String, CollectionSelectedProperties>>),
    CombiningQueriesAndProperties,
    CombinedQueriesAndProperties(Vec<QueryAndProperties>),
    GeneratingQueries,
    GeneratedQueries(Vec<TrackedQuery>), // Modified to include tracked queries
    Searching,
    SearchResults(Vec<QueryMappedSearchResult>), // Modified to include query mapping
}

pub struct AdvancedAutoQueryStepResult {
    pub step: AdvancedAutoQuerySteps,
    pub result: String,
}

// ===== Main Query Processor =====

/// Handles advanced automatic query generation and processing
pub struct AdvancedAutoQuery {
    llm_service: Arc<LLMService>,
    llm_config: Option<InteractionLLMConfig>,
    collection_stats: CollectionStats,
}

impl AdvancedAutoQuery {
    /// Creates a new AdvancedAutoQuery instance
    pub fn new(
        collection_stats: CollectionStats,
        llm_service: Arc<LLMService>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Self {
        Self {
            llm_service,
            collection_stats,
            llm_config,
        }
    }

    /// Runs the complete auto-query pipeline and returns just the search results (legacy compatibility)
    pub async fn run_legacy(
        self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: &ReadApiKey,
        collection_id: CollectionId,
        conversation: Vec<InteractionMessage>,
        invocation_origin: SearchAnalyticEventOrigin,
        user_id: Option<String>,
    ) -> Result<Vec<SearchResult>> {
        let mapped_results = self
            .run(
                read_side,
                read_api_key,
                collection_id,
                conversation,
                None,
                invocation_origin,
                user_id,
            )
            .await?;

        // Flatten all results into a single Vec<SearchResult>
        Ok(mapped_results
            .into_iter()
            .flat_map(|mapped_result| mapped_result.results)
            .collect())
    }

    /// Runs the complete auto-query pipeline
    #[allow(clippy::too_many_arguments)]
    pub async fn run(
        self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: &ReadApiKey,
        collection_id: CollectionId,
        conversation: Vec<InteractionMessage>,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
        invocation_origin: SearchAnalyticEventOrigin,
        user_id: Option<String>,
    ) -> Result<Vec<QueryMappedSearchResult>> {
        let mut stream = self
            .run_stream(
                read_side.clone(),
                read_api_key,
                collection_id,
                conversation,
                log_sender,
                invocation_origin,
                user_id,
            )
            .await;

        let mut query_mapped_results = Vec::new();
        while let Some(step_result) = stream.next().await {
            match step_result {
                // We only return the last step, which contains the mapped results
                Ok(AdvancedAutoQuerySteps::SearchResults(results)) => {
                    query_mapped_results.extend(results);
                    break;
                }
                Ok(_step) => {}
                Err(e) => {
                    eprintln!("Error during auto-query processing: {e}");
                    return Err(e);
                }
            }
        }

        Ok(query_mapped_results)
    }

    /// Runs the auto-query pipeline as a stream of steps. Useful for real-time feedback in UIs.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_stream(
        mut self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: &ReadApiKey,
        collection_id: CollectionId,
        conversation: Vec<InteractionMessage>,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
        invocation_origin: SearchAnalyticEventOrigin,
        user_id: Option<String>,
    ) -> impl Stream<Item = Result<AdvancedAutoQuerySteps>> {
        let (tx, rx) = mpsc::channel(100);

        let read_api_key = read_api_key.clone();
        tokio::spawn(async move {
            let result: Result<()> = async {
                // Step 0: Initialize
                send_step!(tx, AdvancedAutoQuerySteps::Init);

                // Step 1: Convert conversation to JSON
                let conversation_json = serde_json::to_string(&conversation)?;

                // Step 2: Analyze input
                send_step!(tx, AdvancedAutoQuerySteps::OptimizingQuery);
                let optimized_queries = self.analyze_input(conversation_json).await?;
                send_step!(
                    tx,
                    AdvancedAutoQuerySteps::QueryOptimized(optimized_queries.clone())
                );

                // Step 3: Select properties
                send_step!(tx, AdvancedAutoQuerySteps::SelectingProps);
                let selected_properties = self.select_properties(optimized_queries.clone()).await?;
                send_step!(
                    tx,
                    AdvancedAutoQuerySteps::SelectedProps(selected_properties.clone())
                );

                // Step 4: Combine
                send_step!(tx, AdvancedAutoQuerySteps::CombiningQueriesAndProperties);
                let queries_and_properties =
                    self.combine_queries_and_properties(optimized_queries, selected_properties);
                send_step!(
                    tx,
                    AdvancedAutoQuerySteps::CombinedQueriesAndProperties(
                        queries_and_properties.clone()
                    )
                );

                // Step 5: Generate queries with tracking
                send_step!(tx, AdvancedAutoQuerySteps::GeneratingQueries);
                let tracked_queries = self
                    .generate_tracked_search_queries(queries_and_properties, user_id)
                    .await?;
                send_step!(
                    tx,
                    AdvancedAutoQuerySteps::GeneratedQueries(tracked_queries.clone())
                );

                // Step 6: Execute search with mapping
                send_step!(tx, AdvancedAutoQuerySteps::Searching);
                let mapped_results = self
                    .execute_mapped_searches(
                        read_side,
                        &read_api_key,
                        collection_id,
                        tracked_queries,
                        log_sender,
                        invocation_origin,
                    )
                    .await?;

                send_step!(tx, AdvancedAutoQuerySteps::SearchResults(mapped_results));

                Ok(())
            }
            .await;

            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            }
        });

        ReceiverStream::new(rx)
    }

    /// Analyzes the input conversation to extract optimized queries
    async fn analyze_input(&self, conversation_json: String) -> Result<Vec<String>> {
        let current_datetime = chrono::Utc::now();
        let variables = vec![
            ("conversation".to_string(), conversation_json),
            (
                "timestamp".to_string(),
                current_datetime.format("%Y-%m-%d %H:%M:%S").to_string(),
            ),
            (
                "timestamp_unix".to_string(),
                current_datetime.timestamp().to_string(),
            ),
        ];

        let result = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::AdvancedAutoqueryQueryAnalyzer,
                vec![],
                variables,
                None,
                self.llm_config.clone(),
            )
            .await
            .context("Failed to run query analyzer prompt")?;

        let cleaned = repair_json(&result, &Default::default())?;
        serde_json::from_str::<Vec<String>>(&cleaned)
            .context("Failed to parse query analyzer response")
    }

    /// Selects properties for the given queries
    async fn select_properties(
        &mut self,
        queries: Vec<String>,
    ) -> Result<Vec<HashMap<String, CollectionSelectedProperties>>> {
        let selectable_props = serde_json::to_string(&self.format_collection_stats())
            .context("Failed to serialize collection stats")?;

        let futures: Vec<_> = queries
            .iter()
            .map(|query| {
                let variables = vec![
                    ("query".to_string(), query.clone()),
                    ("properties_list".to_string(), selectable_props.clone()),
                ];
                self.llm_service.run_known_prompt(
                    KnownPrompts::AdvancedAutoQueryPropertiesSelector,
                    vec![],
                    variables,
                    None,
                    self.llm_config.clone(),
                )
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut parsed_results = Vec::new();

        for (index, result) in results.into_iter().enumerate() {
            match result {
                Ok(response) => {
                    let cleaned = repair_json(&response, &Default::default())?;
                    let parsed = parse_properties_response(&cleaned).unwrap_or_else(|e| {
                        eprintln!("Failed to parse LLM response at index {index}: {e}");
                        HashMap::new()
                    });
                    parsed_results.push(parsed);
                }
                Err(e) => {
                    eprintln!("LLM call failed at index {index}: {e}");
                    parsed_results.push(HashMap::new());
                }
            }
        }

        Ok(parsed_results)
    }

    /// Combines queries with their selected properties
    fn combine_queries_and_properties(
        &self,
        queries: Vec<String>,
        properties: Vec<HashMap<String, CollectionSelectedProperties>>,
    ) -> Vec<QueryAndProperties> {
        queries
            .into_iter()
            .zip(properties)
            .map(|(query, properties)| QueryAndProperties {
                query,
                properties,
                filter_properties: HashMap::new(),
            })
            .collect()
    }

    /// Generates search queries with tracking information.
    ///
    /// Integrates schema-aware constraint extraction: before calling the LLM,
    /// extracts numeric, string enum, and boolean constraints from the query text
    /// and passes them as hard constraints to the prompt. After the LLM responds,
    /// validates that all constraints are present and injects any that are missing.
    async fn generate_tracked_search_queries(
        &mut self,
        mut query_plan: Vec<QueryAndProperties>,
        user_id: Option<String>,
    ) -> Result<Vec<TrackedQuery>> {
        // Extract filter properties for all queries
        let filter_properties = self.extract_filter_properties(&query_plan)?;

        // Update query plan with filter properties
        for query_and_props in &mut query_plan {
            query_and_props.filter_properties = filter_properties.clone();
        }

        // Extract schema field info and number fields for constraint matching
        let schema_fields = self.extract_schema_fields();
        let number_fields = self.extract_number_fields();

        // Extract constraints for each query and build hard_constraints strings
        let per_query_constraints: Vec<Vec<ExtractedConstraint>> = query_plan
            .iter()
            .map(|qp| extract_constraints(&qp.query, &schema_fields, &filter_properties))
            .collect();

        // Convert to LLM variables (includes hard_constraints)
        let variables_list =
            self.create_llm_variables(&query_plan, &per_query_constraints, &number_fields);
        let futures: Vec<_> = variables_list
            .into_iter()
            .map(|variables| {
                self.llm_service.run_known_prompt(
                    KnownPrompts::AdvancedAutoQueryQueryComposer,
                    vec![],
                    variables,
                    None,
                    self.llm_config.clone(),
                )
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        // Process results with tracking and constraint validation
        let tracked_queries: Result<Vec<TrackedQuery>> = results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result
                    .with_context(|| format!("LLM call failed for query {index}"))
                    .and_then(|response| {
                        let cleaned = repair_json(&response, &Default::default())
                            .context("Failed to clean LLM response")?;
                        let mut search_params = serde_json::from_str::<SearchParams>(&cleaned)
                            .context("Failed to parse search params")?;
                        search_params.user_id = user_id.clone();

                        // Validate and inject missing constraints
                        let constraints = &per_query_constraints[index];
                        let missing =
                            validate_search_params(&search_params, constraints, &number_fields);
                        if !missing.is_empty() {
                            warn!(
                                "LLM omitted {} constraint(s) for query '{}', injecting",
                                missing.len(),
                                query_plan[index].query
                            );
                            inject_constraints(&mut search_params, &missing, &number_fields);
                        }

                        Ok(TrackedQuery {
                            index,
                            original_query: query_plan[index].query.clone(),
                            generated_query_text: cleaned,
                            search_params,
                        })
                    })
            })
            .collect();

        tracked_queries
    }

    /// Executes searches and maps results back to their originating queries
    async fn execute_mapped_searches(
        &self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: &ReadApiKey,
        collection_id: CollectionId,
        tracked_queries: Vec<TrackedQuery>,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
        invocation_origin: SearchAnalyticEventOrigin,
    ) -> Result<Vec<QueryMappedSearchResult>> {
        let search_futures = tracked_queries.iter().map(|tracked_query| {
            let read_side = read_side.clone();
            let tracked_query = tracked_query.clone();
            let log_sender = log_sender.clone();

            async move {
                let search_params = tracked_query.search_params.clone();
                let collection = read_side
                    .get_collection(collection_id, read_api_key)
                    .await?;
                let js_pool = collection.get_js_pool();
                let search_params = run_before_retrieval(
                    js_pool,
                    search_params.clone(),
                    log_sender,
                    ExecOptions::new(),
                )
                .await?;

                let search_result = read_side
                    .search(
                        read_api_key,
                        collection_id,
                        SearchRequest {
                            search_params,
                            search_analytics_event_origin: Some(invocation_origin),
                            analytics_metadata: None,
                            interaction_id: None,
                        },
                    )
                    .await
                    .context("Failed to execute search")?;

                Ok::<QueryMappedSearchResult, anyhow::Error>(QueryMappedSearchResult {
                    original_query: tracked_query.original_query,
                    generated_query: tracked_query.generated_query_text,
                    search_params: tracked_query.search_params,
                    results: vec![search_result],
                    query_index: tracked_query.index,
                })
            }
        });

        let results = join_all(search_futures).await;
        let mapped_results: Vec<QueryMappedSearchResult> = results
            .into_iter()
            .filter_map(|result| match result {
                Ok(mapped_result) => Some(mapped_result),
                Err(e) => {
                    eprintln!("Search execution failed: {e}");
                    None
                }
            })
            .collect();

        Ok(mapped_results)
    }

    /// Extracts filter properties from the query plan
    fn extract_filter_properties(
        &self,
        query_plan: &[QueryAndProperties],
    ) -> Result<HashMap<String, Vec<String>>> {
        let properties_by_collection = self.get_properties_to_retrieve(query_plan);
        let mut all_filter_properties = HashMap::new();

        for (collection_name, properties) in &properties_by_collection {
            let index_id = IndexId::try_new(collection_name).context("Invalid collection name")?;

            let index_stats = self
                .collection_stats
                .indexes_stats
                .iter()
                .find(|index| index.id == index_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("Collection {collection_name} not found in stats")
                })?;

            // Extract keys for each property
            for property in properties {
                let keys = self.extract_property_keys(index_stats, property)?;
                if !keys.is_empty() {
                    all_filter_properties.insert(property.clone(), keys);
                }
            }
        }

        Ok(all_filter_properties)
    }

    /// Extracts keys for a specific property from index statistics
    fn extract_property_keys(
        &self,
        index_stats: &crate::collection_manager::sides::read::IndexStats,
        property: &str,
    ) -> Result<Vec<String>> {
        let mut keys = Vec::new();

        for field_stat in &index_stats.fields_stats {
            if field_stat.field_path == property {
                let stat_json = serde_json::to_string(&field_stat.stats)
                    .context("Failed to serialize field stats")?;

                if let Ok(stat_type) = serde_json::from_str::<FieldStatType>(&stat_json) {
                    if let Some(field_keys) = stat_type.extract_keys() {
                        keys.extend_from_slice(field_keys);
                    }
                }
            }
        }

        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    /// Gets properties to retrieve from the query plan
    fn get_properties_to_retrieve(
        &self,
        query_plan: &[QueryAndProperties],
    ) -> HashMap<String, Vec<String>> {
        let mut properties_map: HashMap<String, Vec<String>> = HashMap::new();

        for query_and_props in query_plan {
            for (collection_name, collection_props) in &query_and_props.properties {
                let props_to_get: Vec<String> = collection_props
                    .selected_properties
                    .iter()
                    .filter(|prop| prop.get)
                    .map(|prop| prop.property.clone())
                    .collect();

                properties_map
                    .entry(collection_name.clone())
                    .or_default()
                    .extend(props_to_get);
            }
        }

        // Remove duplicates and sort
        for properties in properties_map.values_mut() {
            properties.sort();
            properties.dedup();
        }

        properties_map
    }

    /// Creates LLM variables from queries, properties, and extracted constraints.
    ///
    /// Includes a `hard_constraints` variable that lists pre-extracted constraints
    /// for the LLM to include in its generated search parameters.
    fn create_llm_variables(
        &self,
        queries_and_properties: &[QueryAndProperties],
        per_query_constraints: &[Vec<ExtractedConstraint>],
        number_fields: &[String],
    ) -> Vec<Vec<(String, String)>> {
        queries_and_properties
            .iter()
            .enumerate()
            .map(|(i, qp)| {
                let mut variables = vec![("query".to_string(), qp.query.clone())];

                // Flatten all properties from all collections
                let all_properties: Vec<_> = qp
                    .properties
                    .values()
                    .flat_map(|cp| cp.selected_properties.clone())
                    .collect();

                if let Ok(props_json) = serde_json::to_string(&all_properties) {
                    variables.push(("properties_list".to_string(), props_json));
                }

                if let Ok(filter_json) = serde_json::to_string(&qp.filter_properties) {
                    variables.push(("filter_properties".to_string(), filter_json));
                }

                // Add hard constraints from the constraint extractor
                let constraints = per_query_constraints
                    .get(i)
                    .map(|c| c.as_slice())
                    .unwrap_or(&[]);
                let hard_constraints = format_constraints_for_prompt(constraints, number_fields);
                variables.push(("hard_constraints".to_string(), hard_constraints));

                variables
            })
            .collect()
    }

    /// Formats collection statistics for property selection
    fn format_collection_stats(&mut self) -> PropertiesSelectorInput {
        let prefix_regex = Regex::new(r"^(committed|uncommitted)_").expect("Valid regex pattern");

        let mut indexes_stats = Vec::new();
        let mut seen_fields = HashMap::new();

        for index in &self.collection_stats.indexes_stats {
            // Skip temporary indexes
            if index.is_temp {
                continue;
            }

            let fields_stats = self.extract_valid_fields(index, &prefix_regex, &mut seen_fields);

            if !fields_stats.is_empty() {
                indexes_stats.push(IndexStats {
                    id: index.id.as_str().to_string(),
                    fields_stats,
                });
            }
        }

        PropertiesSelectorInput { indexes_stats }
    }

    /// Extracts valid fields from index statistics
    fn extract_valid_fields(
        &self,
        index: &crate::collection_manager::sides::read::IndexStats,
        prefix_regex: &Regex,
        seen_fields: &mut HashMap<String, bool>,
    ) -> Vec<FieldStats> {
        let result: Vec<FieldStats> = index
            .fields_stats
            .iter()
            .filter_map(|stat| {
                let field_path = prefix_regex.replace(&stat.field_path, "").to_string();

                // Skip auto-generated and already seen fields
                if field_path == "___orama_auto_embedding" {
                    return None;
                }

                if seen_fields.contains_key(&field_path) {
                    return None;
                }

                // Try to deserialize and validate the field
                let stat_json = match serde_json::to_string(&stat.stats) {
                    Ok(json) => json,
                    Err(_e) => {
                        return None;
                    }
                };

                let field_stat_type = match serde_json::from_str::<FieldStatType>(&stat_json) {
                    Ok(stat_type) => stat_type,
                    Err(_e) => {
                        return None;
                    }
                };

                // Skip unfilterable strings and fields with no documents
                if field_stat_type.is_unfilterable_string() {
                    return None;
                }

                if field_stat_type.document_count() == 0 {
                    return None;
                }

                seen_fields.insert(field_path.clone(), true);

                Some(FieldStats {
                    field_path,
                    field_type: field_stat_type.field_type_name().to_string(),
                })
            })
            .collect();

        result
    }

    /// Extract schema field info (name + type) from collection stats.
    ///
    /// Used to provide schema context to the constraint extractor.
    fn extract_schema_fields(&self) -> Vec<FieldInfo> {
        let prefix_regex = Regex::new(r"^(committed|uncommitted)_").expect("Valid regex pattern");
        let mut fields = Vec::new();
        let mut seen = HashMap::new();

        for index in &self.collection_stats.indexes_stats {
            if index.is_temp {
                continue;
            }
            for stat in &index.fields_stats {
                let field_path = prefix_regex.replace(&stat.field_path, "").to_string();
                if field_path == "___orama_auto_embedding" || seen.contains_key(&field_path) {
                    continue;
                }
                if let Ok(stat_json) = serde_json::to_string(&stat.stats) {
                    if let Ok(field_stat_type) = serde_json::from_str::<FieldStatType>(&stat_json) {
                        if !field_stat_type.is_unfilterable_string()
                            && field_stat_type.document_count() > 0
                        {
                            seen.insert(field_path.clone(), true);
                            fields.push(FieldInfo {
                                name: field_path,
                                field_type: field_stat_type.field_type_name().to_string(),
                            });
                        }
                    }
                }
            }
        }

        fields
    }

    /// Extract the names of all number-typed fields from collection stats.
    fn extract_number_fields(&self) -> Vec<String> {
        self.extract_schema_fields()
            .iter()
            .filter(|f| f.field_type == "number")
            .map(|f| f.name.clone())
            .collect()
    }

    /// Plan budget allocations across multiple sub-queries when a shared budget is detected.
    ///
    /// Calls the budget planner LLM prompt to split the total budget across sub-queries.
    /// Returns None if no shared budget is detected or if planning fails.
    #[allow(dead_code)]
    async fn plan_budget(
        &self,
        original_query: &str,
        sub_queries: &[String],
        global_constraints: &[ExtractedConstraint],
    ) -> Result<Option<Vec<BudgetAllocation>>> {
        // Only plan budget if there's a shared budget and multiple sub-queries
        if sub_queries.len() <= 1 || !has_shared_budget(original_query) {
            return Ok(None);
        }

        // Find the budget constraint
        let budget_constraint = global_constraints.iter().find(|c| {
            matches!(
                c,
                ExtractedConstraint::Numeric {
                    field_hint: Some(hint),
                    ..
                } if hint == "price"
            )
        });

        let budget_value = match budget_constraint {
            Some(ExtractedConstraint::Numeric { value, .. }) => *value,
            _ => return Ok(None),
        };

        let number_fields = self.extract_number_fields();
        let price_fields_json =
            serde_json::to_string(&number_fields).context("Failed to serialize price fields")?;

        let variables = vec![
            ("original_query".to_string(), original_query.to_string()),
            (
                "sub_queries".to_string(),
                serde_json::to_string(sub_queries).context("Failed to serialize sub-queries")?,
            ),
            ("total_budget".to_string(), budget_value.to_string()),
            ("price_fields".to_string(), price_fields_json),
        ];

        let result = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::V1_1AdvancedAutoQueryBudgetPlanner,
                vec![],
                variables,
                None,
                self.llm_config.clone(),
            )
            .await
            .context("Failed to run budget planner prompt")?;

        let cleaned = repair_json(&result, &Default::default())
            .context("Failed to clean budget planner response")?;

        match serde_json::from_str::<BudgetPlannerResponse>(&cleaned) {
            Ok(response) => Ok(Some(response.allocations)),
            Err(e) => {
                warn!("Failed to parse budget planner response: {e}");
                Ok(None)
            }
        }
    }
}

// ===== Helper Functions =====

/// Flattens QueryMappedSearchResults into a simple Vec<SearchResult> for legacy compatibility
pub fn flatten_mapped_results(mapped_results: Vec<QueryMappedSearchResult>) -> Vec<SearchResult> {
    mapped_results
        .into_iter()
        .flat_map(|mapped_result| mapped_result.results)
        .collect()
}

/// Parses the LLM properties response into the expected format
fn parse_properties_response(
    response: &str,
) -> Result<HashMap<String, CollectionSelectedProperties>> {
    let llm_response: LLMPropertiesResponse =
        serde_json::from_str(response).context("Failed to parse LLM properties response")?;

    Ok(llm_response
        .collections
        .into_iter()
        .map(|(name, props)| {
            (
                name,
                CollectionSelectedProperties {
                    selected_properties: props.selected_properties,
                },
            )
        })
        .collect())
}
