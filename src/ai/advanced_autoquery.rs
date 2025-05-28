use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::llms::{KnownPrompts, LLMService};
use crate::{
    collection_manager::sides::CollectionStats,
    types::{CollectionId, IndexId, InteractionLLMConfig, InteractionMessage, SearchParams},
};

// ===== Data Models =====

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
    UncommittedNumber { _min: f64, _max: f64, count: usize },
    #[serde(rename = "committed_number")]
    CommittedNumber { _min: f64, _max: f64 },
    #[serde(rename = "uncommitted_string_filter")]
    UncommittedStringFilter {
        _key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },
    #[serde(rename = "committed_string_filter")]
    CommittedStringFilter {
        _key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },
    #[serde(rename = "uncommitted_string")]
    UncommittedString {
        _key_count: usize,
        global_info: GlobalInfo,
    },
    #[serde(rename = "committed_string")]
    CommittedString {
        _key_count: usize,
        global_info: GlobalInfo,
    },
    #[serde(rename = "uncommitted_vector")]
    UncommittedVector {
        document_count: usize,
        _vector_count: usize,
    },
    #[serde(rename = "committed_vector")]
    CommittedVector {
        _dimensions: usize,
        _vector_count: usize,
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
            CommittedNumber { .. } => 0,
            UncommittedStringFilter { document_count, .. }
            | CommittedStringFilter { document_count, .. } => *document_count,
            UncommittedString { global_info, .. } | CommittedString { global_info, .. } => {
                global_info.total_documents
            }
            UncommittedVector { document_count, .. } => *document_count,
            CommittedVector { .. } => 0,
        }
    }

    /// Returns the field type as a string
    fn field_type_name(&self) -> &'static str {
        use FieldStatType::*;
        match self {
            UncommittedBoolean { .. } | CommittedBoolean { .. } => "boolean",
            UncommittedNumber { .. } | CommittedNumber { .. } => "number",
            UncommittedStringFilter { .. } | CommittedStringFilter { .. } => "string",
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

pub enum AdvancedAutoQuerySteps {
    Init,
    AnalyzeInput,
    SelectProps,
    GetPropValues,
    GenerateQueries,
    RunQueries,
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
    step_results: Vec<AdvancedAutoQueryStepResult>,
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
            step_results: Vec::new(),
            llm_config,
        }
    }

    /// Runs the complete auto-query pipeline
    pub async fn run(
        &mut self,
        conversation: Vec<InteractionMessage>,
    ) -> Result<Vec<SearchParams>> {
        // Step 1: Analyze the conversation input
        let conversation_json =
            serde_json::to_string(&conversation).context("Failed to serialize conversation")?;
        let optimized_queries = self.analyze_input(conversation_json).await?;

        // Step 2: Select properties for each query
        let selected_properties = self.select_properties(optimized_queries.clone()).await?;

        // Step 3: Combine queries with properties
        let queries_and_properties =
            self.combine_queries_and_properties(optimized_queries, selected_properties);

        // Step 4: Generate search queries
        self.generate_search_queries(queries_and_properties).await
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
                variables,
                self.llm_config.clone(),
            )
            .await
            .context("Failed to run query analyzer prompt")?;

        let cleaned = clean_json_response(&result);
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
                    variables,
                    self.llm_config.clone(),
                )
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        Ok(results
            .into_iter()
            .enumerate()
            .map(|(index, result)| match result {
                Ok(response) => {
                    let cleaned = clean_json_response(&response);
                    parse_properties_response(&cleaned).unwrap_or_else(|e| {
                        eprintln!("Failed to parse LLM response at index {}: {}", index, e);
                        HashMap::new()
                    })
                }
                Err(e) => {
                    eprintln!("LLM call failed at index {}: {}", index, e);
                    HashMap::new()
                }
            })
            .collect::<Vec<_>>())
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

    /// Generates search queries from the query plan
    async fn generate_search_queries(
        &mut self,
        mut query_plan: Vec<QueryAndProperties>,
    ) -> Result<Vec<SearchParams>> {
        // Extract filter properties for all queries
        let filter_properties = self.extract_filter_properties(&query_plan)?;

        // Update query plan with filter properties
        for query_and_props in &mut query_plan {
            query_and_props.filter_properties = filter_properties.clone();
        }

        // Convert to LLM variables and run queries in parallel
        let variables_list = self.create_llm_variables(&query_plan);
        let futures: Vec<_> = variables_list
            .into_iter()
            .map(|variables| {
                self.llm_service.run_known_prompt(
                    KnownPrompts::AdvancedAutoQueryQueryComposer,
                    variables,
                    self.llm_config.clone(),
                )
            })
            .collect();

        let results = futures::future::join_all(futures).await;

        // Process results
        results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result
                    .with_context(|| format!("LLM call failed for query {}", index))
                    .and_then(|response| {
                        let cleaned = clean_json_response(&response);
                        serde_json::from_str::<SearchParams>(&cleaned)
                            .context("Failed to parse search params")
                    })
            })
            .collect()
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
                    anyhow::anyhow!("Collection {} not found in stats", collection_name)
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
        index_stats: &crate::collection_manager::sides::IndexStats,
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

    /// Creates LLM variables from queries and properties
    fn create_llm_variables(
        &self,
        queries_and_properties: &[QueryAndProperties],
    ) -> Vec<Vec<(String, String)>> {
        queries_and_properties
            .iter()
            .map(|qp| {
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

        self.step_results.push(AdvancedAutoQueryStepResult {
            step: AdvancedAutoQuerySteps::SelectProps,
            result: serde_json::to_string(&indexes_stats).unwrap_or_default(),
        });

        PropertiesSelectorInput { indexes_stats }
    }

    /// Extracts valid fields from index statistics
    fn extract_valid_fields(
        &self,
        index: &crate::collection_manager::sides::IndexStats,
        prefix_regex: &Regex,
        seen_fields: &mut HashMap<String, bool>,
    ) -> Vec<FieldStats> {
        index
            .fields_stats
            .iter()
            .filter_map(|stat| {
                let field_path = prefix_regex.replace(&stat.field_path, "").to_string();

                // Skip auto-generated and already seen fields
                if field_path == "___orama_auto_embedding" || seen_fields.contains_key(&field_path)
                {
                    return None;
                }

                // Try to deserialize and validate the field
                let stat_json = serde_json::to_string(&stat.stats).ok()?;
                let field_stat_type = serde_json::from_str::<FieldStatType>(&stat_json).ok()?;

                // Skip unfilterable strings and fields with no documents
                if field_stat_type.is_unfilterable_string() || field_stat_type.document_count() == 0
                {
                    return None;
                }

                seen_fields.insert(field_path.clone(), true);

                Some(FieldStats {
                    field_path,
                    field_type: field_stat_type.field_type_name().to_string(),
                })
            })
            .collect()
    }
}

// ===== Helper Functions =====

/// Removes markdown code block wrappers from JSON responses
fn clean_json_response(input: &str) -> String {
    let trimmed = input.trim();

    if let Some(content) = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
    {
        content
            .trim_start_matches('\n')
            .trim_end_matches("```")
            .trim()
            .to_string()
    } else {
        trimmed.to_string()
    }
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
