use anyhow::Result;
use regex::Regex;
use serde::Serialize;
use serde::{self, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::llms::LLMService;
use crate::types::{IndexId, SearchParams};
use crate::{
    collection_manager::sides::CollectionStats,
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};

#[derive(Serialize, Debug)]
struct FieldStats {
    field_path: String,
    field_type: String,
}

#[derive(Serialize)]
struct IndexesStats {
    pub id: String,
    fields_stats: Vec<FieldStats>,
}

#[derive(Serialize)]
struct PropertiesSelectorInput {
    pub indexes_stats: Vec<IndexesStats>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SelectedProperty {
    pub property: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub get: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SelectedProperties {
    pub query: String,
    pub properties: Vec<SelectedProperty>,
}

#[derive(Deserialize, Debug)]
struct LLMPropertiesResponse {
    #[serde(flatten)]
    collections: HashMap<String, CollectionProperties>,
}

#[derive(Deserialize, Debug)]
struct CollectionProperties {
    selected_properties: Vec<SelectedProperty>,
}

#[derive(Serialize, Debug, Clone)]
pub struct CollectionSelectedProperties {
    pub selected_properties: Vec<SelectedProperty>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum DeserializedStat {
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
    UncommittedNumber { min: f64, max: f64, count: usize },

    #[serde(rename = "committed_number")]
    CommittedNumber { min: f64, max: f64 },

    #[serde(rename = "uncommitted_string_filter")]
    UncommittedStringFilter {
        key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },

    #[serde(rename = "committed_string_filter")]
    CommittedStringFilter {
        key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },

    #[serde(rename = "uncommitted_string")]
    UncommittedString {
        key_count: usize,
        global_info: GlobalInfo,
    },

    #[serde(rename = "committed_string")]
    CommittedString {
        key_count: usize,
        global_info: GlobalInfo,
    },

    #[serde(rename = "uncommitted_vector")]
    UncommittedVector {
        document_count: usize,
        vector_count: usize,
    },

    #[serde(rename = "committed_vector")]
    CommittedVector {
        dimensions: usize,
        vector_count: usize,
    },
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}

impl DeserializedStat {
    // Helper method to extract document count from any variant
    fn get_document_count(&self) -> usize {
        match self {
            DeserializedStat::UncommittedBoolean {
                false_count,
                true_count,
            } => false_count + true_count,
            DeserializedStat::CommittedBoolean {
                false_count,
                true_count,
            } => false_count + true_count,
            DeserializedStat::UncommittedNumber { count, .. } => *count,
            DeserializedStat::CommittedNumber { .. } => 0, // Committed numbers don't have count
            DeserializedStat::UncommittedStringFilter { document_count, .. } => *document_count,
            DeserializedStat::CommittedStringFilter { document_count, .. } => *document_count,
            DeserializedStat::UncommittedString { global_info, .. } => global_info.total_documents,
            DeserializedStat::CommittedString { global_info, .. } => global_info.total_documents,
            DeserializedStat::UncommittedVector { document_count, .. } => *document_count,
            DeserializedStat::CommittedVector { .. } => 0, // Vector count isn't document count
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct QueryAndProperties {
    pub query: String,
    pub properties: HashMap<String, CollectionSelectedProperties>,
    pub filter_properties: HashMap<String, Vec<String>>,
}

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

pub struct AdvancedAutoQuery {
    pub llm_service: Arc<LLMService>,
    pub llm_config: Option<InteractionLLMConfig>,
    pub collection_id: CollectionId,
    pub collection_stats: CollectionStats,
    pub step: AdvancedAutoQuerySteps,
    pub step_results: Vec<AdvancedAutoQueryStepResult>,
}

impl AdvancedAutoQuery {
    pub fn new(
        collection_id: CollectionId,
        collection_stats: CollectionStats,
        llm_service: Arc<LLMService>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Self {
        AdvancedAutoQuery {
            llm_service,
            collection_stats,
            collection_id,
            step: AdvancedAutoQuerySteps::Init,
            step_results: Vec::new(),
            llm_config,
        }
    }

    pub async fn run(
        &mut self,
        conversation: Vec<InteractionMessage>,
    ) -> Result<Vec<QueryAndProperties>> {
        let conversation_to_json = serde_json::to_string(&conversation)?;
        let optimized_queries = self.analyze_input(conversation_to_json).await?;
        let selected_properties = self.select_properties(optimized_queries.clone()).await?;

        let queries_and_properties: Vec<QueryAndProperties> = optimized_queries
            .into_iter()
            .zip(selected_properties)
            .map(|(query, properties)| QueryAndProperties {
                query,
                properties,
                filter_properties: HashMap::new(),
            })
            .collect();

        let search_queries = self
            .get_search_query(queries_and_properties.clone())
            .await?;

        dbg!(&search_queries);

        Ok(queries_and_properties)
    }

    pub async fn analyze_input(&self, conversation_as_json: String) -> Result<Vec<String>> {
        let current_datetime = chrono::Utc::now().timestamp();
        let formatted_datetime =
            chrono::DateTime::<chrono::Utc>::from_timestamp(current_datetime, 0)
                .unwrap()
                .format("%Y-%m-%d %H:%M:%S")
                .to_string();

        let variables = vec![
            ("conversation".to_string(), conversation_as_json),
            ("timestamp".to_string(), formatted_datetime),
            ("timestamp_unix".to_string(), current_datetime.to_string()),
        ];

        let result = self
            .llm_service
            .run_known_prompt(
                super::llms::KnownPrompts::AdvancedAutoqueryQueryAnalyzer,
                variables,
                self.llm_config.clone(),
            )
            .await?;

        let repaired = self.strip_markdown_wrapper(&result);

        Ok(serde_json::from_str::<Vec<String>>(&repaired)?)
    }

    pub async fn select_properties(
        &mut self,
        queries: Vec<String>,
    ) -> Result<Vec<HashMap<String, CollectionSelectedProperties>>> {
        let selectable_props = serde_json::to_string(&self.format_collection_stats())?;

        let futures = queries
            .iter()
            .map(|query| {
                let variables = vec![
                    ("query".to_string(), query.clone()),
                    ("properties_list".to_string(), selectable_props.clone()),
                ];
                self.llm_service.run_known_prompt(
                    super::llms::KnownPrompts::AdvancedAutoQueryPropertiesSelector,
                    variables,
                    self.llm_config.clone(),
                )
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(futures).await;

        let mut properties_list = Vec::new();

        for (index, result) in results.into_iter().enumerate() {
            let repaired = self.strip_markdown_wrapper(&result?);

            match serde_json::from_str::<LLMPropertiesResponse>(&repaired) {
                Ok(llm_response) => {
                    // Convert the LLM response to the expected output format
                    let mut collection_properties = HashMap::new();

                    for (collection_name, props) in llm_response.collections {
                        collection_properties.insert(
                            collection_name,
                            CollectionSelectedProperties {
                                selected_properties: props.selected_properties,
                            },
                        );
                    }

                    properties_list.push(collection_properties);
                }
                Err(e) => {
                    eprintln!("Failed to parse LLM response at index {}: {}", index, e);
                    eprintln!("Raw response: {}", repaired);

                    // Create empty result as fallback
                    properties_list.push(HashMap::new());
                }
            }
        }

        Ok(properties_list)
    }

    async fn get_search_query(
        &mut self,
        query_plan: Vec<QueryAndProperties>,
    ) -> Result<Vec<SearchParams>> {
        let properties_by_collection = self.get_properties_values_to_retrieve(query_plan.clone());

        // Collect all filter properties from all collections
        let mut all_filter_properties: HashMap<String, Vec<String>> = HashMap::new();

        for (collection_name, properties) in &properties_by_collection {
            let index_stats = self
                .collection_stats
                .indexes_stats
                .iter()
                .find(|index| index.id == IndexId::try_new(&collection_name).unwrap())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Collection {} not found in collection stats",
                        collection_name
                    )
                })?;

            let matching_fields = index_stats
                .fields_stats
                .iter()
                .filter(|field| properties.iter().any(|prop| *prop == field.field_path))
                .collect::<Vec<_>>();

            // Extract keys for each property
            for property in properties {
                let mut keys_for_property: Vec<String> = Vec::new();

                // Find all field stats for this specific property
                let property_fields: Vec<_> = matching_fields
                    .iter()
                    .filter(|field| field.field_path == *property)
                    .collect();

                for field_stat in property_fields {
                    // Serialize and deserialize the field stat to extract keys
                    let field_stat_json = serde_json::to_string(&field_stat.stats)?;

                    match serde_json::from_str::<DeserializedStat>(&field_stat_json) {
                        Ok(deserialized_stat) => {
                            match deserialized_stat {
                                DeserializedStat::CommittedStringFilter {
                                    keys: Some(keys),
                                    ..
                                } => {
                                    keys_for_property.extend(keys);
                                }
                                DeserializedStat::UncommittedStringFilter {
                                    keys: Some(keys),
                                    ..
                                } => {
                                    keys_for_property.extend(keys);
                                }
                                _ => {} // Other stat types don't have keys we care about
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to deserialize stat for {}: {}", property, e);
                        }
                    }
                }

                // Remove duplicates and sort
                keys_for_property.sort();
                keys_for_property.dedup();

                if !keys_for_property.is_empty() {
                    all_filter_properties.insert(property.clone(), keys_for_property);
                }
            }
        }

        // Update the existing query_plan with filter_properties
        let updated_queries_and_properties: Vec<QueryAndProperties> = query_plan
            .into_iter()
            .map(|mut query_and_props| {
                query_and_props.filter_properties = all_filter_properties.clone();
                query_and_props
            })
            .collect();

        let to_variables =
            self.query_and_properties_to_llm_variables(&updated_queries_and_properties);

        // Run LLM calls in parallel
        let futures = to_variables
            .into_iter()
            .map(|variables| {
                self.llm_service.run_known_prompt(
                    super::llms::KnownPrompts::AdvancedAutoQueryQueryComposer,
                    variables,
                    self.llm_config.clone(),
                )
            })
            .collect::<Vec<_>>();

        let results = futures::future::join_all(futures).await;

        // Process results and handle any errors
        let mut processed_results = Vec::new();

        for (index, result) in results.into_iter().enumerate() {
            match result {
                Ok(response) => {
                    let repaired = self.strip_markdown_wrapper(&response);
                    let to_search_params = serde_json::from_str::<SearchParams>(&repaired)?;
                    processed_results.push(to_search_params);
                    // Do something with the repaired result if needed
                }
                Err(e) => {
                    eprintln!("LLM call failed for query {}: {}", index, e);
                    // Handle error case - you might want to return an error or continue
                    return Err(e.into());
                }
            }
        }

        Ok(processed_results)
    }

    fn get_properties_values_to_retrieve(
        &mut self,
        query_plan: Vec<QueryAndProperties>,
    ) -> HashMap<String, Vec<String>> {
        let mut properties_to_retrieve: HashMap<String, Vec<String>> = HashMap::new();

        for query_and_properties in query_plan.iter() {
            for (collection_name, collection_props) in &query_and_properties.properties {
                let properties: Vec<String> = collection_props
                    .selected_properties
                    .iter()
                    .filter(|prop| prop.get)
                    .map(|prop| prop.property.clone())
                    .collect();

                properties_to_retrieve
                    .entry(collection_name.clone())
                    .or_insert_with(Vec::new)
                    .extend(properties);
            }
        }

        for (_, properties) in properties_to_retrieve.iter_mut() {
            properties.sort();
            properties.dedup();
        }

        properties_to_retrieve
    }

    fn query_and_properties_to_llm_variables(
        &self,
        queries_and_properties: &[QueryAndProperties],
    ) -> Vec<Vec<(String, String)>> {
        queries_and_properties
            .iter()
            .map(|query_and_props| {
                let mut variables = Vec::new();

                // Add the query
                variables.push(("query".to_string(), query_and_props.query.clone()));

                // Transform properties - flatten all collections into a single list
                let mut all_properties = Vec::new();
                for (_collection_name, collection_props) in &query_and_props.properties {
                    all_properties.extend(collection_props.selected_properties.clone());
                }

                // Serialize the properties list
                if let Ok(properties_json) = serde_json::to_string(&all_properties) {
                    variables.push(("properties_list".to_string(), properties_json));
                }

                // Serialize the filter properties
                if let Ok(filter_properties_json) =
                    serde_json::to_string(&query_and_props.filter_properties)
                {
                    variables.push(("filter_properties".to_string(), filter_properties_json));
                }

                variables
            })
            .collect()
    }

    fn format_collection_stats(&mut self) -> PropertiesSelectorInput {
        let prefix_re = Regex::new(r"^(committed|uncommitted)_").unwrap();

        let mut indexes_stats: Vec<IndexesStats> = Vec::new();
        let mut seen: HashMap<String, bool> = HashMap::new();

        for index in self.collection_stats.indexes_stats.iter() {
            let mut fields_stats: Vec<FieldStats> = Vec::new();
            if !index.is_temp {
                let index_id = index.id;

                for stat in index.fields_stats.iter() {
                    let mut field_path = stat.field_path.clone();
                    field_path = prefix_re.replace(&field_path, "").to_string();

                    // Skip the auto embedding index
                    if field_path == "___orama_auto_embedding" {
                        continue;
                    }

                    let field_stat = serde_json::to_string(&stat.stats).unwrap(); // @todo: handle error

                    match serde_json::from_str::<DeserializedStat>(&field_stat) {
                        Ok(deserialized_stat) => {
                            let total_docs = deserialized_stat.get_document_count();
                            let is_unfiltrable_string = matches!(
                                deserialized_stat,
                                DeserializedStat::UncommittedString { .. }
                                    | DeserializedStat::CommittedString { .. }
                            );

                            if is_unfiltrable_string {
                                // Skip unfiltrable strings
                                continue;
                            }

                            if total_docs > 0 {
                                if seen.contains_key(&field_path) {
                                    continue;
                                }

                                seen.insert(field_path.clone(), true);

                                fields_stats.push(FieldStats {
                                    field_path: field_path.clone(),
                                    field_type: match deserialized_stat {
                                        DeserializedStat::UncommittedBoolean { .. } => {
                                            "boolean".to_string()
                                        }
                                        DeserializedStat::CommittedBoolean { .. } => {
                                            "boolean".to_string()
                                        }
                                        DeserializedStat::UncommittedNumber { .. } => {
                                            "number".to_string()
                                        }
                                        DeserializedStat::CommittedNumber { .. } => {
                                            "number".to_string()
                                        }
                                        DeserializedStat::UncommittedStringFilter { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::CommittedStringFilter { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::UncommittedString { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::CommittedString { .. } => {
                                            "string".to_string()
                                        }
                                        DeserializedStat::UncommittedVector { .. } => {
                                            "vector".to_string()
                                        }
                                        DeserializedStat::CommittedVector { .. } => {
                                            "vector".to_string()
                                        }
                                    },
                                });
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to deserialize field stat for {}: {}", field_path, e);
                            continue;
                        }
                    }
                }

                if !fields_stats.is_empty() {
                    indexes_stats.push(IndexesStats {
                        id: index_id.as_str().to_string(),
                        fields_stats,
                    });
                }
            }
        }

        self.step_results.push(AdvancedAutoQueryStepResult {
            step: AdvancedAutoQuerySteps::SelectProps,
            result: serde_json::to_string(&indexes_stats).unwrap(),
        });

        PropertiesSelectorInput { indexes_stats }
    }

    fn strip_markdown_wrapper(&self, input: &str) -> String {
        let trimmed = input.trim();
        if trimmed.starts_with("```") {
            let lines: Vec<&str> = trimmed.lines().collect();

            if lines.len() >= 3 {
                let start_idx = if lines[0] == "```" || lines[0].starts_with("```json") {
                    1
                } else {
                    0
                };

                let end_idx = if lines.last() == Some(&"```") {
                    lines.len() - 1
                } else {
                    lines.len()
                };

                if start_idx < end_idx {
                    return lines[start_idx..end_idx].join("\n");
                }
            }
        }

        trimmed.to_string()
    }
}
