use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

use oramacore_lib::generic_kv::KV;
use std::{collections::HashMap, sync::Arc};
use strum_macros::Display;

use crate::collection_manager::sides::read::SearchRequest;
use crate::types::{
    InteractionLLMConfig, InteractionMessage, TrainingSetQueriesOptimizerQuerySet,
    TrainingSetsQueriesOptimizerResponse,
};
use crate::{
    ai::llms::LLMService,
    collection_manager::sides::read::ReadSide,
    types::{
        CollectionId, CollectionStatsRequest, FulltextMode, Limit, Properties, ReadApiKey,
        SearchMode, SearchOffset, SearchParams, WhereFilter,
    },
};

use super::llms::KnownPrompts;

#[derive(Debug, Serialize)]
pub enum QueryPlannerOptions {
    Answer,
    AdvancedAutoquery,
    Suggestions,
}

#[derive(Debug, Serialize)]
pub struct QueryPlannerTrainingData {
    pub training_set: Vec<(String, QueryPlannerOptions)>,
}

#[derive(Debug, Serialize)]
pub struct QueryOptimizerTrainingData {
    pub training_set: Vec<(String, Vec<String>)>,
}

#[derive(Debug, Serialize)]
pub enum TrainingData {
    QueryPlannerTrainingData(QueryPlannerTrainingData),
    QueryOptimizerTrainingData(QueryOptimizerTrainingData),
    // @todo: add query filtering
}

#[derive(Debug, Serialize, Display)]
pub enum TrainingDestination {
    QueryPlanner,
    QueryOptimizer,
    QueryFiltering,
}

impl TrainingDestination {
    pub fn try_from_str(s: &str) -> Option<Self> {
        match s {
            "query_planner" => Some(TrainingDestination::QueryPlanner),
            "query_optimizer" => Some(TrainingDestination::QueryOptimizer),
            "query_filtering" => Some(TrainingDestination::QueryFiltering),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TrainingSetsQueriesGeneratorResponse {
    pub simple: Vec<String>,
    pub multiple_terms: Vec<String>,
    pub advanced: Vec<String>,
}

#[derive(Clone)]
pub struct TrainingSetInterface {
    kv: Arc<KV>,
}

impl TrainingSetInterface {
    pub fn new(kv: Arc<KV>) -> Self {
        TrainingSetInterface { kv }
    }

    pub async fn insert_training_set(
        &self,
        collection_id: CollectionId,
        destination: TrainingDestination,
        training_data: String,
    ) -> anyhow::Result<()> {
        let key = self.get_training_set_kv_key(collection_id, destination);
        self.kv.insert(key, training_data).await
    }

    pub async fn list_training_sets(
        &self,
        collection_id: CollectionId,
        destination: TrainingDestination,
    ) -> anyhow::Result<Vec<String>> {
        let prefix = self.get_training_set_kv_key(collection_id, destination);
        self.kv.prefix_scan(&prefix).await
    }

    pub async fn get_training_set(
        &self,
        collection_id: CollectionId,
        destination: TrainingDestination,
    ) -> Option<anyhow::Result<String>> {
        let key = self.get_training_set_kv_key(collection_id, destination);
        self.kv.get(&key).await
    }

    pub async fn delete_training_set(
        &self,
        collection_id: CollectionId,
        destination: TrainingDestination,
    ) -> anyhow::Result<()> {
        let key = self.get_training_set_kv_key(collection_id, destination);
        self.kv.delete_with_prefix(&key).await
    }

    fn get_training_set_kv_key(
        &self,
        collection_id: CollectionId,
        destination: TrainingDestination,
    ) -> String {
        format!("training_set:{collection_id}:{destination}")
    }
}

pub struct TrainingSet {
    pub collection_id: CollectionId,
    pub read_side: Arc<ReadSide>,
    pub read_api_key: ReadApiKey,
    pub llm_service: Arc<LLMService>,
    pub llm_config: Option<InteractionLLMConfig>,
}

impl TrainingSet {
    pub fn new(
        collection_id: CollectionId,
        read_side: Arc<ReadSide>,
        read_api_key: ReadApiKey,
        llm_service: Arc<LLMService>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Self {
        TrainingSet {
            collection_id,
            read_side,
            read_api_key,
            llm_service,
            llm_config,
        }
    }

    pub async fn generate_training_data_for(
        &self,
        destination: TrainingDestination,
    ) -> anyhow::Result<TrainingSetsQueriesOptimizerResponse> {
        match destination {
            TrainingDestination::QueryPlanner => {
                println!("@todo");
                unimplemented!()
            }
            TrainingDestination::QueryOptimizer => {
                return self.generate_query_optimizer_training_data().await;
            }
            TrainingDestination::QueryFiltering => {
                println!("@todo");
                unimplemented!()
            }
        }
    }

    async fn generate_query_optimizer_training_data(
        &self,
    ) -> Result<TrainingSetsQueriesOptimizerResponse> {
        let random_documents = self.get_random_data_from_collection(5usize).await?;
        let variables = vec![("documents".to_string(), random_documents.join("\n"))];
        let random_queries = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::TrainingSetsQueriesGenerator,
                vec![],
                variables,
                None,
                self.llm_config.clone(),
            )
            .await?;

        let repaired_random_queries =
            llm_json::repair_json(&random_queries, &llm_json::RepairOptions::default())?;
        let ser_random_queries: TrainingSetsQueriesGeneratorResponse =
            serde_json::from_str(&repaired_random_queries)?;

        let mut results = TrainingSetsQueriesOptimizerResponse {
            simple: vec![],
            multiple_terms: vec![],
            advanced: vec![],
        };

        results.simple = self.optimize_queries(&ser_random_queries.simple).await?;
        results.multiple_terms = self
            .optimize_queries(&ser_random_queries.multiple_terms)
            .await?;
        results.advanced = self.optimize_queries(&ser_random_queries.advanced).await?;

        Ok(results)
    }

    async fn optimize_queries(
        &self,
        queries: &[String],
    ) -> Result<Vec<TrainingSetQueriesOptimizerQuerySet>> {
        let mut optimized_queries = Vec::new();

        for query in queries.iter() {
            let messages = vec![InteractionMessage {
                content: query.clone(),
                role: crate::types::Role::User,
            }];

            let messages_as_json = serde_json::to_string(&messages)?;
            let current_time = chrono::Utc::now().to_string();
            let current_timestamp = chrono::Utc::now().timestamp();

            let variables = vec![
                ("conversation".to_string(), messages_as_json),
                ("timestamp".to_string(), current_time),
                ("timestamp".to_string(), current_timestamp.to_string()),
            ];

            let resp = self
                .llm_service
                .run_known_prompt(
                    KnownPrompts::TrainingSetsQueriesOptimizer,
                    vec![],
                    variables,
                    None,
                    self.llm_config.clone(),
                )
                .await?;

            let repaired = llm_json::repair_json(&resp, &llm_json::RepairOptions::default())?;

            let deserialized: Vec<String> = match serde_json::from_str(&repaired) {
                Ok(vec) => vec,
                Err(_e) => match serde_json::from_str::<serde_json::Value>(&repaired) {
                    Ok(serde_json::Value::Object(obj)) => {
                        if let Some(optimized) = obj.get("optimized") {
                            if let Some(arr) = optimized.as_array() {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            } else if let Some(s) = optimized.as_str() {
                                vec![s.to_string()]
                            } else {
                                vec![format!(
                                    "Error: Unexpected optimized format: {:?}",
                                    optimized
                                )]
                            }
                        } else {
                            obj.values()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        }
                    }
                    Ok(serde_json::Value::String(s)) => vec![s],
                    _ => {
                        vec![query.clone()]
                    }
                },
            };

            optimized_queries.push(TrainingSetQueriesOptimizerQuerySet {
                original: query.clone(),
                optimized: deserialized,
            });
        }

        Ok(optimized_queries)
    }

    async fn get_random_data_from_collection(&self, size: usize) -> Result<Vec<String>> {
        let collection_stats = self
            .read_side
            .collection_stats(
                self.read_api_key,
                self.collection_id,
                CollectionStatsRequest { with_keys: false },
            )
            .await?;

        let docs_in_collection = collection_stats.document_count;
        let random_offset = if docs_in_collection <= size {
            0
        } else {
            rand::rng().random_range(0..docs_in_collection - size)
        };

        let search_params = SearchParams {
            mode: SearchMode::FullText(FulltextMode {
                term: "".to_string(),
                exact: false,
                threshold: None,
                tolerance: None,
            }),
            limit: Limit(size),
            offset: SearchOffset(random_offset),
            boost: HashMap::new(),
            facets: HashMap::new(),
            indexes: None,
            properties: Properties::Star,
            sort_by: None,
            user_id: None,
            where_filter: WhereFilter {
                filter_on_fields: vec![],
                and: None,
                or: None,
                not: None,
            },
            group_by: None,
        };

        let random_search_results = self
            .read_side
            .search(
                self.read_api_key,
                self.collection_id,
                SearchRequest {
                    search_params,
                    analytics_metadata: None,
                    interaction_id: None,
                    search_analytics_event_origin: None,
                },
            )
            .await?;

        let as_json_docs = random_search_results
            .hits
            .iter()
            .map(|hit| serde_json::to_string(&hit.document))
            .collect::<Result<Vec<String>, _>>()?;

        Ok(as_json_docs)
    }
}
