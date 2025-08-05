use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::collection_manager::sides::read::AnalyticSearchEventInvocationType;
use crate::types::InteractionLLMConfig;
use crate::{
    ai::llms::LLMService,
    collection_manager::sides::read::ReadSide,
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, FulltextMode, Limit, Properties, SearchMode,
        SearchOffset, SearchParams, WhereFilter,
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
    QueryPlannerTrainingData,
    QueryOptimizerTrainingData,
    // @todo: add query filtering
}

#[derive(Debug, Serialize)]
pub enum TraningDestination {
    QueryPlanner,
    QueryOptimizer,
    QueryFiltering,
}

#[derive(Debug, Deserialize, Clone)]
struct TrainingSetsQueriesGeneratorResponse {
    pub simple: Vec<String>,
    pub multiple_terms: Vec<String>,
    pub advanced: Vec<String>,
}

pub struct TrainingSet {
    pub collection_id: CollectionId,
    pub read_side: Arc<ReadSide>,
    pub read_api_key: ApiKey,
    pub llm_service: Arc<LLMService>,
    pub llm_config: Option<InteractionLLMConfig>,
}

impl TrainingSet {
    pub fn new(
        collection_id: CollectionId,
        read_side: Arc<ReadSide>,
        read_api_key: ApiKey,
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
        destination: TraningDestination,
    ) -> anyhow::Result<TrainingData> {
        match destination {
            TraningDestination::QueryPlanner => {
                println!("@todo");
            }
            TraningDestination::QueryOptimizer => {
                self.generate_query_optimizer_training_data().await?;
            }
            TraningDestination::QueryFiltering => {
                println!("@todo");
            }
        }

        unimplemented!()
    }

    // will return QueryOptimizerTrainingData
    async fn generate_query_optimizer_training_data(
        &self,
    ) -> Result<TrainingSetsQueriesGeneratorResponse> {
        let random_documents = self.get_random_data_from_collection(5usize).await?;
        let variables = vec![("documents".to_string(), random_documents.join("\n"))];
        let random_queries = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::TrainingSetsQueriesGenerator,
                variables,
                self.llm_config.clone(),
            )
            .await?;

        let repaired = llm_json::repair_json(&random_queries, &llm_json::RepairOptions::default())?;

        let random_queries: TrainingSetsQueriesGeneratorResponse = serde_json::from_str(&repaired)?;

        Ok(random_queries)
    }

    async fn get_random_data_from_collection(&self, size: usize) -> Result<Vec<String>> {
        let collection_stats = self
            .read_side
            .collection_stats(
                self.read_api_key.clone(),
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
        };

        let random_search_results = self
            .read_side
            .search(
                self.read_api_key.clone(),
                self.collection_id,
                search_params,
                AnalyticSearchEventInvocationType::TrainingDataGen,
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
