use std::{collections::HashMap, path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use oramacore_lib::analytics::{AnalyticConfig, AnalyticLogStream, AnalyticsStorage};
use serde::{Deserialize, Serialize};

use crate::types::{ApiKey, CollectionId, SearchMode, SearchParams, SearchResult};

#[derive(Deserialize, Clone)]
pub struct OramaCoreAnalyticConfig {
    pub api_key: ApiKey,
}

pub struct OramaCoreAnalytics {
    api_key: ApiKey,
    inner: AnalyticsStorage<AnalyticEvent>,
}

impl OramaCoreAnalytics {
    pub fn try_new(data_dir: PathBuf, config: OramaCoreAnalyticConfig) -> Result<Self> {
        let inner = AnalyticsStorage::try_new(AnalyticConfig { data_dir })?;
        Ok(Self {
            inner,
            api_key: config.api_key,
        })
    }

    pub fn add_event<EV: Into<AnalyticEvent>>(&self, event: EV) -> Result<()> {
        self.inner.add_event(event.into())
    }

    pub async fn get_and_erase(&self, api_key: ApiKey) -> Result<AnalyticLogStream> {
        if self.api_key != api_key {
            return Err(anyhow::anyhow!("Invalid analytics API key"));
        }
        self.inner.get_and_erase().await
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub enum SearchAnalyticEventOrigin {
    #[serde(rename = "direct")]
    Direct,
    #[serde(rename = "rag")]
    RAG,
    #[serde(rename = "mcp")]
    MCP,
    #[serde(rename = "nlp")]
    NLP,
}

#[derive(Serialize, Deserialize)]
pub enum SearchAnalyticEventSearchType {
    #[serde(rename = "fulltext")]
    Fulltext,
    #[serde(rename = "hybrid")]
    Hybrid,
    #[serde(rename = "vector")]
    Vector,
    #[serde(rename = "auto")]
    Auto,
}

#[derive(Serialize, Deserialize)]
pub struct SearchAnalyticEvent {
    pub timestamp: DateTime<Utc>,
    pub collection_id: CollectionId,
    pub origin: SearchAnalyticEventOrigin,
    pub search_type: SearchAnalyticEventSearchType,
    pub visitor_id: Option<String>,
    pub raw_search_term: String,
    pub raw_query: String,
    pub has_filters: bool,
    pub has_groups: bool,
    pub has_sorting: bool,
    pub has_facets: bool,
    pub has_pins_rules: bool,
    pub has_pinned_results: bool,
    pub results_count: usize,
    pub search_duration: Duration,
    pub results: String,
    pub metadata: HashMap<String, String>,
}

impl SearchAnalyticEvent {
    pub fn try_new(
        collection_id: CollectionId,
        search_params: &SearchParams,
        search_result: &SearchResult,
        origin: SearchAnalyticEventOrigin,
        search_duration: Duration,
        has_pins_rules: bool,
        has_pinned_results: bool,
    ) -> Result<Self> {
        let has_filters = search_params.where_filter.is_empty();
        let has_facets = !search_params.facets.is_empty();
        let has_groups = search_params.group_by.is_some();
        let has_sorting = search_params.sort_by.is_some();

        let results_count = search_result.hits.len();

        let raw_query = serde_json::to_string(&search_params)
            .context("Cannot serialize search query for analytics")?;
        let results = serde_json::to_string(&search_result)
            .context("Cannot serialize search result for analytics")?;

        let (raw_search_term, search_type) = match &search_params.mode {
            SearchMode::Auto(a) => (a.term.clone(), SearchAnalyticEventSearchType::Auto),
            SearchMode::Vector(v) => (v.term.clone(), SearchAnalyticEventSearchType::Vector),
            SearchMode::Hybrid(h) => (h.term.clone(), SearchAnalyticEventSearchType::Hybrid),
            SearchMode::Default(d) => (d.term.clone(), SearchAnalyticEventSearchType::Fulltext),
            SearchMode::FullText(f) => (f.term.clone(), SearchAnalyticEventSearchType::Fulltext),
        };

        Ok(Self {
            timestamp: Utc::now(),
            collection_id,
            origin,
            raw_search_term,
            raw_query,
            has_filters,
            has_groups,
            has_sorting,
            has_facets,
            has_pins_rules,
            has_pinned_results,
            results_count,
            search_duration,
            results,
            search_type,
            visitor_id: None,
            metadata: HashMap::new(),
        })
    }
}

#[derive(Serialize, Deserialize)]
pub enum AnalyticEvent {
    Search(SearchAnalyticEvent),
}

impl From<SearchAnalyticEvent> for AnalyticEvent {
    fn from(event: SearchAnalyticEvent) -> Self {
        AnalyticEvent::Search(event)
    }
}
