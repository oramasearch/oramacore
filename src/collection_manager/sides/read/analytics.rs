use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use oramacore_lib::analytics::{AnalyticConfig, AnalyticLogStream, AnalyticsStorage};
use serde::{Deserialize, Serialize};

use crate::types::{ApiKey, CollectionId, SearchMode, SearchParams, SearchResult};

#[derive(Deserialize, Clone)]
pub struct MetadataFfromHeadersPair {
    pub header: String,
    pub metadata_key: String,
}

#[derive(Deserialize, Clone)]
pub struct OramaCoreAnalyticConfig {
    pub api_key: ApiKey,
    pub metadata_from_headers: Vec<MetadataFfromHeadersPair>,
}

#[derive(Clone)]
pub struct OramaCoreAnalytics {
    api_key: ApiKey,
    inner: Arc<AnalyticsStorage<AnalyticEvent>>,
    metadata_from_headers: Vec<MetadataFfromHeadersPair>,
}

impl OramaCoreAnalytics {
    pub fn try_new(data_dir: PathBuf, config: OramaCoreAnalyticConfig) -> Result<Self> {
        let inner = AnalyticsStorage::try_new(AnalyticConfig { data_dir })?;
        Ok(Self {
            inner: Arc::new(inner),
            api_key: config.api_key,
            metadata_from_headers: config.metadata_from_headers,
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

    pub fn get_metadata_from_headers(&self) -> &[MetadataFfromHeadersPair] {
        &self.metadata_from_headers
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
    #[serde(rename = "f")]
    Fulltext,
    #[serde(rename = "h")]
    Hybrid,
    #[serde(rename = "v")]
    Vector,
    #[serde(rename = "a")]
    Auto,
}

#[derive(Serialize, Deserialize)]
pub struct SearchAnalyticEventV1 {
    #[serde(rename = "ts")]
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "coll")]
    pub collection_id: CollectionId,
    #[serde(rename = "o")]
    pub origin: SearchAnalyticEventOrigin,
    #[serde(rename = "st")]
    pub search_type: SearchAnalyticEventSearchType,
    #[serde(rename = "v_id", skip_serializing_if = "Option::is_none")]
    pub visitor_id: Option<String>,
    #[serde(rename = "i_id", skip_serializing_if = "Option::is_none")]
    pub interaction_id: Option<String>,
    #[serde(rename = "rst")]
    pub raw_search_term: String,
    #[serde(rename = "rq")]
    pub raw_query: String,
    #[serde(rename = "hflt", serialize_with = "serialize_bool_as_int")]
    pub has_filters: bool,
    #[serde(rename = "hg", serialize_with = "serialize_bool_as_int")]
    pub has_groups: bool,
    #[serde(rename = "hs", serialize_with = "serialize_bool_as_int")]
    pub has_sorting: bool,
    #[serde(rename = "hfct", serialize_with = "serialize_bool_as_int")]
    pub has_facets: bool,
    #[serde(rename = "hpr", serialize_with = "serialize_bool_as_int")]
    pub has_pins_rules: bool,
    #[serde(rename = "hpres", serialize_with = "serialize_bool_as_int")]
    pub has_pinned_results: bool,
    #[serde(rename = "rc")]
    pub results_count: usize,
    #[serde(rename = "sd")]
    pub search_duration: Duration,
    #[serde(rename = "r")]
    pub results: String,
    #[serde(rename = "md", skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>
}

#[derive(Serialize, Deserialize)]
pub struct AnalyticsMetadataFromRequest {
    #[serde(flatten)]
    pub headers: HashMap<String, String>,
}

impl AnalyticsMetadataFromRequest {
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }
}

impl SearchAnalyticEventV1 {
    pub fn try_new(
        collection_id: CollectionId,
        search_params: &SearchParams,
        search_result: &SearchResult,
        origin: SearchAnalyticEventOrigin,
        search_duration: Duration,
        has_pins_rules: bool,
        has_pinned_results: bool,
        analytics_metadata: Option<AnalyticsMetadataFromRequest>,
        interaction_id: Option<String>,
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

        let metadata = analytics_metadata.map(|a| a.headers).unwrap_or_default();

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
            visitor_id: search_params.user_id.clone(),
            interaction_id,
            metadata,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum AnalyticEvent {
    #[serde(rename = "sv1")]
    SearchV1(SearchAnalyticEventV1),
}

impl From<SearchAnalyticEventV1> for AnalyticEvent {
    fn from(event: SearchAnalyticEventV1) -> Self {
        AnalyticEvent::SearchV1(event)
    }
}

fn serialize_bool_as_int<S>(b: &bool, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_u8(if *b { 1 } else { 0 })
}
