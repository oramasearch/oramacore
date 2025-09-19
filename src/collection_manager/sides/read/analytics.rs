use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use oramacore_lib::analytics::{AnalyticConfig, AnalyticLogStream, AnalyticsStorage};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::read::ReadSide,
    types::{ApiKey, CollectionId, Interaction, Role, SearchMode, SearchParams, SearchResult},
};

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
    pub metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Default)]
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
pub struct InteractionAnalyticEventV1 {
    #[serde(rename = "ts")]
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "coll")]
    pub collection_id: CollectionId,
    #[serde(rename = "conv")]
    pub conversation_id: String,
    #[serde(rename = "inter_id")]
    pub interaction_id: String,
    #[serde(rename = "v_id", skip_serializing_if = "Option::is_none")]
    pub visitor_id: Option<String>,
    #[serde(rename = "sysprt_id", skip_serializing_if = "Option::is_none")]
    pub system_prompt_id: Option<String>,
    #[serde(rename = "usr_msg")]
    pub user_message: String,
    #[serde(rename = "asst_res")]
    pub assistant_response: String,
    #[serde(rename = "mp")]
    pub model_provider: String,
    #[serde(rename = "mn")]
    pub model_name: String,
    #[serde(rename = "cxt")]
    pub full_context: Option<String>,
    #[serde(rename = "gq", skip_serializing_if = "Option::is_none")]
    pub generated_related_queries: Option<String>,
    #[serde(rename = "rs")]
    pub rag_steps: String,
    #[serde(rename = "uit")]
    pub user_input_tokens: u32,
    #[serde(rename = "uot")]
    pub output_tokens: u32,
    #[serde(rename = "tps")]
    pub tokens_per_second: f32,
    #[serde(rename = "d")]
    pub duration: Duration,
    #[serde(rename = "ttft")]
    pub time_to_first_token: Duration,
    #[serde(rename = "err", skip_serializing_if = "Option::is_none")]
    pub interaction_error: Option<String>,
    #[serde(rename = "md", skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum AnalyticEvent {
    #[serde(rename = "sv1")]
    SearchV1(SearchAnalyticEventV1),
    #[serde(rename = "iv1")]
    InteractionV1(InteractionAnalyticEventV1),
}

impl From<SearchAnalyticEventV1> for AnalyticEvent {
    fn from(event: SearchAnalyticEventV1) -> Self {
        AnalyticEvent::SearchV1(event)
    }
}

impl From<InteractionAnalyticEventV1> for AnalyticEvent {
    fn from(event: InteractionAnalyticEventV1) -> Self {
        AnalyticEvent::InteractionV1(event)
    }
}

fn serialize_bool_as_int<S>(b: &bool, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_u8(if *b { 1 } else { 0 })
}

pub struct AnalyticsHolder {
    read_side: Arc<ReadSide>,
    start: Instant,
    start_time: DateTime<Utc>,
    collection_id: CollectionId,
    user_message: String,
    interaction_id: String,
    conversation_id: String,
    visitor_id: String,
    analytics_metadata: AnalyticsMetadataFromRequest,
    generated_related_queries: Option<String>,
    llm_provider: Option<String>,
    llm_model: Option<String>,
    system_prompt_id: Option<String>,
    assistant_response: Option<String>,
    full_context: Option<String>,
    rag_steps: Vec<serde_json::Value>,
    output_tokens: Option<u32>,
    user_input_tokens: Option<u32>,
    tokens_per_second: Option<f32>,
    time_to_first_token: Option<Duration>,
    interaction_error: Option<String>,
}

impl AnalyticsHolder {
    pub fn new(
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        interaction: &Interaction,
        analytics_metadata: AnalyticsMetadataFromRequest,
    ) -> Self {
        // TODO: avoid 2 different time sources
        let start_time = chrono::Utc::now();
        let start = Instant::now();

        let user_message = interaction
            .messages
            .iter()
            .rev()
            .find(|i| i.role == Role::User)
            .map(|i| i.content.clone())
            .unwrap_or_default();
        let user_input_tokens = if let Ok(t) = tiktoken_rs::o200k_base() {
            let len = t.encode(&user_message, Default::default()).len();
            Some(len as u32)
        } else {
            None
        };

        let interaction_id = interaction.interaction_id.clone();
        let conversation_id = interaction.conversation_id.clone();
        let visitor_id = interaction.visitor_id.clone();

        Self {
            read_side,
            start_time,
            collection_id,
            user_message,
            interaction_id,
            conversation_id,
            visitor_id,
            analytics_metadata,
            start,
            generated_related_queries: None,
            llm_model: None,
            llm_provider: None,
            assistant_response: None,
            full_context: None,
            rag_steps: vec![],
            system_prompt_id: None,
            interaction_error: None,
            output_tokens: None,
            time_to_first_token: None,
            tokens_per_second: None,
            user_input_tokens,
        }
    }

    pub(crate) fn set_generated_related_queries(&mut self, queries: String) {
        self.generated_related_queries = Some(queries);
    }

    pub(crate) fn set_llm_info(&mut self, provider: String, model: String) {
        self.llm_provider = Some(provider);
        self.llm_model = Some(model);
    }

    pub(crate) fn set_system_prompt_id(&mut self, system_prompt_id: String) {
        self.system_prompt_id = Some(system_prompt_id);
    }

    pub(crate) fn set_time_to_first_token(&mut self, time_to_first_token: Duration) {
        self.time_to_first_token = Some(time_to_first_token);
    }

    pub(crate) fn set_assistant_response(&mut self, assistant_response: String, delta: Duration) {
        if let Ok(t) = tiktoken_rs::o200k_base() {
            let len = t.encode(&assistant_response, Default::default()).len();
            self.output_tokens = Some(len as u32);
            self.tokens_per_second = Some(len as f32 / delta.as_secs() as f32);
        }

        self.assistant_response = Some(assistant_response);
    }

    pub(crate) fn set_full_context(&mut self, full_context: String) {
        self.full_context = Some(full_context);
    }

    pub(crate) fn set_rag_steps(&mut self, rag_steps: Vec<serde_json::Value>) {
        self.rag_steps.extend(rag_steps);
    }

    //// MISSING
    pub(crate) fn set_error(&mut self, interaction_error: String) {
        self.interaction_error = Some(interaction_error);
    }
}

impl Drop for AnalyticsHolder {
    fn drop(&mut self) {
        if let Some(analytics_logs) = self.read_side.get_analytics_logs() {
            let duration = self.start.elapsed();

            let rag_steps = std::mem::take(&mut self.rag_steps);
            let rag_steps = match serde_json::to_string(&rag_steps) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(error = ?e, "Cannot serialize rag_steps for analytics");
                    "[]".to_string()
                }
            };

            if let Err(err) = analytics_logs.add_event(InteractionAnalyticEventV1 {
                timestamp: self.start_time,
                collection_id: self.collection_id,
                conversation_id: self.conversation_id.clone(),
                interaction_id: self.interaction_id.clone(),
                visitor_id: Some(self.visitor_id.clone()),
                user_message: self.user_message.clone(),
                model_provider: self.llm_provider.take().unwrap_or_default(),
                model_name: self.llm_model.take().unwrap_or_default(),
                generated_related_queries: self.generated_related_queries.take(),
                metadata: self.analytics_metadata.headers.clone(),
                duration,
                system_prompt_id: self.system_prompt_id.take(),
                assistant_response: self.assistant_response.take().unwrap_or_default(),
                full_context: self.full_context.take(),
                rag_steps,
                output_tokens: self.output_tokens.unwrap_or_default(),
                user_input_tokens: self.user_input_tokens.unwrap_or_default(),
                tokens_per_second: self.tokens_per_second.unwrap_or_default(),
                time_to_first_token: self.time_to_first_token.unwrap_or_default(),
                interaction_error: self.interaction_error.clone(),
            }) {
                tracing::error!(error = ?err, "Cannot log interaction analytic event");
            }
        }
    }
}
