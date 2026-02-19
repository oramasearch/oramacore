use orama_js_pool::OutputChannel;
use oramacore_lib::{
    data_structures::ShouldInclude,
    filters::{DocId, FilterResult, PlainFilterResult},
    hook_storage::HookType,
    pin_rules::{Consequence, PinRulesReader},
};
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::Arc,
    time::Instant,
};

use crate::{
    collection_manager::sides::read::{
        analytics::{
            AnalyticsMetadataFromRequest, OramaCoreAnalytics, SearchAnalyticEventOrigin,
            SearchAnalyticEventV1,
        },
        collection::ReadIndexesLockGuard,
        facet::{FacetContext, FacetParams},
        filter::FilterContext,
        group::{GroupContext, GroupParams},
        sort::{sort_groups, sort_token_scores},
        token_score::{TokenScoreContext, TokenScoreParams},
        GroupValue, ReadError,
    },
    metrics::{search::SEARCH_CALCULATION_TIME, SearchCollectionLabels},
    types::{
        DocumentId, FacetResult, GroupedResult, RawJSONDocument, ScoreMode, SearchMode,
        SearchParams, SearchResult, SearchResultHit, TokenScore,
    },
};

use super::collection::CollectionReader;

use tracing::{error, info, trace};

/// Applies OMC (Orama Custom Multiplier) to token scores.
/// Documents with an OMC value have their scores multiplied by that value.
/// Uncommitted values take precedence over committed values.
fn apply_omc_multipliers(
    scores: &mut HashMap<DocumentId, f32>,
    uncommitted_omc: &HashMap<DocumentId, f32>,
    committed_omc: &HashMap<DocumentId, f32>,
) {
    if uncommitted_omc.is_empty() && committed_omc.is_empty() {
        return;
    }
    for (doc_id, score) in scores.iter_mut() {
        // Check uncommitted first (newer values), then committed
        if let Some(multiplier) = uncommitted_omc
            .get(doc_id)
            .or_else(|| committed_omc.get(doc_id))
        {
            *score *= multiplier;
        }
    }
}

pub struct SearchRequest {
    pub search_params: SearchParams,
    pub search_analytics_event_origin: Option<SearchAnalyticEventOrigin>,
    pub analytics_metadata: Option<AnalyticsMetadataFromRequest>,
    pub interaction_id: Option<String>,
}

pub struct HookConfig {
    pub log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    pub secrets: Arc<HashMap<String, String>>,
}

pub struct Search<'collection, 'analytics_storage> {
    collection: &'collection CollectionReader,
    analytics_storage: Option<&'analytics_storage OramaCoreAnalytics>,
    request: SearchRequest,
    score_mode: ScoreMode,
    hook_config: HookConfig,
}

impl<'collection, 'analytics_storage> Search<'collection, 'analytics_storage> {
    pub fn new(
        collection: &'collection CollectionReader,
        analytics_storage: Option<&'analytics_storage OramaCoreAnalytics>,
        request: SearchRequest,
        score_mode: ScoreMode,
        hook_config: HookConfig,
    ) -> Self {
        Self {
            collection,
            analytics_storage,
            request,
            score_mode,
            hook_config,
        }
    }

    pub async fn execute(self) -> Result<SearchResult, ReadError> {
        let start = Instant::now();

        let Self {
            collection,
            analytics_storage,
            request,
            score_mode,
            hook_config,
        } = self;
        let HookConfig {
            log_sender,
            secrets,
        } = hook_config;
        let SearchRequest {
            search_params,
            search_analytics_event_origin,
            analytics_metadata,
            interaction_id,
        } = request;

        let collection_id = collection.id();

        let has_filters = !search_params.where_filter.is_empty();
        let has_facets = !search_params.facets.is_empty();
        let has_groups = search_params.group_by.is_none();
        let has_sorting = search_params.sort_by.is_none();

        let m = SEARCH_CALCULATION_TIME.create(SearchCollectionLabels {
            collection: collection_id.to_string().into(),
            mode: search_params.mode.as_str(),
            has_filter: if has_filters { "true" } else { "false" },
            has_facet: if has_facets { "true" } else { "false" },
            has_group: if has_groups { "true" } else { "false" },
            has_sorting: if has_sorting { "true" } else { "false" },
        });

        // Let's suppose the number of matching document is 1/3 of the total number of documents
        // This is a very rough estimation, but it is better than nothing
        // This allows us to avoid reallocation of the result map
        // TODO: we can use an Area allocator to reuse the memory between searches
        let matching_document_count_estimation = collection.get_document_count_estimation() / 3;

        let indexes = collection
            .get_index_read_locks(search_params.indexes.as_ref())
            .await?;

        let pin_rule_consequences = extract_pin_rules(
            &*collection.get_pin_rules_reader("search").await,
            &search_params,
            &indexes,
        );

        let has_pin_rules = !pin_rule_consequences.is_empty();

        let (top_results, count, facets, groups) = search_on_indexes(
            &indexes,
            &search_params,
            &score_mode,
            matching_document_count_estimation,
            pin_rule_consequences,
        )
        .await?;

        let collection_document_storage = collection.get_document_storage();
        let group = groups
            .as_ref()
            .map(|g| {
                Box::new(
                    g.values()
                        .flat_map(|token_scores| token_scores.iter().map(|ts| ts.document_id)),
                ) as Box<dyn Iterator<Item = DocumentId>>
            })
            .unwrap_or_else(|| Box::new(std::iter::empty()));
        let all_docs_ids = top_results.iter().map(|ts| ts.document_id).chain(group);
        let mut docs = collection_document_storage
            .get_documents_by_ids(all_docs_ids.collect())
            .await?;

        // Hook signature: function transformDocumentAfterSearch(documents, collectionValues, secrets)
        // - documents: the fetched documents from storage
        // - collectionValues: key-value pairs associated with the collection
        // - secrets: secrets fetched from external providers (e.g. AWS Secrets Manager)
        // Note: We must clone here because, if hook returns None (null/undefined in JS),
        // we need to preserve original docs.
        let hook_input = docs.clone();
        let collection_values = collection.list_values().await;
        let result: Option<Vec<(DocumentId, Arc<RawJSONDocument>)>> = collection
            .run_hook(
                HookType::TransformDocumentAfterSearch,
                (hook_input, collection_values, secrets),
                log_sender,
            )
            .await?;

        let mut count = count;
        if let Some(modified_documents) = result {
            info!("Documents modified by TransformDocumentAfterSearch hook");
            count = modified_documents.len();
            docs = modified_documents;
        }

        let hits = top_results
            .into_iter()
            .filter_map(|ts| {
                let TokenScore {
                    document_id: id,
                    score,
                } = ts;
                docs.iter()
                    .find(|(doc_id, _)| doc_id == &id)
                    .cloned()
                    .map(|(_, doc)| {
                        let doc_id = doc.id.clone();
                        SearchResultHit {
                            id: doc_id.unwrap_or_default(),
                            score,
                            document: Some(doc),
                        }
                    })
            })
            .collect();

        let groups = if let Some(grouped_results) = groups {
            let mut final_groups = Vec::new();
            for (group_value, token_scores) in grouped_results {
                let mut hits = Vec::new();
                for token_score in token_scores {
                    if let Some((_, doc)) = docs
                        .iter()
                        .find(|(d_id, _)| d_id == &token_score.document_id)
                    {
                        let doc_id = doc.id.clone().unwrap_or_default();
                        hits.push(SearchResultHit {
                            id: doc_id,
                            score: token_score.score,
                            document: Some(doc.clone()),
                        });
                    }
                }

                let group = GroupedResult {
                    result: hits,
                    values: group_value.into_iter().map(|v| v.into()).collect(),
                };

                final_groups.push(group);
            }
            Some(final_groups)
        } else {
            None
        };

        drop(m);

        let result = SearchResult {
            count,
            hits,
            facets: Some(facets),
            groups,
        };

        let search_duration = start.elapsed();
        info!(duration=?search_duration, "Search completed");

        if let Some(analytics_storage) = analytics_storage.as_ref() {
            if let Some(search_analytics_event_origin) = search_analytics_event_origin {
                match SearchAnalyticEventV1::try_new(
                    collection_id,
                    &search_params,
                    &result,
                    search_analytics_event_origin,
                    search_duration,
                    has_pin_rules,
                    false,
                    analytics_metadata,
                    interaction_id,
                ) {
                    Ok(ev) => {
                        if let Err(e) = analytics_storage.add_event(ev) {
                            error!(?e, "Failed to add search event to analytics storage");
                        }
                    }
                    Err(e) => {
                        error!(?e, "Failed to create search event for analytics. Ignored");
                    }
                };
            }
        }

        Ok(result)
    }
}

pub fn extract_term_from_search_mode(search_mode: &SearchMode) -> &str {
    match search_mode {
        SearchMode::FullText(f) | SearchMode::Default(f) => &f.term,
        SearchMode::Hybrid(h) => &h.term,
        SearchMode::Vector(v) => &v.term,
        SearchMode::Auto(a) => &a.term,
    }
}

fn extract_pin_rules<'collection>(
    rules: &PinRulesReader<DocumentId>,
    search_params: &SearchParams,
    indexes: &ReadIndexesLockGuard<'collection>,
) -> Vec<Consequence<DocumentId>> {
    let term = extract_term_from_search_mode(&search_params.mode);
    let mut consequences: Vec<_> = indexes
        .iter()
        .flat_map(|index| {
            let text_parser = index.get_text_parser();
            rules.apply(term, text_parser)
        })
        .collect();

    consequences.sort_by(|a, b| {
        a.promote
            .iter()
            .map(|item| (item.position, item.doc_id))
            .cmp(b.promote.iter().map(|item| (item.position, item.doc_id)))
    });

    consequences.dedup();

    consequences
}

async fn search_on_indexes(
    indexes: &ReadIndexesLockGuard<'_>,
    search_params: &SearchParams,
    score_mode: &ScoreMode,
    matching_document_count_estimation: u64,
    pin_rule_consequences: Vec<Consequence<DocumentId>>,
) -> Result<
    (
        Vec<TokenScore>,
        usize,
        HashMap<String, FacetResult>,
        Option<HashMap<Vec<GroupValue>, Vec<TokenScore>>>,
    ),
    ReadError,
> {
    let mut token_score_results: HashMap<DocumentId, f32> =
        HashMap::with_capacity((matching_document_count_estimation) as usize);
    let mut facets_results = HashMap::new();
    let mut group_results = HashMap::new();

    let mut stores = Vec::with_capacity(indexes.len());

    for index in indexes.iter() {
        let search_store = index.get_search_store().await;

        let filter_context = FilterContext::new(
            search_store.document_count,
            search_store.path_to_field_id_map,
            &search_store.uncommitted_fields,
            &search_store.committed_fields,
            search_store.uncommitted_deleted_documents,
        );
        let filtered_document_ids = filter_context.execute_filter(&search_params.where_filter)?;

        trace!("Calculating token scores for index {}", index.id());
        let token_score_context = TokenScoreContext::new(
            index.id(),
            search_store.document_count,
            &search_store.uncommitted_fields,
            &search_store.committed_fields,
            search_store.text_parser,
            search_store.read_side_context,
            search_store.path_to_field_id_map,
        );
        token_score_context.execute(
            &TokenScoreParams {
                mode: score_mode,
                boost: &search_params.boost,
                properties: &search_params.properties,
                limit_hint: search_params.limit,
                filtered_doc_ids: filtered_document_ids.as_ref(),
            },
            &mut token_score_results,
        )?;

        // Apply OMC (Orama Custom Multiplier) to the token scores for this index.
        // OMC values are stored per-index, so we need to apply them for each index.
        let omc_lock = index.get_all_omc().await;
        let (uncommitted_omc, committed_omc) = &**omc_lock;
        apply_omc_multipliers(&mut token_score_results, uncommitted_omc, committed_omc);
        drop(omc_lock);

        if !search_params.facets.is_empty() {
            // Orama provides a UI component that shows the search results
            // and the number of how many documents fall in each variant of field value (facets).
            // By default, the "all" category is selected and list all the results,
            // but if a user clicks on a category, it will see the results filtered by that category.
            // Hence, the second call has filters.
            // Anyway, the second call re-ask the number for each variant. That means,
            // the call contains facets and filters, and the returned facets are affected by the chosen category.
            // So, the user will see the facets number change when it clicks on a category.
            // And this is weird.
            // So, the pair (facets, filters) has to be reconsidered:
            // - we calculate directly the facets if there are no filters
            // - we recalculate the score without any filters and after the facets
            // NB: This is not performant, but required for a good UI.
            //     The FE could reuse the facets calculation of the first call
            //     keeping the results in memory.
            let token_scores: Cow<'_, HashMap<DocumentId, f32>> =
                if search_params.where_filter.is_empty() {
                    Cow::Borrowed(&token_score_results)
                } else {
                    let filtered_doc_ids = Some(FilterResult::Not(Box::new(FilterResult::Filter(
                        PlainFilterResult::from_iter(
                            search_store.document_count,
                            search_store.uncommitted_deleted_documents.iter().cloned(),
                        ),
                    ))));

                    let mut token_score_results_for_facets =
                        HashMap::with_capacity((matching_document_count_estimation) as usize);

                    let token_score_context = TokenScoreContext::new(
                        index.id(),
                        search_store.document_count,
                        &search_store.uncommitted_fields,
                        &search_store.committed_fields,
                        search_store.text_parser,
                        search_store.read_side_context,
                        search_store.path_to_field_id_map,
                    );
                    token_score_context.execute(
                        &TokenScoreParams {
                            mode: score_mode,
                            boost: &search_params.boost,
                            properties: &search_params.properties,
                            limit_hint: search_params.limit,
                            filtered_doc_ids: filtered_doc_ids.as_ref(),
                        },
                        &mut token_score_results_for_facets,
                    )?;

                    Cow::Owned(token_score_results_for_facets)
                };

            trace!("Calculating facets for index {}", index.id());
            let facet_context = FacetContext::new(
                search_store.path_to_field_id_map,
                &search_store.uncommitted_fields,
                &search_store.committed_fields,
            );

            facet_context.execute(
                &FacetParams {
                    facets: &search_params.facets,
                    token_scores: &token_scores,
                },
                &mut facets_results,
            )?;
        }

        if let Some(group_by) = &search_params.group_by {
            let group_context = GroupContext::new(
                search_store.path_to_field_id_map,
                &search_store.uncommitted_fields,
                &search_store.committed_fields,
            );

            group_context.execute(
                &GroupParams {
                    properties: &group_by.properties,
                },
                &mut group_results,
            )?;
        }

        stores.push(search_store);
    }

    if token_score_results.is_empty() {
        // Test if at least one index have the filter properties
        let all_keys: Vec<String> = search_params.where_filter.get_all_keys();
        for k in all_keys {
            let mut found = false;
            for index in indexes.iter() {
                if index.has_filter_field(&k) {
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(ReadError::FilterFieldNotFound(k));
            }
        }
    }

    let expected_facets_count = search_params.facets.len();
    if facets_results.len() != expected_facets_count {
        // The user asked for some facets that are not present in the collection
        let present_facet_fields: Vec<&String> = facets_results.keys().collect();
        let requested_facet_fields: Vec<&String> = search_params.facets.keys().collect();
        let missing_facet_fields: Vec<String> = requested_facet_fields
            .iter()
            .filter(|f| !present_facet_fields.contains(f))
            .cloned()
            .cloned()
            .collect();
        return Err(ReadError::FacetFieldNotFound(missing_facet_fields));
    }

    let group_by_results = if let Some(group_by_config) = &search_params.group_by {
        let sorted_group_results = sort_groups(
            &stores,
            &token_score_results,
            group_results,
            group_by_config.max_results,
            search_params.sort_by.as_ref(),
            &pin_rule_consequences,
        )?;

        Some(sorted_group_results)
    } else {
        None
    };

    // This is all matching documents across all indexes
    let count = token_score_results.len();

    let top_results = sort_token_scores(
        &stores,
        &token_score_results,
        search_params.limit,
        search_params.offset,
        search_params.sort_by.as_ref(),
        &pin_rule_consequences,
    )?;
    trace!("Top results: {:?} (of {})", top_results, count);

    let search_top_results = top_results
        .into_iter()
        .skip(search_params.offset.0)
        .take(search_params.limit.0)
        .collect::<Vec<_>>();

    Ok((search_top_results, count, facets_results, group_by_results))
}

pub struct SearchDocumentContext<'a, DocumentId> {
    deleted_documents: &'a HashSet<DocumentId>,
    pub filtered_doc_ids: Option<FilterResult<DocumentId>>,
}

impl<DocumentId: Sync + Send + Hash + Eq + DocId> ShouldInclude<DocumentId>
    for SearchDocumentContext<'_, DocumentId>
{
    fn should_include(&self, doc_id: &DocumentId) -> bool {
        if self.deleted_documents.contains(doc_id) {
            return false;
        }
        match &self.filtered_doc_ids {
            Some(filtered_doc_ids) => filtered_doc_ids.contains(doc_id),
            None => true,
        }
    }
}

#[cfg(test)]
impl ShouldInclude<DocumentId> for () {
    fn should_include(&self, _: &DocumentId) -> bool {
        true
    }
}
