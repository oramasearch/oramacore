use std::{borrow::Cow, collections::HashMap, time::Instant};

use anyhow::{bail, Result};
use futures::StreamExt;
use oramacore_lib::pin_rules::Consequence;

use crate::{
    collection_manager::sides::read::{
        analytics::{OramaCoreAnalytics, SearchAnalyticEvent, SearchAnalyticEventOrigin},
        collection::ReadIndexesLockGuard,
        document_storage::DocumentStorage,
        sort::sort_documents_in_groups,
        GroupValue, ReadError, SortContext,
    },
    metrics::{search::SEARCH_CALCULATION_TIME, SearchCollectionLabels},
    types::{
        DocumentId, FacetResult, GroupByConfig, GroupedResult, SearchMode, SearchParams,
        SearchResult, SearchResultHit, SortBy, TokenScore, WhereFilter,
    },
};

use super::collection::CollectionReader;

use tracing::{error, info, trace};

pub struct Search<'collection, 'document_storage, 'analytics_storage> {
    collection: &'collection CollectionReader,
    document_storage: &'document_storage DocumentStorage,
    analytics_storage: Option<&'analytics_storage OramaCoreAnalytics>,
    search_params: SearchParams,
    search_analytics_event_origin: Option<SearchAnalyticEventOrigin>,
}

impl<'collection, 'document_storage, 'analytics_storage>
    Search<'collection, 'document_storage, 'analytics_storage>
{
    pub fn new(
        collection: &'collection CollectionReader,
        document_storage: &'document_storage DocumentStorage,
        analytics_storage: Option<&'analytics_storage OramaCoreAnalytics>,
        search_params: SearchParams,
        search_analytics_event_origin: Option<SearchAnalyticEventOrigin>,
    ) -> Self {
        Self {
            collection,
            document_storage,
            analytics_storage,
            search_params,
            search_analytics_event_origin,
        }
    }

    pub async fn execute(self) -> Result<SearchResult, ReadError> {
        let start = Instant::now();

        let Self {
            collection,
            document_storage,
            analytics_storage,
            search_params,
            search_analytics_event_origin,
        } = self;

        let collection_id = collection.id();

        let limit = search_params.limit;
        let offset = search_params.offset;

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

        let index_ids = collection
            .get_index_ids(search_params.indexes.as_ref())
            .await?;

        // Let's suppose the number of matching document is 1/3 of the total number of documents
        // This is a very rough estimation, but it is better than nothing
        // This allows us to avoid reallocation of the result map
        // TODO: we can use an Area allocator to reuse the memory between searches
        let document_count_estimation = collection.get_document_count_estimation();
        let mut token_scores: HashMap<DocumentId, f32> =
            HashMap::with_capacity((document_count_estimation / 3) as usize);

        let indexes = collection.get_indexes_lock(&index_ids).await?;
        calculate_token_score_for_indexes(&indexes, &search_params, &mut token_scores).await?;

        let facets = if has_facets {
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
                    Cow::Borrowed(&token_scores)
                } else {
                    let mut token_scores: HashMap<DocumentId, f32> =
                        HashMap::with_capacity((document_count_estimation / 3) as usize);

                    let mut search_params_without_filters = search_params.clone();
                    search_params_without_filters.where_filter = WhereFilter {
                        filter_on_fields: vec![],
                        and: None,
                        not: None,
                        or: None,
                    };
                    calculate_token_score_for_indexes(
                        &indexes,
                        &search_params_without_filters,
                        &mut token_scores,
                    )
                    .await?;

                    Cow::Owned(token_scores)
                };

            let facets = calculate_facets(&indexes, &search_params, token_scores.as_ref()).await?;

            Some(facets)
        } else {
            None
        };

        let pin_rules = extract_pin_rules(&indexes, &search_params).await;

        let groups = if let Some(group_by) = &search_params.group_by {
            let groups = calculate_groups(
                &indexes,
                &token_scores,
                group_by,
                search_params.sort_by.as_ref(),
                &pin_rules,
            )
            .await?;

            let groups: Vec<_> = futures::stream::iter(groups.into_iter())
                .filter_map(|(k, ids)| async {
                    if ids.is_empty() {
                        return None;
                    }

                    let docs = document_storage
                        .get_documents_by_ids(ids.clone())
                        .await
                        .ok()?;

                    Some(GroupedResult {
                        values: k.into_iter().map(|v| v.into()).collect(),
                        result: docs
                            .into_iter()
                            .zip(ids.into_iter())
                            .take(group_by.max_results)
                            .map(|(document, doc_id)| {
                                let id = document
                                    .as_ref()
                                    .and_then(|d| d.id.clone())
                                    .unwrap_or_default();

                                SearchResultHit {
                                    id,
                                    score: *token_scores.get(&doc_id).unwrap_or(&0.0),
                                    document,
                                }
                            })
                            .collect(),
                    })
                })
                .collect()
                .await;

            Some(groups)
        } else {
            None
        };

        let count = token_scores.len();

        let top_results: Vec<TokenScore> =
            sort_and_truncate(&indexes, &token_scores, &search_params, &pin_rules).await?;
        trace!("Top results: {:?}", top_results);

        let result = top_results
            .into_iter()
            .skip(offset.0)
            .take(limit.0)
            .collect::<Vec<_>>();

        let docs = document_storage
            .get_documents_by_ids(result.iter().map(|m| m.document_id).collect())
            .await?;

        trace!("Calculates hits");
        let hits: Vec<_> = result
            .into_iter()
            .zip(docs)
            .map(|(token_score, document)| {
                let id = document
                    .as_ref()
                    .and_then(|d| d.id.clone())
                    .unwrap_or_default();
                SearchResultHit {
                    id,
                    score: token_score.score,
                    document,
                }
            })
            .collect();

        drop(m);

        let result = SearchResult {
            count,
            hits,
            facets,
            groups,
        };

        let search_duration = start.elapsed();
        info!(duration=?search_duration, "Search completed");

        if let Some(analytics_storage) = analytics_storage.as_ref() {
            if let Some(search_analytics_event_origin) = search_analytics_event_origin {
                match SearchAnalyticEvent::try_new(
                    collection_id,
                    &search_params,
                    &result,
                    search_analytics_event_origin,
                    search_duration,
                    !pin_rules.is_empty(),
                    false,
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

async fn extract_pin_rules(
    indexes: &ReadIndexesLockGuard<'_, '_>,
    search_params: &SearchParams,
) -> Vec<Consequence<DocumentId>> {
    let term = extract_term_from_search_mode(&search_params.mode);
    let mut pins: Vec<_> = Vec::with_capacity(indexes.len());
    for index in indexes.iter() {
        let pin_rules = index.get_read_lock_on_pin_rules().await;
        pins.extend(pin_rules.apply(term));
    }
    pins
}

async fn calculate_token_score_for_indexes(
    indexes: &ReadIndexesLockGuard<'_, '_>,
    search_params: &SearchParams,
    token_scores: &mut HashMap<DocumentId, f32>,
) -> Result<(), ReadError> {
    for index in indexes.iter() {
        index
            .calculate_token_score(search_params, token_scores)
            .await?;
    }

    if token_scores.is_empty() {
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

    Ok(())
}

async fn calculate_facets(
    indexes: &ReadIndexesLockGuard<'_, '_>,
    search_params: &SearchParams,
    token_scores: &HashMap<DocumentId, f32>,
) -> Result<HashMap<String, FacetResult>> {
    let mut result: HashMap<String, FacetResult> = HashMap::new();

    for index in indexes.iter() {
        index
            .calculate_facets(token_scores, &search_params.facets, &mut result)
            .await?;
    }

    let expected_facets_count = search_params.facets.len();
    if result.len() != expected_facets_count {
        // The user asked for some facets that are not present in the collection
        let present_facet_fields: Vec<&String> = result.keys().collect();
        let requested_facet_fields: Vec<&String> = search_params.facets.keys().collect();
        let missing_facet_fields: Vec<&&String> = requested_facet_fields
            .iter()
            .filter(|f| !present_facet_fields.contains(f))
            .collect();
        bail!(
            "Some fields are not present in the collection for facets: {:?}",
            missing_facet_fields
        );
    }

    Ok(result)
}

pub async fn calculate_groups(
    indexes: &ReadIndexesLockGuard<'_, '_>,
    token_scores: &HashMap<DocumentId, f32>,
    group_by_config: &GroupByConfig,
    sort_by: Option<&SortBy>,
    pin_rules: &[Consequence<DocumentId>],
) -> Result<HashMap<Vec<GroupValue>, Vec<DocumentId>>> {
    let mut results = HashMap::new();
    for index in indexes.iter() {
        index
            .calculate_groups(&group_by_config.properties, &mut results)
            .await?
    }

    // Sort documents within each group
    let sorted_results = sort_documents_in_groups(
        indexes,
        results,
        token_scores,
        sort_by,
        group_by_config.max_results,
        pin_rules,
    )
    .await?;

    Ok(sorted_results)
}

pub async fn sort_and_truncate(
    indexes: &ReadIndexesLockGuard<'_, '_>,
    token_scores: &HashMap<DocumentId, f32>,
    search_params: &SearchParams,
    pin_rules: &[Consequence<DocumentId>],
) -> Result<Vec<TokenScore>> {
    let context = SortContext::new(indexes, token_scores, search_params, pin_rules);
    context.execute().await
}
