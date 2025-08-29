use std::collections::{HashMap, HashSet};
use std::iter::Peekable;

use anyhow::{anyhow, bail, Result};
use ordered_float::NotNan;
use tracing::info;

use super::index::Index;
use crate::pin_rules::Consequence;
use crate::types::{SearchMode, SearchParams};
use crate::{
    capped_heap::CappedHeap,
    types::{DocumentId, Limit, Number, SearchOffset, SortBy, SortOrder, TokenScore},
};

/// Context structure that encapsulates all parameters needed for sorting operations
pub struct SortContext<'a> {
    pub indexes: &'a [Index],
    pub token_scores: &'a HashMap<DocumentId, f32>,
    pub search_params: &'a SearchParams,
    pub limit: Limit,
    pub offset: SearchOffset,
}

pub struct SortIterator<'s1, 's2, T: Ord + Clone> {
    iter1: Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's1>>,
    iter2: Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's2>>,
    order: SortOrder,
}

impl<'s1, 's2, T: Ord + Clone> SortIterator<'s1, 's2, T> {
    pub fn new(
        iter1: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's1>,
        iter2: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's2>,
        order: SortOrder,
    ) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            order,
        }
    }
}

impl<T: Ord + Clone> Iterator for SortIterator<'_, '_, T> {
    type Item = (T, HashSet<DocumentId>);

    fn next(&mut self) -> Option<Self::Item> {
        let el1 = self.iter1.peek();
        let el2 = self.iter2.peek();

        match (el1, el2) {
            (None, None) => None,
            (Some((_, _)), None) => self.iter1.next(),
            (None, Some((_, _))) => self.iter2.next(),
            (Some((k1, _)), Some((k2, _))) => match self.order {
                SortOrder::Ascending => {
                    if k1 < k2 {
                        self.iter1.next()
                    } else {
                        self.iter2.next()
                    }
                }
                SortOrder::Descending => {
                    if k1 > k2 {
                        self.iter1.next()
                    } else {
                        self.iter2.next()
                    }
                }
            },
        }
    }
}

/// Process sort iterator and limit results to desired count
fn process_sort_iterator(
    iter: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_>,
    desiderata: usize,
) -> Vec<(Number, HashSet<DocumentId>)> {
    // The worst scenario is that documents doesn't share the "number" value, so the hashset contains
    // only one document.
    // For a good scenario, the documents share the "number" value, so the hashset contains
    // multiple documents.
    // We use `desiderata / 2` to avoid excessive memory usage
    let mut index_results = Vec::with_capacity((desiderata / 2).min(1000));

    let mut total_item = 0_usize;
    for (n, h) in iter {
        let c = h.len();
        index_results.push((n, h));
        total_item += c;
        if total_item >= desiderata {
            break;
        }
    }

    index_results
}

/// Truncate results based on top_count, applying token scores
fn truncate<I: Iterator<Item = (Number, HashSet<DocumentId>)>>(
    token_scores: &HashMap<DocumentId, f32>,
    output: I,
    top_count: usize,
) -> Vec<TokenScore> {
    let mut res = Vec::with_capacity(top_count);
    let mut k = 0_usize;
    for (_, docs) in output {
        for doc in docs {
            let score = token_scores.get(&doc);
            if let Some(score) = score {
                if k < top_count {
                    res.push(TokenScore {
                        document_id: doc,
                        score: *score,
                    });
                }

                k += 1;

                if k > top_count {
                    break;
                }
            }
        }
    }
    res
}

/// Get top N results based on token scores when no sorting is requested
fn top_n<'s, I: Iterator<Item = (&'s DocumentId, &'s f32)>>(map: I, n: usize) -> Vec<TokenScore> {
    let mut capped_heap = CappedHeap::new(n);

    for (key, value) in map {
        let k = match NotNan::new(*value) {
            Ok(k) => k,
            Err(_) => continue,
        };
        let v = key;
        capped_heap.insert(k, v);
    }

    capped_heap
        .into_top()
        .map(|(value, key)| TokenScore {
            document_id: *key,
            score: value.into_inner(),
        })
        .collect()
}

/// Main sorting and truncation logic for documents
async fn sort_and_truncate_documents(
    relevant_indexes: Option<(&SortBy, Vec<&Index>)>,
    pins: Vec<Consequence<DocumentId>>,
    token_scores: &HashMap<DocumentId, f32>,
    limit: Limit,
    offset: SearchOffset,
) -> Result<Vec<TokenScore>> {
    let top_count = if pins.is_empty() {
        limit.0 + offset.0
    } else {
        // The user could ask 0 to N documents,
        // but a pin rules can move it after N.
        // In this case, we have to remove it and get the next item.
        // Anyway, it would fall on the next page, and we don't have it.
        // This is why we double the limit, so we have the next page ready.
        (limit.0 + offset.0) * 2
    };

    let top = if let Some((sort_by, relevant_indexes)) = relevant_indexes {
        if relevant_indexes.len() == 1 {
            // If there's only one index, use the existing optimized path
            let output = relevant_indexes[0]
                .get_sort_iterator(&sort_by.property, sort_by.order, |iter| {
                    process_sort_iterator(iter, top_count)
                })
                .await?;

            truncate(token_scores, output.into_iter(), top_count)
        } else {
            // - For the relevant_indexes, we obtain the first `top_count` items.
            //   So `all_results` contains something like
            //
            //   |sorted data from index1|
            //   |sorted data from index2|
            //   |sorted data from index3|
            //
            //   which aren't globally sorted
            //
            // - Re-sorting the merged results to maintain global order across indexes
            // - Truncating the results to the desired limit
            // Note: Each index only returns up to 'top_count' results, not all documents
            // NB: this is not optimal because we collect all the data and sort it again.
            //     we can optimize it, without collecting them and re-order.
            let mut all_results = Vec::new();

            for index in relevant_indexes {
                let index_results = index
                    .get_sort_iterator(&sort_by.property, sort_by.order, |iter| {
                        process_sort_iterator(iter, top_count)
                    })
                    .await?;

                all_results.extend(index_results);
            }

            // Sort the merged results by their numeric values to ensure global ordering
            all_results.sort_by_key(|(n, _)| *n);

            // Apply descending order if requested by reversing the sorted results
            if sort_by.order == SortOrder::Descending {
                all_results.reverse();
            }

            // Get top
            truncate(token_scores, all_results.into_iter(), top_count)
        }
    } else {
        // No sorting requested - fall back to simple top-N selection based on token scores
        top_n(token_scores.iter(), top_count)
    };

    let top = if !pins.is_empty() {
        apply_pin_rules(pins, token_scores, top)
    } else {
        top
    };

    Ok(top)
}

fn apply_pin_rules(
    pins: Vec<Consequence<DocumentId>>,
    token_scores: &HashMap<DocumentId, f32>,
    mut top: Vec<TokenScore>,
) -> Vec<TokenScore> {
    if pins.is_empty() {
        return top;
    }

    // Collect all promote items from all pin rule consequences
    let mut promote_items = Vec::new();
    for consequence in pins {
        promote_items.extend(consequence.promote);
    }

    // If no promote items, return original results
    if promote_items.is_empty() {
        return top;
    }

    let doc_ids_to_promote: HashSet<_> = promote_items.iter().map(|p| p.doc_id).collect();

    // If re want to remove all the doc_ids that are already present in `top`
    // This:
    // - avoid document duplication
    // - handle the pagination correctly
    top.retain(|ts| !doc_ids_to_promote.contains(&ts.document_id));

    // Sort promote items by position to handle them in order
    promote_items.sort_by_key(|item| item.position);

    // Make sure in the below loop there's no continuous re-allocation.
    top.reserve(promote_items.len());

    for promote_item in promote_items {
        let target_position = promote_item.position as usize;

        // Skip if target position is beyond the vector size
        if target_position >= top.len() {
            continue;
        }

        if let Some(score) = token_scores.get(&promote_item.doc_id) {
            // Insert it at the target position.
            // The other are shifted.
            top.insert(
                target_position,
                TokenScore {
                    document_id: promote_item.doc_id,
                    score: *score,
                },
            );
        }
    }

    top
}

fn extract_term_from_search_mode(search_params: &SearchParams) -> &str {
    match &search_params.mode {
        SearchMode::FullText(f) | SearchMode::Default(f) => &f.term,
        SearchMode::Hybrid(h) => &h.term,
        SearchMode::Vector(v) => &v.term,
        SearchMode::Auto(a) => &a.term,
    }
}

async fn extract_pin_rules(context: &SortContext<'_>) -> Vec<Consequence<DocumentId>> {
    let term = extract_term_from_search_mode(&context.search_params);
    let mut pins: Vec<_> = Vec::with_capacity(context.indexes.len());
    for index in context.indexes {
        let pin_rules = index.get_read_lock_on_pin_rules().await;
        pins.extend(pin_rules.apply(term));
    }
    pins
}

pub async fn sort_with_context(context: SortContext<'_>) -> Result<Vec<TokenScore>> {
    let sort_by = context.search_params.sort_by.as_ref();
    info!(
        "Sorting and truncating results: limit = {:?}, offset = {:?}, sort_by = {:?}",
        context.limit, context.offset, sort_by
    );

    // Identify relevant indexes for sorting
    let relevant_indexes_for_sorting: Option<(&SortBy, Vec<_>)> = if let Some(sort_by) = sort_by {
        let indexes: Vec<_> = context
            .indexes
            .iter()
            .filter(|index| !index.is_deleted() && index.has_filter_field(&sort_by.property))
            .collect();
        if indexes.is_empty() {
            bail!("Cannot sort by {:?}: no index has that field", sort_by.property);
        }
        Some((sort_by, indexes))
    } else {
        None
    };

    let pins = extract_pin_rules(&context).await;

    // Call the existing sort_and_truncate_documents function
    sort_and_truncate_documents(
        relevant_indexes_for_sorting,
        pins,
        context.token_scores,
        context.limit,
        context.offset,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentId, SortOrder};

    #[test]
    fn test_sort_iterator_ascending() {
        let data1 = vec![(1, 10), (3, 30), (5, 50)];
        let data2 = vec![(2, 20), (4, 40), (6, 60)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_sort_iterator_descending() {
        let data1 = vec![(6, 60), (4, 40), (2, 20)];
        let data2 = vec![(5, 50), (3, 30), (1, 10)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Descending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();

        assert_eq!(result, vec![60, 50, 40, 30, 20, 10]);
    }

    #[test]
    fn test_sort_iterator_mixed_lengths() {
        let data1 = vec![(1, 10), (3, 30)];
        let data2 = vec![(2, 20), (4, 40), (5, 50)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_sort_iterator_with_duplicates() {
        // Both iterators contain the same key and value
        let data1 = vec![(1, 10), (2, 20), (3, 30)];
        let data2 = vec![(2, 20), (3, 30), (4, 40)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();

        assert_eq!(result, vec![10, 20, 20, 30, 30, 40]);
    }

    fn make_iter(data: Vec<(u64, u64)>) -> Box<dyn Iterator<Item = (u64, HashSet<DocumentId>)>> {
        Box::new(
            data.into_iter()
                .map(|(k, v)| (k, HashSet::from([DocumentId(v)]))),
        )
    }
}
