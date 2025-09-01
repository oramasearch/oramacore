use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use ordered_float::NotNan;
use tracing::{info, warn};

use super::index::Index;
use super::{GroupValue, SortedField};
use crate::collection_manager::sides::read::sort::sort_iter::MergeSortedIterator;
use crate::pin_rules::Consequence;
use crate::types::{SearchMode, SearchParams};
use crate::{
    capped_heap::CappedHeap,
    types::{DocumentId, Limit, Number, SearchOffset, SortBy, TokenScore},
};

/// Context structure that encapsulates all parameters needed for sorting operations
pub struct SortContext<'a> {
    pub indexes: &'a [Index],
    pub token_scores: &'a HashMap<DocumentId, f32>,
    pub search_params: &'a SearchParams,
    pub limit: Limit,
    pub offset: SearchOffset,
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

async fn sort_and_truncate_documents_by_field(
    sort_by: &SortBy,
    relevant_indexes: &[&Index],
    token_scores: &HashMap<DocumentId, f32>,
    top_count: usize,
) -> Result<Vec<TokenScore>> {
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

    let v = if relevant_indexes.len() == 1 {
        // If there's only one index, use the existing optimized path
        let sorted_field = relevant_indexes[0]
            .get_sort_iterator(&sort_by.property, sort_by.order)
            .await?;
        let output = process_sort_iterator(sorted_field.iter(), top_count);

        truncate(token_scores, output.into_iter(), top_count)
    } else {
        let mut sorted_fields = Vec::with_capacity(relevant_indexes.len());
        for index in relevant_indexes {
            let sorted_field = index
                .get_sort_iterator(&sort_by.property, sort_by.order)
                .await?;
            sorted_fields.push(sorted_field);
        }

        let mut merge_sorted_iterator = MergeSortedIterator::new(sort_by.order);
        for i in &sorted_fields {
            merge_sorted_iterator.add(i.iter());
        }

        // Get top
        truncate(token_scores, merge_sorted_iterator, top_count)
    };
    Ok(v)
}

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
        sort_and_truncate_documents_by_field(sort_by, &relevant_indexes, token_scores, top_count)
            .await?
    } else {
        // No sorting requested - fall back to simple top-N selection based on token scores
        top_n(token_scores.iter(), top_count)
    };

    let top = if !pins.is_empty() {
        apply_pin_rules(&pins, token_scores, top)
    } else {
        top
    };

    Ok(top)
}

// Maximum number of promoted items to prevent excessive memory usage
const MAX_PROMOTED_ITEMS: usize = 10_000;

/// Internal shared implementation for applying pin rules with optional document filtering
fn apply_pin_rules_internal<F>(
    pins: &[Consequence<DocumentId>],
    token_scores: &HashMap<DocumentId, f32>,
    mut top: Vec<TokenScore>,
    document_filter: Option<F>,
) -> Vec<TokenScore>
where
    F: Fn(&DocumentId) -> bool,
{
    if pins.is_empty() {
        return top;
    }

    // Estimate the total number of promote items to avoid excessive memory allocation
    let estimated_promote_items: usize = pins.iter().map(|c| c.promote.len()).sum();
    if estimated_promote_items > MAX_PROMOTED_ITEMS {
        // Log warning but continue with a truncated set
        warn!(
            "Pin rules contain {} promote items, which exceeds the limit of {}. This may impact performance.",
            estimated_promote_items,
            MAX_PROMOTED_ITEMS
        );
    }

    let mut promote_items = Vec::with_capacity(estimated_promote_items.min(MAX_PROMOTED_ITEMS));
    if let Some(ref filter) = document_filter {
        for consequence in pins {
            promote_items.extend(
                consequence
                    .promote
                    .iter()
                    .filter(|item| filter(&item.doc_id))
                    .cloned(),
            );
        }
    } else {
        for consequence in pins {
            promote_items.extend(consequence.promote.iter().cloned());
        }
    }

    if promote_items.is_empty() {
        // Early return if no promote items after filtering
        return top;
    }

    let promoted_doc_ids: HashSet<_> = promote_items.iter().map(|item| item.doc_id).collect();

    // Remove documents that are being promoted from original results
    // This prevents document duplication and handles pagination correctly
    top.retain(|token_score| !promoted_doc_ids.contains(&token_score.document_id));

    // Sort promote items by position to handle them in the correct order
    promote_items.sort_by_key(|item| item.position);

    // Pre-allocate space to avoid continuous re-allocation during insertions
    top.reserve(promote_items.len());

    for promote_item in promote_items {
        let target_position = promote_item.position as usize;

        // Cap insertion position to vector bounds to prevent out-of-bounds insertion
        // If position is beyond current length, append at the end
        let insertion_position = target_position.min(top.len());

        // Retrieve the document's score from token_scores, or use 0.0 as default
        // This allows promoted documents that weren't in original search results to be included
        let score = token_scores
            .get(&promote_item.doc_id)
            .copied()
            .unwrap_or(0.0);

        // Insert the promoted document at the calculated position
        // Other documents are automatically shifted to accommodate the insertion
        top.insert(
            insertion_position,
            TokenScore {
                document_id: promote_item.doc_id,
                score,
            },
        );
    }

    top
}

fn apply_pin_rules(
    pins: &[Consequence<DocumentId>],
    token_scores: &HashMap<DocumentId, f32>,
    top: Vec<TokenScore>,
) -> Vec<TokenScore> {
    apply_pin_rules_internal(pins, token_scores, top, None::<fn(&DocumentId) -> bool>)
}

fn apply_pin_rules_to_group(
    all_document_ids: HashSet<DocumentId>,
    pins: &[Consequence<DocumentId>],
    token_scores: &HashMap<DocumentId, f32>,
    top: Vec<TokenScore>,
) -> Vec<TokenScore> {
    apply_pin_rules_internal(
        pins,
        token_scores,
        top,
        Some(|doc_id: &DocumentId| all_document_ids.contains(doc_id)),
    )
}

pub fn extract_term_from_search_mode(search_params: &SearchParams) -> &str {
    match &search_params.mode {
        SearchMode::FullText(f) | SearchMode::Default(f) => &f.term,
        SearchMode::Hybrid(h) => &h.term,
        SearchMode::Vector(v) => &v.term,
        SearchMode::Auto(a) => &a.term,
    }
}

async fn extract_pin_rules(context: &SortContext<'_>) -> Vec<Consequence<DocumentId>> {
    let term = extract_term_from_search_mode(context.search_params);
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
            bail!(
                "Cannot sort by {:?}: no index has that field",
                sort_by.property
            );
        }
        Some((sort_by, indexes))
    } else {
        None
    };

    let pins = extract_pin_rules(&context).await;

    sort_and_truncate_documents(
        relevant_indexes_for_sorting,
        pins,
        context.token_scores,
        context.limit,
        context.offset,
    )
    .await
}

pub async fn sort_documents_in_groups(
    groups: HashMap<Vec<GroupValue>, HashSet<DocumentId>>,
    token_scores: &HashMap<DocumentId, f32>,
    sort_by: Option<&SortBy>,
    relevant_indexes: &[&Index],
    max_results: usize,
    pin_rules: &[Consequence<DocumentId>],
) -> Result<HashMap<Vec<GroupValue>, Vec<DocumentId>>> {
    let mut sorted_groups = HashMap::with_capacity(groups.len());

    for (group_key, doc_set) in groups {
        let docs = if let Some(sort_by) = sort_by {
            top_n_documents_in_group_by_field(
                doc_set,
                sort_by,
                relevant_indexes,
                token_scores,
                max_results,
                pin_rules,
            )
            .await?
        } else {
            top_n_documents_in_group_by_score(doc_set, token_scores, max_results, pin_rules)
        };

        sorted_groups.insert(group_key, docs);
    }

    Ok(sorted_groups)
}

fn top_n_documents_in_group_by_score(
    docs: HashSet<DocumentId>,
    token_scores: &HashMap<DocumentId, f32>,
    max_results: usize,
    pin_rules: &[Consequence<DocumentId>],
) -> Vec<DocumentId> {
    // Increase limit to account for potential pin rule insertions
    let expanded_limit = if pin_rules.is_empty() {
        max_results
    } else {
        max_results * 2 // Double the limit like we do in the main sort function
    };

    let mut capped_heap = CappedHeap::new(expanded_limit);

    for doc_id in &docs {
        if let Some(score) = token_scores.get(doc_id) {
            if let Ok(k) = NotNan::new(*score) {
                capped_heap.insert(k, doc_id);
            }
        }
    }

    let mut result: Vec<TokenScore> = capped_heap
        .into_top()
        .map(|(score, doc_id)| TokenScore {
            document_id: *doc_id,
            score: score.into_inner(),
        })
        .collect();

    // Apply pin rules to this group's documents
    if !pin_rules.is_empty() {
        result = apply_pin_rules_to_group(docs, pin_rules, token_scores, result);
    }

    // Truncate to max_results and extract document IDs
    result
        .into_iter()
        .take(max_results)
        .map(|ts| ts.document_id)
        .collect()
}

async fn top_n_documents_in_group_by_field(
    docs: HashSet<DocumentId>,
    sort_by: &SortBy,
    relevant_indexes: &[&Index],
    token_scores: &HashMap<DocumentId, f32>,
    max_results: usize,
    pin_rules: &[Consequence<DocumentId>],
) -> Result<Vec<DocumentId>> {
    // Increase limit to account for potential pin rule insertions
    let expanded_limit = if pin_rules.is_empty() {
        max_results
    } else {
        max_results * 2 // Double the limit like we do in the main sort function
    };

    let v: Vec<_> = if relevant_indexes.len() == 1 {
        let output = relevant_indexes[0]
            .get_sort_iterator(&sort_by.property, sort_by.order)
            .await?;

        output
            .iter()
            .flat_map(|(_, h)| h.into_iter())
            .filter(|doc_id| docs.contains(doc_id) && token_scores.contains_key(doc_id))
            .take(expanded_limit)
            .collect()
    } else {
        let mut v: Vec<_> = Vec::with_capacity(relevant_indexes.len());
        for index in relevant_indexes {
            let index_results: SortedField = index
                .get_sort_iterator(&sort_by.property, sort_by.order)
                .await?;
            v.push(index_results);
        }

        let mut merge_sorted_iterator = MergeSortedIterator::new(sort_by.order);
        for i in &v {
            merge_sorted_iterator.add(i.iter());
        }

        merge_sorted_iterator
            .flat_map(|(_, h)| h.into_iter())
            .filter(|doc_id| docs.contains(doc_id) && token_scores.contains_key(doc_id))
            .take(expanded_limit)
            .collect()
    };

    // Apply pin rules if any exist
    if !pin_rules.is_empty() {
        // Convert document IDs to TokenScore for pin rule application
        let mut token_scores_vec: Vec<TokenScore> = v
            .into_iter()
            .map(|doc_id| TokenScore {
                document_id: doc_id,
                score: *token_scores.get(&doc_id).unwrap_or(&0.0),
            })
            .collect();

        // Apply pin rules
        token_scores_vec =
            apply_pin_rules_to_group(docs, pin_rules, token_scores, token_scores_vec);

        // Extract document IDs and truncate to max_results
        Ok(token_scores_vec
            .into_iter()
            .take(max_results)
            .map(|ts| ts.document_id)
            .collect())
    } else {
        Ok(v.into_iter().take(max_results).collect())
    }
}

mod sort_iter {
    use crate::types::{DocumentId, Number, SortOrder};
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::iter::Peekable;

    pub trait OrderedKey: Ord + Eq + Clone + Debug {}
    impl OrderedKey for Number {}

    pub struct MergeSortedIterator<'iter, T: OrderedKey> {
        iters: Vec<Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 'iter>>>,
        order: SortOrder,
    }

    impl<'iter, T: OrderedKey> MergeSortedIterator<'iter, T> {
        pub fn new(order: SortOrder) -> Self {
            Self {
                iters: vec![],
                order,
            }
        }

        pub fn add(&mut self, iter: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 'iter>) {
            self.iters.push(iter.peekable());
        }
    }

    impl<T: OrderedKey> Iterator for MergeSortedIterator<'_, T> {
        type Item = (T, HashSet<DocumentId>);

        fn next(&mut self) -> Option<Self::Item> {
            fn cmp_pair<T: Ord>(
                (_, key1): &(usize, T),
                (_, key2): &(usize, T),
            ) -> std::cmp::Ordering {
                key1.cmp(key2)
            }

            let iter = self
                .iters
                .iter_mut()
                .enumerate()
                .filter_map(|(index, iter)| iter.peek().map(|(n, _)| (index, n.clone())));
            let index_key_pair = match self.order {
                SortOrder::Ascending => iter.min_by(cmp_pair),
                SortOrder::Descending => iter.max_by(cmp_pair),
            };

            index_key_pair.and_then(|(index, _)| self.iters[index].next())
        }
    }

    #[cfg(test)]
    mod sort_iter_test {
        use super::*;

        impl OrderedKey for u8 {}
        impl OrderedKey for i16 {}

        #[test]
        fn test_empty() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);
            assert!(merged.next().is_none());
        }

        #[test]
        fn test_one() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);
            merged.add(Box::new(
                vec![
                    (
                        0,
                        HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)]),
                    ),
                    (
                        2,
                        HashSet::from([DocumentId(6), DocumentId(8), DocumentId(9)]),
                    ),
                ]
                .into_iter(),
            ));
            let collected: Vec<_> = merged.collect();
            assert_eq!(
                collected,
                vec![
                    (
                        0,
                        HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)])
                    ),
                    (
                        2,
                        HashSet::from([DocumentId(6), DocumentId(8), DocumentId(9)])
                    )
                ]
            );
        }

        #[test]
        fn test_multiple_iterators_ascending() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: [1, 5, 9]
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(10)])),
                    (5, HashSet::from([DocumentId(50)])),
                    (9, HashSet::from([DocumentId(90)])),
                ]
                .into_iter(),
            ));

            // Iterator 2: [2, 6, 10]
            merged.add(Box::new(
                vec![
                    (2, HashSet::from([DocumentId(20)])),
                    (6, HashSet::from([DocumentId(60)])),
                    (10, HashSet::from([DocumentId(100)])),
                ]
                .into_iter(),
            ));

            // Iterator 3: [3, 4, 7, 8]
            merged.add(Box::new(
                vec![
                    (3, HashSet::from([DocumentId(30)])),
                    (4, HashSet::from([DocumentId(40)])),
                    (7, HashSet::from([DocumentId(70)])),
                    (8, HashSet::from([DocumentId(80)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

            // Verify document IDs are preserved correctly
            assert_eq!(collected[0].1, HashSet::from([DocumentId(10)]));
            assert_eq!(collected[1].1, HashSet::from([DocumentId(20)]));
            assert_eq!(collected[9].1, HashSet::from([DocumentId(100)]));
        }

        #[test]
        fn test_multiple_iterators_descending() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Descending);

            // Iterator 1: [9, 5, 1] (descending)
            merged.add(Box::new(
                vec![
                    (9, HashSet::from([DocumentId(90)])),
                    (5, HashSet::from([DocumentId(50)])),
                    (1, HashSet::from([DocumentId(10)])),
                ]
                .into_iter(),
            ));

            // Iterator 2: [10, 6, 2] (descending)
            merged.add(Box::new(
                vec![
                    (10, HashSet::from([DocumentId(100)])),
                    (6, HashSet::from([DocumentId(60)])),
                    (2, HashSet::from([DocumentId(20)])),
                ]
                .into_iter(),
            ));

            // Iterator 3: [8, 7, 4, 3] (descending)
            merged.add(Box::new(
                vec![
                    (8, HashSet::from([DocumentId(80)])),
                    (7, HashSet::from([DocumentId(70)])),
                    (4, HashSet::from([DocumentId(40)])),
                    (3, HashSet::from([DocumentId(30)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

            // Verify document IDs are preserved correctly
            assert_eq!(collected[0].1, HashSet::from([DocumentId(100)]));
            assert_eq!(collected[1].1, HashSet::from([DocumentId(90)]));
            assert_eq!(collected[9].1, HashSet::from([DocumentId(10)]));
        }

        #[test]
        fn test_duplicate_keys_across_iterators() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: [1, 3, 5]
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(10)])),
                    (3, HashSet::from([DocumentId(30)])),
                    (5, HashSet::from([DocumentId(50)])),
                ]
                .into_iter(),
            ));

            // Iterator 2: [1, 3, 4] - has duplicate keys 1 and 3
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(11)])),
                    (3, HashSet::from([DocumentId(31)])),
                    (4, HashSet::from([DocumentId(40)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            // Should preserve all items, including duplicates
            assert_eq!(keys, vec![1, 1, 3, 3, 4, 5]);

            // Verify both document sets for key 1 are preserved
            assert_eq!(collected[0].1, HashSet::from([DocumentId(10)]));
            assert_eq!(collected[1].1, HashSet::from([DocumentId(11)]));

            // Verify both document sets for key 3 are preserved
            assert_eq!(collected[2].1, HashSet::from([DocumentId(30)]));
            assert_eq!(collected[3].1, HashSet::from([DocumentId(31)]));
        }

        #[test]
        fn test_duplicate_keys_same_iterator() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Single iterator with duplicate keys
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(10)])),
                    (1, HashSet::from([DocumentId(11)])),
                    (2, HashSet::from([DocumentId(20)])),
                    (2, HashSet::from([DocumentId(21)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 1, 2, 2]);
            assert_eq!(collected[0].1, HashSet::from([DocumentId(10)]));
            assert_eq!(collected[1].1, HashSet::from([DocumentId(11)]));
            assert_eq!(collected[2].1, HashSet::from([DocumentId(20)]));
            assert_eq!(collected[3].1, HashSet::from([DocumentId(21)]));
        }

        #[test]
        fn test_single_empty_iterator() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Add a single empty iterator
            merged.add(Box::new(std::iter::empty()));

            let collected: Vec<_> = merged.collect();
            assert!(collected.is_empty());
        }

        #[test]
        fn test_mix_empty_and_non_empty_iterators() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Empty iterator
            merged.add(Box::new(std::iter::empty()));

            // Non-empty iterator
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(10)])),
                    (3, HashSet::from([DocumentId(30)])),
                ]
                .into_iter(),
            ));

            // Another empty iterator
            merged.add(Box::new(std::iter::empty()));

            // Another non-empty iterator
            merged.add(Box::new(
                vec![
                    (2, HashSet::from([DocumentId(20)])),
                    (4, HashSet::from([DocumentId(40)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            // Should only get results from non-empty iterators
            assert_eq!(keys, vec![1, 2, 3, 4]);
            assert_eq!(collected[0].1, HashSet::from([DocumentId(10)]));
            assert_eq!(collected[1].1, HashSet::from([DocumentId(20)]));
            assert_eq!(collected[2].1, HashSet::from([DocumentId(30)]));
            assert_eq!(collected[3].1, HashSet::from([DocumentId(40)]));
        }

        #[test]
        fn test_all_iterators_exhausted_simultaneously() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: [1]
            merged.add(Box::new(
                vec![(1, HashSet::from([DocumentId(10)]))].into_iter(),
            ));

            // Iterator 2: [2]
            merged.add(Box::new(
                vec![(2, HashSet::from([DocumentId(20)]))].into_iter(),
            ));

            // Iterator 3: [3]
            merged.add(Box::new(
                vec![(3, HashSet::from([DocumentId(30)]))].into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3]);
            assert_eq!(collected.len(), 3);
        }

        #[test]
        fn test_iterators_different_lengths() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Short iterator: [1, 2]
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(10)])),
                    (2, HashSet::from([DocumentId(20)])),
                ]
                .into_iter(),
            ));

            // Long iterator: [3, 4, 5, 6, 7, 8]
            merged.add(Box::new(
                vec![
                    (3, HashSet::from([DocumentId(30)])),
                    (4, HashSet::from([DocumentId(40)])),
                    (5, HashSet::from([DocumentId(50)])),
                    (6, HashSet::from([DocumentId(60)])),
                    (7, HashSet::from([DocumentId(70)])),
                    (8, HashSet::from([DocumentId(80)])),
                ]
                .into_iter(),
            ));

            // Medium iterator: [9, 10, 11]
            merged.add(Box::new(
                vec![
                    (9, HashSet::from([DocumentId(90)])),
                    (10, HashSet::from([DocumentId(100)])),
                    (11, HashSet::from([DocumentId(110)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            assert_eq!(collected.len(), 11);
        }

        #[test]
        fn test_one_iterator_much_longer() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Descending);

            // Very short iterator: [100]
            merged.add(Box::new(
                vec![(100, HashSet::from([DocumentId(1000)]))].into_iter(),
            ));

            // Much longer iterator: [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
            merged.add(Box::new(
                vec![
                    (99, HashSet::from([DocumentId(990)])),
                    (98, HashSet::from([DocumentId(980)])),
                    (97, HashSet::from([DocumentId(970)])),
                    (96, HashSet::from([DocumentId(960)])),
                    (95, HashSet::from([DocumentId(950)])),
                    (94, HashSet::from([DocumentId(940)])),
                    (93, HashSet::from([DocumentId(930)])),
                    (92, HashSet::from([DocumentId(920)])),
                    (91, HashSet::from([DocumentId(910)])),
                    (90, HashSet::from([DocumentId(900)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]);

            // Verify first item comes from the shorter iterator
            assert_eq!(collected[0].1, HashSet::from([DocumentId(1000)]));

            // Verify remaining items come from the longer iterator
            assert_eq!(collected[1].1, HashSet::from([DocumentId(990)]));
            assert_eq!(collected[10].1, HashSet::from([DocumentId(900)]));
        }

        #[test]
        fn test_document_id_set_merging() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator with multiple documents per key
            merged.add(Box::new(
                vec![
                    (
                        1,
                        HashSet::from([DocumentId(10), DocumentId(11), DocumentId(12)]),
                    ),
                    (3, HashSet::from([DocumentId(30), DocumentId(31)])),
                ]
                .into_iter(),
            ));

            // Iterator with single documents per key
            merged.add(Box::new(
                vec![
                    (2, HashSet::from([DocumentId(20)])),
                    (4, HashSet::from([DocumentId(40)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4]);

            // Verify multi-document sets are preserved
            assert_eq!(
                collected[0].1,
                HashSet::from([DocumentId(10), DocumentId(11), DocumentId(12)])
            );
            assert_eq!(collected[1].1, HashSet::from([DocumentId(20)]));
            assert_eq!(
                collected[2].1,
                HashSet::from([DocumentId(30), DocumentId(31)])
            );
            assert_eq!(collected[3].1, HashSet::from([DocumentId(40)]));
        }

        #[test]
        fn test_overlapping_document_sets() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: contains DocumentId(100) in multiple sets
            merged.add(Box::new(
                vec![
                    (1, HashSet::from([DocumentId(100), DocumentId(101)])),
                    (3, HashSet::from([DocumentId(100), DocumentId(103)])),
                ]
                .into_iter(),
            ));

            // Iterator 2: also contains DocumentId(100)
            merged.add(Box::new(
                vec![
                    (2, HashSet::from([DocumentId(100), DocumentId(102)])),
                    (4, HashSet::from([DocumentId(104)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4]);

            // Each set should maintain its own DocumentId combinations
            // DocumentId(100) appears in multiple sets, which is expected behavior
            assert_eq!(
                collected[0].1,
                HashSet::from([DocumentId(100), DocumentId(101)])
            );
            assert_eq!(
                collected[1].1,
                HashSet::from([DocumentId(100), DocumentId(102)])
            );
            assert_eq!(
                collected[2].1,
                HashSet::from([DocumentId(100), DocumentId(103)])
            );
            assert_eq!(collected[3].1, HashSet::from([DocumentId(104)]));
        }

        #[test]
        fn test_min_max_value_usage() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator with min and max values
            merged.add(Box::new(
                vec![
                    (u8::MIN, HashSet::from([DocumentId(0)])),
                    (1, HashSet::from([DocumentId(1)])),
                    (u8::MAX, HashSet::from([DocumentId(255)])),
                ]
                .into_iter(),
            ));

            // Iterator with intermediate values
            merged.add(Box::new(
                vec![
                    (2, HashSet::from([DocumentId(2)])),
                    (127, HashSet::from([DocumentId(127)])),
                    (254, HashSet::from([DocumentId(254)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![0, 1, 2, 127, 254, 255]);
            assert_eq!(collected[0].1, HashSet::from([DocumentId(0)]));
            assert_eq!(collected[5].1, HashSet::from([DocumentId(255)]));
        }

        #[test]
        fn test_custom_ordered_key_type() {
            // Test with a different OrderedKey type (using i16 to have different bounds)

            let mut merged = MergeSortedIterator::<i16>::new(SortOrder::Descending);

            // Iterator with negative values
            merged.add(Box::new(
                vec![
                    (i16::MAX, HashSet::from([DocumentId(32767)])),
                    (0, HashSet::from([DocumentId(0)])),
                    (i16::MIN, HashSet::from([DocumentId(32768)])),
                ]
                .into_iter(),
            ));

            // Iterator with other values
            merged.add(Box::new(
                vec![
                    (1000, HashSet::from([DocumentId(1000)])),
                    (-1000, HashSet::from([DocumentId(2000)])),
                ]
                .into_iter(),
            ));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<i16> = collected.iter().map(|(k, _)| *k).collect();

            // Should be in descending order
            assert_eq!(keys, vec![32767, 1000, 0, -1000, -32768]);
            assert_eq!(collected[0].1, HashSet::from([DocumentId(32767)]));
            assert_eq!(collected[4].1, HashSet::from([DocumentId(32768)]));
        }
    }
}
