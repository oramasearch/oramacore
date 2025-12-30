use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use anyhow::Result;
use oramacore_lib::data_structures::capped_heap::CappedHeap;
use ordered_float::NotNan;
use tracing::warn;

use super::{GroupValue, IndexSortContext};
use crate::collection_manager::sides::read::sort::sort_iter::MergeSortedIterator;
use crate::collection_manager::sides::read::{IndexSearchStore, ReadError};
use crate::types::{DocumentId, Number, SortBy, TokenScore};
use crate::types::{Limit, SearchOffset};
use oramacore_lib::pin_rules::Consequence;

pub fn sort_token_scores<'index>(
    stores: &[IndexSearchStore<'index>],
    token_scores: &HashMap<DocumentId, f32>,
    limit: Limit,
    offset: SearchOffset,
    sort_by: Option<&SortBy>,
    pin_rule_consequences: &[Consequence<DocumentId>],
) -> Result<Vec<TokenScore>, ReadError> {
    let top_count = if pin_rule_consequences.is_empty() {
        limit.0 + offset.0
    } else {
        // The user could ask 0 to N documents,
        // but a pin rules can move it after N.
        // In this case, we have to remove it and get the next item.
        // Anyway, it would fall on the next page, and we don't have it.
        // This is why we double the limit, so we have the next page ready.
        (limit.0 + offset.0) * 2
    };

    let top = if let Some(sort_by) = sort_by {
        sort_token_scores_by_field(stores, token_scores, top_count, sort_by)?
    } else {
        // No sorting requested - fall back to simple top-N selection based on token scores
        top_n(token_scores.iter(), top_count)
    };

    let top = apply_pin_rules(pin_rule_consequences, token_scores, top);

    Ok(top)
}

fn sort_token_scores_by_field<'index>(
    stores: &[IndexSearchStore<'index>],
    token_scores: &HashMap<DocumentId, f32>,
    top_count: usize,
    sort_by: &SortBy,
) -> Result<Vec<TokenScore>, ReadError> {
    let result = if stores.len() == 1 {
        // If there's only one index, use the existing optimized path
        let store = &stores[0];

        let index_sort_context = IndexSortContext::new(
            store.path_to_field_id_map,
            &store.uncommitted_fields,
            &store.committed_fields,
            &sort_by.property,
            sort_by.order,
        );

        let output = process_sort_iterator(index_sort_context.execute()?, top_count);

        truncate(token_scores, output.into_iter(), top_count)
    } else {
        let mut v: Vec<IndexSortContext<'_>> = Vec::with_capacity(stores.len());
        for store in stores.iter() {
            let index_sort_context = IndexSortContext::new(
                store.path_to_field_id_map,
                &store.uncommitted_fields,
                &store.committed_fields,
                &sort_by.property,
                sort_by.order,
            );
            v.push(index_sort_context);
        }

        let mut merge_sorted_iterator = MergeSortedIterator::new(sort_by.order);
        for i in &v {
            merge_sorted_iterator.add(i.execute()?);
        }

        let data = truncate(token_scores, merge_sorted_iterator, top_count);

        drop(v);

        data
    };

    Ok(result)
}

/// Process sort iterator to collect results up to the desired count
///
/// Returns a vector of (Number, Cow<HashSet<DocumentId>>) tuples where
/// the sum of the sizes of the HashSets is at least `desiderata`.
/// Uses Cow to avoid cloning HashSets when they are borrowed.
fn process_sort_iterator<'a>(
    iter: Box<dyn Iterator<Item = (Number, Cow<'a, HashSet<DocumentId>>)> + 'a>,
    desiderata: usize,
) -> Vec<(Number, Cow<'a, HashSet<DocumentId>>)> {
    // The worst scenario is that documents doesn't share the "number" value,
    // so each iterator item contains only one document.
    // For a good scenario, the documents share the "number" value,
    // so the hashset contains multiple documents.
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

/// Sort groups of documents based on token scores and optional field sorting
pub fn sort_groups<'index>(
    stores: &[IndexSearchStore<'index>],
    token_scores: &HashMap<DocumentId, f32>,
    groups: HashMap<Vec<GroupValue>, HashSet<DocumentId>>,
    group_limit: usize,
    sort_by: Option<&SortBy>,
    pin_rule_consequences: &[Consequence<DocumentId>],
) -> Result<HashMap<Vec<GroupValue>, Vec<TokenScore>>, ReadError> {
    // Same reason as `sort_token_scores`. See there for more details.
    let top_count = if pin_rule_consequences.is_empty() {
        group_limit
    } else {
        group_limit * 2
    };

    let mut sorted_groups = HashMap::with_capacity(groups.len());

    for (group_key, docs) in groups {
        let result: Vec<DocumentId> = if let Some(sort_by) = sort_by {
            if stores.len() == 1 {
                let store = &stores[0];

                let index_sort_context = IndexSortContext::new(
                    store.path_to_field_id_map,
                    &store.uncommitted_fields,
                    &store.committed_fields,
                    &sort_by.property,
                    sort_by.order,
                );

                let sorted_iter = index_sort_context.execute()?;

                sorted_iter
                    .flat_map(|(_, h)| h.into_owned().into_iter())
                    .filter(|doc_id| docs.contains(doc_id) && token_scores.contains_key(doc_id))
                    .take(top_count)
                    .collect()
            } else {
                let mut v: Vec<IndexSortContext<'_>> = Vec::with_capacity(stores.len());
                for store in stores.iter() {
                    let index_sort_context = IndexSortContext::new(
                        store.path_to_field_id_map,
                        &store.uncommitted_fields,
                        &store.committed_fields,
                        &sort_by.property,
                        sort_by.order,
                    );
                    v.push(index_sort_context);
                }

                let mut merge_sorted_iterator = MergeSortedIterator::new(sort_by.order);
                for i in &v {
                    merge_sorted_iterator.add(i.execute()?);
                }

                let data: Vec<DocumentId> = merge_sorted_iterator
                    .flat_map(|(_, h)| h.into_owned().into_iter())
                    .filter(|doc_id| docs.contains(doc_id) && token_scores.contains_key(doc_id))
                    .take(top_count)
                    .collect();

                drop(v);

                data
            }
        } else {
            let mut capped_heap = CappedHeap::new(top_count);

            for doc_id in &docs {
                if let Some(score) = token_scores.get(doc_id) {
                    if let Ok(k) = NotNan::new(*score) {
                        capped_heap.insert(k, doc_id);
                    }
                }
            }

            capped_heap.into_top().map(|(_, doc_id)| *doc_id).collect()
        };

        let docs = apply_pin_rules_to_group(
            docs,
            pin_rule_consequences,
            token_scores,
            result
                .into_iter()
                .map(|doc_id| TokenScore {
                    document_id: doc_id,
                    score: *token_scores.get(&doc_id).unwrap_or(&0.0),
                })
                .collect(),
        );

        sorted_groups.insert(group_key, docs);
    }

    Ok(sorted_groups)
}

/// Truncate results based on top_count, applying token scores.
/// Works with Cow<HashSet> to avoid unnecessary cloning.
fn truncate<'a, I: Iterator<Item = (Number, Cow<'a, HashSet<DocumentId>>)>>(
    token_scores: &HashMap<DocumentId, f32>,
    output: I,
    top_count: usize,
) -> Vec<TokenScore> {
    let mut res = Vec::with_capacity(top_count);
    'outer: for (_, docs) in output {
        for doc in docs.iter() {
            // Early exit before processing when we have enough results
            if res.len() >= top_count {
                break 'outer;
            }
            if let Some(&score) = token_scores.get(doc) {
                res.push(TokenScore {
                    document_id: *doc,
                    score,
                });
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

mod sort_iter {
    use crate::types::{DocumentId, Number, SortOrder};
    use std::borrow::Cow;
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::iter::Peekable;

    pub trait OrderedKey: Ord + Eq + Clone + Debug {}
    impl OrderedKey for Number {}

    /// Iterator that merges multiple sorted iterators into a single sorted stream.
    /// Uses Cow to avoid cloning HashSets when possible.
    pub struct MergeSortedIterator<'iter, T: OrderedKey> {
        iters:
            Vec<Peekable<Box<dyn Iterator<Item = (T, Cow<'iter, HashSet<DocumentId>>)> + 'iter>>>,
        order: SortOrder,
    }

    impl<'iter, T: OrderedKey> MergeSortedIterator<'iter, T> {
        pub fn new(order: SortOrder) -> Self {
            Self {
                iters: vec![],
                order,
            }
        }

        pub fn add(
            &mut self,
            iter: Box<dyn Iterator<Item = (T, Cow<'iter, HashSet<DocumentId>>)> + 'iter>,
        ) {
            self.iters.push(iter.peekable());
        }
    }

    impl<'iter, T: OrderedKey> Iterator for MergeSortedIterator<'iter, T> {
        type Item = (T, Cow<'iter, HashSet<DocumentId>>);

        fn next(&mut self) -> Option<Self::Item> {
            // Find the index of the iterator with the best (min/max) key.
            // We store the best key clone to avoid double mutable borrow of self.iters.
            let mut best: Option<(usize, T)> = None;

            for (index, iter) in self.iters.iter_mut().enumerate() {
                if let Some((key, _)) = iter.peek() {
                    let is_better = match &best {
                        None => true,
                        Some((_, best_key)) => match self.order {
                            SortOrder::Ascending => key < best_key,
                            SortOrder::Descending => key > best_key,
                        },
                    };
                    if is_better {
                        best = Some((index, key.clone()));
                    }
                }
            }

            best.and_then(|(index, _)| self.iters[index].next())
        }
    }

    #[cfg(test)]
    mod sort_iter_test {
        use super::*;

        impl OrderedKey for u8 {}
        impl OrderedKey for i16 {}

        /// Helper to create an iterator of (T, Cow<HashSet>) from a vec of (T, HashSet)
        fn cow_iter<T: OrderedKey + 'static>(
            data: Vec<(T, HashSet<DocumentId>)>,
        ) -> Box<dyn Iterator<Item = (T, Cow<'static, HashSet<DocumentId>>)>> {
            Box::new(data.into_iter().map(|(k, v)| (k, Cow::Owned(v))))
        }

        /// Helper to compare Cow<HashSet> with HashSet
        fn cow_eq(cow: &Cow<'_, HashSet<DocumentId>>, expected: HashSet<DocumentId>) -> bool {
            cow.as_ref() == &expected
        }

        #[test]
        fn test_empty() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);
            assert!(merged.next().is_none());
        }

        #[test]
        fn test_one() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);
            merged.add(cow_iter(vec![
                (
                    0,
                    HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)]),
                ),
                (
                    2,
                    HashSet::from([DocumentId(6), DocumentId(8), DocumentId(9)]),
                ),
            ]));
            let collected: Vec<_> = merged.collect();
            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].0, 0);
            assert!(cow_eq(
                &collected[0].1,
                HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)])
            ));
            assert_eq!(collected[1].0, 2);
            assert!(cow_eq(
                &collected[1].1,
                HashSet::from([DocumentId(6), DocumentId(8), DocumentId(9)])
            ));
        }

        #[test]
        fn test_multiple_iterators_ascending() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: [1, 5, 9]
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(10)])),
                (5, HashSet::from([DocumentId(50)])),
                (9, HashSet::from([DocumentId(90)])),
            ]));

            // Iterator 2: [2, 6, 10]
            merged.add(cow_iter(vec![
                (2, HashSet::from([DocumentId(20)])),
                (6, HashSet::from([DocumentId(60)])),
                (10, HashSet::from([DocumentId(100)])),
            ]));

            // Iterator 3: [3, 4, 7, 8]
            merged.add(cow_iter(vec![
                (3, HashSet::from([DocumentId(30)])),
                (4, HashSet::from([DocumentId(40)])),
                (7, HashSet::from([DocumentId(70)])),
                (8, HashSet::from([DocumentId(80)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

            // Verify document IDs are preserved correctly
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(20)])));
            assert!(cow_eq(&collected[9].1, HashSet::from([DocumentId(100)])));
        }

        #[test]
        fn test_multiple_iterators_descending() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Descending);

            // Iterator 1: [9, 5, 1] (descending)
            merged.add(cow_iter(vec![
                (9, HashSet::from([DocumentId(90)])),
                (5, HashSet::from([DocumentId(50)])),
                (1, HashSet::from([DocumentId(10)])),
            ]));

            // Iterator 2: [10, 6, 2] (descending)
            merged.add(cow_iter(vec![
                (10, HashSet::from([DocumentId(100)])),
                (6, HashSet::from([DocumentId(60)])),
                (2, HashSet::from([DocumentId(20)])),
            ]));

            // Iterator 3: [8, 7, 4, 3] (descending)
            merged.add(cow_iter(vec![
                (8, HashSet::from([DocumentId(80)])),
                (7, HashSet::from([DocumentId(70)])),
                (4, HashSet::from([DocumentId(40)])),
                (3, HashSet::from([DocumentId(30)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

            // Verify document IDs are preserved correctly
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(100)])));
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(90)])));
            assert!(cow_eq(&collected[9].1, HashSet::from([DocumentId(10)])));
        }

        #[test]
        fn test_duplicate_keys_across_iterators() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: [1, 3, 5]
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(10)])),
                (3, HashSet::from([DocumentId(30)])),
                (5, HashSet::from([DocumentId(50)])),
            ]));

            // Iterator 2: [1, 3, 4] - has duplicate keys 1 and 3
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(11)])),
                (3, HashSet::from([DocumentId(31)])),
                (4, HashSet::from([DocumentId(40)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            // Should preserve all items, including duplicates
            assert_eq!(keys, vec![1, 1, 3, 3, 4, 5]);

            // Verify both document sets for key 1 are preserved
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(11)])));

            // Verify both document sets for key 3 are preserved
            assert!(cow_eq(&collected[2].1, HashSet::from([DocumentId(30)])));
            assert!(cow_eq(&collected[3].1, HashSet::from([DocumentId(31)])));
        }

        #[test]
        fn test_duplicate_keys_same_iterator() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Single iterator with duplicate keys
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(10)])),
                (1, HashSet::from([DocumentId(11)])),
                (2, HashSet::from([DocumentId(20)])),
                (2, HashSet::from([DocumentId(21)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 1, 2, 2]);
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(11)])));
            assert!(cow_eq(&collected[2].1, HashSet::from([DocumentId(20)])));
            assert!(cow_eq(&collected[3].1, HashSet::from([DocumentId(21)])));
        }

        #[test]
        fn test_single_empty_iterator() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Add a single empty iterator
            merged.add(cow_iter(vec![]));

            let collected: Vec<_> = merged.collect();
            assert!(collected.is_empty());
        }

        #[test]
        fn test_mix_empty_and_non_empty_iterators() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Empty iterator
            merged.add(cow_iter(vec![]));

            // Non-empty iterator
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(10)])),
                (3, HashSet::from([DocumentId(30)])),
            ]));

            // Another empty iterator
            merged.add(cow_iter(vec![]));

            // Another non-empty iterator
            merged.add(cow_iter(vec![
                (2, HashSet::from([DocumentId(20)])),
                (4, HashSet::from([DocumentId(40)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            // Should only get results from non-empty iterators
            assert_eq!(keys, vec![1, 2, 3, 4]);
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(20)])));
            assert!(cow_eq(&collected[2].1, HashSet::from([DocumentId(30)])));
            assert!(cow_eq(&collected[3].1, HashSet::from([DocumentId(40)])));
        }

        #[test]
        fn test_all_iterators_exhausted_simultaneously() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: [1]
            merged.add(cow_iter(vec![(1, HashSet::from([DocumentId(10)]))]));

            // Iterator 2: [2]
            merged.add(cow_iter(vec![(2, HashSet::from([DocumentId(20)]))]));

            // Iterator 3: [3]
            merged.add(cow_iter(vec![(3, HashSet::from([DocumentId(30)]))]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3]);
            assert_eq!(collected.len(), 3);
        }

        #[test]
        fn test_iterators_different_lengths() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Short iterator: [1, 2]
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(10)])),
                (2, HashSet::from([DocumentId(20)])),
            ]));

            // Long iterator: [3, 4, 5, 6, 7, 8]
            merged.add(cow_iter(vec![
                (3, HashSet::from([DocumentId(30)])),
                (4, HashSet::from([DocumentId(40)])),
                (5, HashSet::from([DocumentId(50)])),
                (6, HashSet::from([DocumentId(60)])),
                (7, HashSet::from([DocumentId(70)])),
                (8, HashSet::from([DocumentId(80)])),
            ]));

            // Medium iterator: [9, 10, 11]
            merged.add(cow_iter(vec![
                (9, HashSet::from([DocumentId(90)])),
                (10, HashSet::from([DocumentId(100)])),
                (11, HashSet::from([DocumentId(110)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            assert_eq!(collected.len(), 11);
        }

        #[test]
        fn test_one_iterator_much_longer() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Descending);

            // Very short iterator: [100]
            merged.add(cow_iter(vec![(100, HashSet::from([DocumentId(1000)]))]));

            // Much longer iterator: [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
            merged.add(cow_iter(vec![
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
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]);

            // Verify first item comes from the shorter iterator
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(1000)])));

            // Verify remaining items come from the longer iterator
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(990)])));
            assert!(cow_eq(&collected[10].1, HashSet::from([DocumentId(900)])));
        }

        #[test]
        fn test_document_id_set_merging() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator with multiple documents per key
            merged.add(cow_iter(vec![
                (
                    1,
                    HashSet::from([DocumentId(10), DocumentId(11), DocumentId(12)]),
                ),
                (3, HashSet::from([DocumentId(30), DocumentId(31)])),
            ]));

            // Iterator with single documents per key
            merged.add(cow_iter(vec![
                (2, HashSet::from([DocumentId(20)])),
                (4, HashSet::from([DocumentId(40)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4]);

            // Verify multi-document sets are preserved
            assert!(cow_eq(
                &collected[0].1,
                HashSet::from([DocumentId(10), DocumentId(11), DocumentId(12)])
            ));
            assert!(cow_eq(&collected[1].1, HashSet::from([DocumentId(20)])));
            assert!(cow_eq(
                &collected[2].1,
                HashSet::from([DocumentId(30), DocumentId(31)])
            ));
            assert!(cow_eq(&collected[3].1, HashSet::from([DocumentId(40)])));
        }

        #[test]
        fn test_overlapping_document_sets() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator 1: contains DocumentId(100) in multiple sets
            merged.add(cow_iter(vec![
                (1, HashSet::from([DocumentId(100), DocumentId(101)])),
                (3, HashSet::from([DocumentId(100), DocumentId(103)])),
            ]));

            // Iterator 2: also contains DocumentId(100)
            merged.add(cow_iter(vec![
                (2, HashSet::from([DocumentId(100), DocumentId(102)])),
                (4, HashSet::from([DocumentId(104)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4]);

            // Each set should maintain its own DocumentId combinations
            // DocumentId(100) appears in multiple sets, which is expected behavior
            assert!(cow_eq(
                &collected[0].1,
                HashSet::from([DocumentId(100), DocumentId(101)])
            ));
            assert!(cow_eq(
                &collected[1].1,
                HashSet::from([DocumentId(100), DocumentId(102)])
            ));
            assert!(cow_eq(
                &collected[2].1,
                HashSet::from([DocumentId(100), DocumentId(103)])
            ));
            assert!(cow_eq(&collected[3].1, HashSet::from([DocumentId(104)])));
        }

        #[test]
        fn test_min_max_value_usage() {
            let mut merged = MergeSortedIterator::<u8>::new(SortOrder::Ascending);

            // Iterator with min and max values
            merged.add(cow_iter(vec![
                (u8::MIN, HashSet::from([DocumentId(0)])),
                (1, HashSet::from([DocumentId(1)])),
                (u8::MAX, HashSet::from([DocumentId(255)])),
            ]));

            // Iterator with intermediate values
            merged.add(cow_iter(vec![
                (2, HashSet::from([DocumentId(2)])),
                (127, HashSet::from([DocumentId(127)])),
                (254, HashSet::from([DocumentId(254)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![0, 1, 2, 127, 254, 255]);
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(0)])));
            assert!(cow_eq(&collected[5].1, HashSet::from([DocumentId(255)])));
        }

        #[test]
        fn test_custom_ordered_key_type() {
            // Test with a different OrderedKey type (using i16 to have different bounds)

            let mut merged = MergeSortedIterator::<i16>::new(SortOrder::Descending);

            // Iterator with negative values
            merged.add(cow_iter(vec![
                (i16::MAX, HashSet::from([DocumentId(32767)])),
                (0, HashSet::from([DocumentId(0)])),
                (i16::MIN, HashSet::from([DocumentId(32768)])),
            ]));

            // Iterator with other values
            merged.add(cow_iter(vec![
                (1000, HashSet::from([DocumentId(1000)])),
                (-1000, HashSet::from([DocumentId(2000)])),
            ]));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<i16> = collected.iter().map(|(k, _)| *k).collect();

            // Should be in descending order
            assert_eq!(keys, vec![32767, 1000, 0, -1000, -32768]);
            assert!(cow_eq(&collected[0].1, HashSet::from([DocumentId(32767)])));
            assert!(cow_eq(&collected[4].1, HashSet::from([DocumentId(32768)])));
        }
    }
}
