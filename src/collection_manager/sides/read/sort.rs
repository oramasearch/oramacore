use std::collections::{HashMap, HashSet};

use anyhow::Result;
use oramacore_lib::data_structures::capped_heap::CappedHeap;
use ordered_float::NotNan;
use tracing::warn;

use super::{GroupValue, IndexSortContext};
use crate::collection_manager::sides::read::sort::sort_iter::{
    CombinedSortFilter, MergeSortedIterator,
};
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

        // Use token_scores as filter to skip batches where no documents have scores
        let mut merge_sorted_iterator = MergeSortedIterator::new(sort_by.order, token_scores);
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
/// Returns a vector of (Number, &HashSet<DocumentId>) tuples where
/// the sum of the sizes of the HashSets is at least `desiderata`.
fn process_sort_iterator<'a>(
    iter: Box<dyn Iterator<Item = (Number, &'a HashSet<DocumentId>)> + 'a>,
    desiderata: usize,
) -> Vec<(Number, &'a HashSet<DocumentId>)> {
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
                    .flat_map(|(_, h)| h.iter().copied())
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

                // Use CombinedSortFilter to filter by both group membership and token_scores
                let filter = CombinedSortFilter {
                    set: &docs,
                    map: token_scores,
                };
                let mut merge_sorted_iterator = MergeSortedIterator::new(sort_by.order, filter);
                for i in &v {
                    merge_sorted_iterator.add(i.execute()?);
                }

                // With the filter, we still need per-doc filtering since batch may contain mixed docs
                let data: Vec<DocumentId> = merge_sorted_iterator
                    .flat_map(|(_, h)| h.iter().copied())
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
fn truncate<'a, I: Iterator<Item = (Number, &'a HashSet<DocumentId>)>>(
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
    use std::collections::{HashMap, HashSet};
    use std::fmt::Debug;
    use std::hash::Hash;
    use std::iter::Peekable;

    pub trait OrderedKey: Ord + Eq + Clone + Debug {}
    impl OrderedKey for Number {}

    // =============================================================================
    // SortFilter trait - allows early filtering during sort iteration
    // =============================================================================

    /// Trait for filtering documents during sort iteration.
    ///
    /// This trait enables early filtering of documents as they are yielded by
    /// the sort iterator, avoiding iteration over documents that won't be in
    /// the final result.
    ///
    /// Implementations are provided for:
    /// - `()` - No filtering (includes all documents)
    /// - `&HashSet<DocId>` - Include only documents in the set (for group filtering)
    /// - `&HashMap<DocId, V>` - Include only documents with entries (for token_scores filtering)
    pub trait SortFilter<DocId> {
        fn should_include(&self, doc_id: &DocId) -> bool;
    }

    /// No-op filter that includes all documents.
    impl<DocId> SortFilter<DocId> for () {
        #[inline]
        fn should_include(&self, _: &DocId) -> bool {
            true
        }
    }

    /// Filter that includes only documents present in the HashSet.
    /// Used for group filtering where we have a set of document IDs in a group.
    impl<DocId: Eq + Hash> SortFilter<DocId> for &HashSet<DocId> {
        #[inline]
        fn should_include(&self, doc_id: &DocId) -> bool {
            self.contains(doc_id)
        }
    }

    /// Filter that includes only documents with entries in the HashMap.
    /// Used for token_scores filtering where we want documents that have scores.
    impl<DocId: Eq + Hash, V> SortFilter<DocId> for &HashMap<DocId, V> {
        #[inline]
        fn should_include(&self, doc_id: &DocId) -> bool {
            self.contains_key(doc_id)
        }
    }

    /// Combined filter that requires documents to pass both a HashSet check AND a HashMap check.
    /// Used for group sorting where we need to filter by both group membership and token_scores.
    pub struct CombinedFilter<'a, DocId, V> {
        pub set: &'a HashSet<DocId>,
        pub map: &'a HashMap<DocId, V>,
    }

    impl<DocId: Eq + Hash, V> SortFilter<DocId> for CombinedFilter<'_, DocId, V> {
        #[inline]
        fn should_include(&self, doc_id: &DocId) -> bool {
            self.set.contains(doc_id) && self.map.contains_key(doc_id)
        }
    }

    // =============================================================================
    // MergeSortedIterator - N-way merge with filtering
    // =============================================================================

    /// A boxed iterator yielding (sort_key, document_ids) pairs.
    type SortedIter<'s, T> = Box<dyn Iterator<Item = (T, &'s HashSet<DocumentId>)> + 's>;

    /// A peekable boxed iterator for merge operations.
    type PeekableSortedIter<'s, T> = Peekable<SortedIter<'s, T>>;

    /// Iterator that merges multiple sorted iterators into a single sorted stream.
    ///
    /// This iterator takes N already-sorted iterators and produces a merged
    /// output that maintains the sort order. It uses a peek-compare-advance
    /// strategy to efficiently merge without buffering all data.
    ///
    /// The iterator also supports filtering documents via the `SortFilter` trait,
    /// allowing early exclusion of batches where no documents match the filter.
    ///
    /// # Filter Behavior (Batch-Level Filtering)
    ///
    /// **Important**: The filter operates at the batch level, not the document level.
    /// - If NO documents in a batch match the filter → the batch is **skipped entirely**
    /// - If ANY document in a batch matches the filter → the batch is **returned as-is**
    ///   (including documents that don't match the filter)
    ///
    /// This means the caller is still responsible for filtering individual documents
    /// within returned batches. The filter's purpose is to skip entirely non-matching
    /// batches early, avoiding unnecessary iteration over them.
    pub struct MergeSortedIterator<'iter, Key: OrderedKey, Filter: SortFilter<DocumentId>> {
        iters: Vec<PeekableSortedIter<'iter, Key>>,
        order: SortOrder,
        filter: Filter,
    }

    impl<'iter, Key: OrderedKey, Filter: SortFilter<DocumentId>>
        MergeSortedIterator<'iter, Key, Filter>
    {
        /// Creates a new MergeSortedIterator with the specified filter.
        ///
        /// Documents in batches where no document passes the filter will be skipped
        /// at the iterator level.
        pub fn new(order: SortOrder, filter: Filter) -> Self {
            Self {
                iters: vec![],
                order,
                filter,
            }
        }

        /// Adds a sorted iterator to be merged.
        ///
        /// The iterator must already be sorted according to the order specified
        /// during construction.
        pub fn add(&mut self, iter: SortedIter<'iter, Key>) {
            self.iters.push(iter.peekable());
        }
    }

    impl<'iter, Key: OrderedKey, Filter: SortFilter<DocumentId>> Iterator
        for MergeSortedIterator<'iter, Key, Filter>
    {
        type Item = (Key, &'iter HashSet<DocumentId>);

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                // Find the index of the iterator with the best (min/max) key.
                // We store the best key clone to avoid double mutable borrow of self.iters.
                let mut best: Option<(usize, Key)> = None;

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

                let (index, _) = best?;
                let (key, docs) = self.iters[index].next()?;

                // Skip batch if no documents match filter
                let has_matching_doc = docs.iter().any(|doc| self.filter.should_include(doc));
                if !has_matching_doc {
                    continue; // Skip this batch entirely
                }

                return Some((key, docs));
            }
        }
    }

    // Re-export for use in tests
    pub use CombinedFilter as CombinedSortFilter;

    #[cfg(test)]
    mod sort_iter_test {
        use super::*;

        impl OrderedKey for u8 {}
        impl OrderedKey for i16 {}

        /// Helper to compare HashSet references with expected HashSet
        fn set_eq(set: &HashSet<DocumentId>, expected: HashSet<DocumentId>) -> bool {
            set == &expected
        }

        #[test]
        fn test_empty() {
            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            assert!(merged.next().is_none());
        }

        #[test]
        fn test_one() {
            let data = [
                (
                    0_u8,
                    HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)]),
                ),
                (
                    2,
                    HashSet::from([DocumentId(6), DocumentId(8), DocumentId(9)]),
                ),
            ];
            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));
            let collected: Vec<_> = merged.collect();
            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].0, 0);
            assert!(set_eq(
                collected[0].1,
                HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)])
            ));
            assert_eq!(collected[1].0, 2);
            assert!(set_eq(
                collected[1].1,
                HashSet::from([DocumentId(6), DocumentId(8), DocumentId(9)])
            ));
        }

        #[test]
        fn test_multiple_iterators_ascending() {
            // Iterator 1: [1, 5, 9]
            let data1 = [
                (1_u8, HashSet::from([DocumentId(10)])),
                (5, HashSet::from([DocumentId(50)])),
                (9, HashSet::from([DocumentId(90)])),
            ];

            // Iterator 2: [2, 6, 10]
            let data2 = [
                (2_u8, HashSet::from([DocumentId(20)])),
                (6, HashSet::from([DocumentId(60)])),
                (10, HashSet::from([DocumentId(100)])),
            ];

            // Iterator 3: [3, 4, 7, 8]
            let data3 = vec![
                (3_u8, HashSet::from([DocumentId(30)])),
                (4, HashSet::from([DocumentId(40)])),
                (7, HashSet::from([DocumentId(70)])),
                (8, HashSet::from([DocumentId(80)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data3.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

            // Verify document IDs are preserved correctly
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(20)])));
            assert!(set_eq(collected[9].1, HashSet::from([DocumentId(100)])));
        }

        #[test]
        fn test_multiple_iterators_descending() {
            // Iterator 1: [9, 5, 1] (descending)
            let data1 = [
                (9_u8, HashSet::from([DocumentId(90)])),
                (5, HashSet::from([DocumentId(50)])),
                (1, HashSet::from([DocumentId(10)])),
            ];

            // Iterator 2: [10, 6, 2] (descending)
            let data2 = [
                (10_u8, HashSet::from([DocumentId(100)])),
                (6, HashSet::from([DocumentId(60)])),
                (2, HashSet::from([DocumentId(20)])),
            ];

            // Iterator 3: [8, 7, 4, 3] (descending)
            let data3 = vec![
                (8_u8, HashSet::from([DocumentId(80)])),
                (7, HashSet::from([DocumentId(70)])),
                (4, HashSet::from([DocumentId(40)])),
                (3, HashSet::from([DocumentId(30)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Descending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data3.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);

            // Verify document IDs are preserved correctly
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(100)])));
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(90)])));
            assert!(set_eq(collected[9].1, HashSet::from([DocumentId(10)])));
        }

        #[test]
        fn test_duplicate_keys_across_iterators() {
            // Iterator 1: [1, 3, 5]
            let data1 = [
                (1_u8, HashSet::from([DocumentId(10)])),
                (3, HashSet::from([DocumentId(30)])),
                (5, HashSet::from([DocumentId(50)])),
            ];

            // Iterator 2: [1, 3, 4] - has duplicate keys 1 and 3
            let data2 = [
                (1_u8, HashSet::from([DocumentId(11)])),
                (3, HashSet::from([DocumentId(31)])),
                (4, HashSet::from([DocumentId(40)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            // Should preserve all items, including duplicates
            assert_eq!(keys, vec![1, 1, 3, 3, 4, 5]);

            // Verify both document sets for key 1 are preserved
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(11)])));

            // Verify both document sets for key 3 are preserved
            assert!(set_eq(collected[2].1, HashSet::from([DocumentId(30)])));
            assert!(set_eq(collected[3].1, HashSet::from([DocumentId(31)])));
        }

        #[test]
        fn test_duplicate_keys_same_iterator() {
            // Single iterator with duplicate keys
            let data = vec![
                (1_u8, HashSet::from([DocumentId(10)])),
                (1, HashSet::from([DocumentId(11)])),
                (2, HashSet::from([DocumentId(20)])),
                (2, HashSet::from([DocumentId(21)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 1, 2, 2]);
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(11)])));
            assert!(set_eq(collected[2].1, HashSet::from([DocumentId(20)])));
            assert!(set_eq(collected[3].1, HashSet::from([DocumentId(21)])));
        }

        #[test]
        fn test_single_empty_iterator() {
            let data: Vec<(u8, HashSet<DocumentId>)> = vec![];
            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            assert!(collected.is_empty());
        }

        #[test]
        fn test_mix_empty_and_non_empty_iterators() {
            let empty1: Vec<(u8, HashSet<DocumentId>)> = vec![];
            let data1 = [
                (1_u8, HashSet::from([DocumentId(10)])),
                (3, HashSet::from([DocumentId(30)])),
            ];
            let empty2: Vec<(u8, HashSet<DocumentId>)> = vec![];
            let data2 = [
                (2_u8, HashSet::from([DocumentId(20)])),
                (4, HashSet::from([DocumentId(40)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(empty1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(empty2.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            // Should only get results from non-empty iterators
            assert_eq!(keys, vec![1, 2, 3, 4]);
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(10)])));
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(20)])));
            assert!(set_eq(collected[2].1, HashSet::from([DocumentId(30)])));
            assert!(set_eq(collected[3].1, HashSet::from([DocumentId(40)])));
        }

        #[test]
        fn test_all_iterators_exhausted_simultaneously() {
            let data1 = [(1_u8, HashSet::from([DocumentId(10)]))];
            let data2 = [(2_u8, HashSet::from([DocumentId(20)]))];
            let data3 = [(3_u8, HashSet::from([DocumentId(30)]))];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data3.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3]);
            assert_eq!(collected.len(), 3);
        }

        #[test]
        fn test_iterators_different_lengths() {
            // Short iterator: [1, 2]
            let data1 = [
                (1_u8, HashSet::from([DocumentId(10)])),
                (2, HashSet::from([DocumentId(20)])),
            ];

            // Long iterator: [3, 4, 5, 6, 7, 8]
            let data2 = vec![
                (3_u8, HashSet::from([DocumentId(30)])),
                (4, HashSet::from([DocumentId(40)])),
                (5, HashSet::from([DocumentId(50)])),
                (6, HashSet::from([DocumentId(60)])),
                (7, HashSet::from([DocumentId(70)])),
                (8, HashSet::from([DocumentId(80)])),
            ];

            // Medium iterator: [9, 10, 11]
            let data3 = [
                (9_u8, HashSet::from([DocumentId(90)])),
                (10, HashSet::from([DocumentId(100)])),
                (11, HashSet::from([DocumentId(110)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data3.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            assert_eq!(collected.len(), 11);
        }

        #[test]
        fn test_one_iterator_much_longer() {
            // Very short iterator: [100]
            let data1 = [(100_u8, HashSet::from([DocumentId(1000)]))];

            // Much longer iterator: [99, 98, 97, 96, 95, 94, 93, 92, 91, 90]
            let data2 = vec![
                (99_u8, HashSet::from([DocumentId(990)])),
                (98, HashSet::from([DocumentId(980)])),
                (97, HashSet::from([DocumentId(970)])),
                (96, HashSet::from([DocumentId(960)])),
                (95, HashSet::from([DocumentId(950)])),
                (94, HashSet::from([DocumentId(940)])),
                (93, HashSet::from([DocumentId(930)])),
                (92, HashSet::from([DocumentId(920)])),
                (91, HashSet::from([DocumentId(910)])),
                (90, HashSet::from([DocumentId(900)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Descending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]);

            // Verify first item comes from the shorter iterator
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(1000)])));

            // Verify remaining items come from the longer iterator
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(990)])));
            assert!(set_eq(collected[10].1, HashSet::from([DocumentId(900)])));
        }

        #[test]
        fn test_document_id_set_merging() {
            // Iterator with multiple documents per key
            let data1 = [
                (
                    1_u8,
                    HashSet::from([DocumentId(10), DocumentId(11), DocumentId(12)]),
                ),
                (3, HashSet::from([DocumentId(30), DocumentId(31)])),
            ];

            // Iterator with single documents per key
            let data2 = [
                (2_u8, HashSet::from([DocumentId(20)])),
                (4, HashSet::from([DocumentId(40)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4]);

            // Verify multi-document sets are preserved
            assert!(set_eq(
                collected[0].1,
                HashSet::from([DocumentId(10), DocumentId(11), DocumentId(12)])
            ));
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(20)])));
            assert!(set_eq(
                collected[2].1,
                HashSet::from([DocumentId(30), DocumentId(31)])
            ));
            assert!(set_eq(collected[3].1, HashSet::from([DocumentId(40)])));
        }

        #[test]
        fn test_overlapping_document_sets() {
            // Iterator 1: contains DocumentId(100) in multiple sets
            let data1 = [
                (1_u8, HashSet::from([DocumentId(100), DocumentId(101)])),
                (3, HashSet::from([DocumentId(100), DocumentId(103)])),
            ];

            // Iterator 2: also contains DocumentId(100)
            let data2 = [
                (2_u8, HashSet::from([DocumentId(100), DocumentId(102)])),
                (4, HashSet::from([DocumentId(104)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![1, 2, 3, 4]);

            // Each set should maintain its own DocumentId combinations
            // DocumentId(100) appears in multiple sets, which is expected behavior
            assert!(set_eq(
                collected[0].1,
                HashSet::from([DocumentId(100), DocumentId(101)])
            ));
            assert!(set_eq(
                collected[1].1,
                HashSet::from([DocumentId(100), DocumentId(102)])
            ));
            assert!(set_eq(
                collected[2].1,
                HashSet::from([DocumentId(100), DocumentId(103)])
            ));
            assert!(set_eq(collected[3].1, HashSet::from([DocumentId(104)])));
        }

        #[test]
        fn test_min_max_value_usage() {
            // Iterator with min and max values
            let data1 = [
                (u8::MIN, HashSet::from([DocumentId(0)])),
                (1_u8, HashSet::from([DocumentId(1)])),
                (u8::MAX, HashSet::from([DocumentId(255)])),
            ];

            // Iterator with intermediate values
            let data2 = [
                (2_u8, HashSet::from([DocumentId(2)])),
                (127, HashSet::from([DocumentId(127)])),
                (254, HashSet::from([DocumentId(254)])),
            ];

            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<u8> = collected.iter().map(|(k, _)| *k).collect();

            assert_eq!(keys, vec![0, 1, 2, 127, 254, 255]);
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(0)])));
            assert!(set_eq(collected[5].1, HashSet::from([DocumentId(255)])));
        }

        #[test]
        fn test_custom_ordered_key_type() {
            // Test with a different OrderedKey type (using i16 to have different bounds)

            // Iterator with negative values
            let data1 = [
                (i16::MAX, HashSet::from([DocumentId(32767)])),
                (0_i16, HashSet::from([DocumentId(0)])),
                (i16::MIN, HashSet::from([DocumentId(32768)])),
            ];

            // Iterator with other values
            let data2 = [
                (1000_i16, HashSet::from([DocumentId(1000)])),
                (-1000, HashSet::from([DocumentId(2000)])),
            ];

            let mut merged = MergeSortedIterator::<i16, ()>::new(SortOrder::Descending, ());
            merged.add(Box::new(data1.iter().map(|(k, v)| (*k, v))));
            merged.add(Box::new(data2.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();
            let keys: Vec<i16> = collected.iter().map(|(k, _)| *k).collect();

            // Should be in descending order
            assert_eq!(keys, vec![32767, 1000, 0, -1000, -32768]);
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(32767)])));
            assert!(set_eq(collected[4].1, HashSet::from([DocumentId(32768)])));
        }

        // =============================================================================
        // Filter tests
        // =============================================================================

        #[test]
        fn test_filter_skips_empty_batches() {
            // Data with 3 batches
            let data = [
                (1_u8, HashSet::from([DocumentId(10), DocumentId(11)])),
                (2, HashSet::from([DocumentId(20)])),
                (3, HashSet::from([DocumentId(30), DocumentId(31)])),
            ];

            // Filter that only includes DocumentId(20) and DocumentId(31)
            let filter_set: HashSet<DocumentId> = HashSet::from([DocumentId(20), DocumentId(31)]);
            let mut merged = MergeSortedIterator::<u8, _>::new(SortOrder::Ascending, &filter_set);
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();

            // Should skip batch 1 (no matches), return batch 2 and 3
            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].0, 2); // Batch with DocumentId(20)
                                           // Verify batch 2 document set
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(20)])));

            assert_eq!(collected[1].0, 3); // Batch with DocumentId(31) (and 30)
                                           // Verify batch 3 returns FULL set (batch-level filtering returns entire batch)
            assert!(set_eq(
                collected[1].1,
                HashSet::from([DocumentId(30), DocumentId(31)])
            ));
        }

        #[test]
        fn test_filter_with_hashmap() {
            let data = [
                (1_u8, HashSet::from([DocumentId(10), DocumentId(11)])),
                (2, HashSet::from([DocumentId(20)])),
                (3, HashSet::from([DocumentId(30)])),
            ];

            // Filter using HashMap (simulating token_scores)
            let mut scores: HashMap<DocumentId, f32> = HashMap::new();
            scores.insert(DocumentId(11), 1.0);
            scores.insert(DocumentId(30), 2.0);

            let mut merged = MergeSortedIterator::<u8, _>::new(SortOrder::Ascending, &scores);
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();

            // Should include batch 1 (has DocumentId(11)) and batch 3 (has DocumentId(30))
            // Should skip batch 2 (DocumentId(20) not in scores)
            assert_eq!(collected.len(), 2);
            assert_eq!(collected[0].0, 1);
            // Verify batch 1 returns FULL set (both docs, even though only DocumentId(11) is in filter)
            assert!(set_eq(
                collected[0].1,
                HashSet::from([DocumentId(10), DocumentId(11)])
            ));

            assert_eq!(collected[1].0, 3);
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(30)])));
        }

        #[test]
        fn test_combined_filter() {
            let data = [
                (1_u8, HashSet::from([DocumentId(10), DocumentId(11)])),
                (2, HashSet::from([DocumentId(20), DocumentId(21)])),
                (3, HashSet::from([DocumentId(30)])),
            ];

            // Group filter - only allow DocumentId(11), DocumentId(20), DocumentId(30)
            let group_set: HashSet<DocumentId> =
                HashSet::from([DocumentId(11), DocumentId(20), DocumentId(30)]);

            // Score filter - only DocumentId(11) and DocumentId(21) have scores
            let mut scores: HashMap<DocumentId, f32> = HashMap::new();
            scores.insert(DocumentId(11), 1.0);
            scores.insert(DocumentId(21), 2.0);

            // Combined filter: must be in BOTH group AND scores
            // Only DocumentId(11) passes both filters
            let filter = CombinedFilter {
                set: &group_set,
                map: &scores,
            };

            let mut merged = MergeSortedIterator::<u8, _>::new(SortOrder::Ascending, filter);
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();

            // Only batch 1 should be returned (has DocumentId(11) which is in both filters)
            // Batch 2: DocumentId(20) in group but not scores, DocumentId(21) in scores but not group
            // Batch 3: DocumentId(30) in group but not scores
            assert_eq!(collected.len(), 1);
            assert_eq!(collected[0].0, 1);
            // Verify batch 1 returns FULL set (batch-level filtering returns entire batch)
            assert!(set_eq(
                collected[0].1,
                HashSet::from([DocumentId(10), DocumentId(11)])
            ));
        }

        #[test]
        fn test_noop_filter_includes_all() {
            let data = [
                (1_u8, HashSet::from([DocumentId(10)])),
                (2, HashSet::from([DocumentId(20)])),
                (3, HashSet::from([DocumentId(30)])),
            ];

            // () filter should include everything
            let mut merged = MergeSortedIterator::<u8, ()>::new(SortOrder::Ascending, ());
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();

            // All batches should be included with correct keys and document sets
            assert_eq!(collected.len(), 3);
            assert_eq!(collected[0].0, 1);
            assert!(set_eq(collected[0].1, HashSet::from([DocumentId(10)])));
            assert_eq!(collected[1].0, 2);
            assert!(set_eq(collected[1].1, HashSet::from([DocumentId(20)])));
            assert_eq!(collected[2].0, 3);
            assert!(set_eq(collected[2].1, HashSet::from([DocumentId(30)])));
        }

        #[test]
        fn test_filter_all_batches_empty() {
            let data = [
                (1_u8, HashSet::from([DocumentId(10)])),
                (2, HashSet::from([DocumentId(20)])),
                (3, HashSet::from([DocumentId(30)])),
            ];

            // Filter that matches nothing
            let filter_set: HashSet<DocumentId> = HashSet::from([DocumentId(999)]);
            let mut merged = MergeSortedIterator::<u8, _>::new(SortOrder::Ascending, &filter_set);
            merged.add(Box::new(data.iter().map(|(k, v)| (*k, v))));

            let collected: Vec<_> = merged.collect();

            // No batches should be returned
            assert_eq!(collected.len(), 0);
        }
    }
}
