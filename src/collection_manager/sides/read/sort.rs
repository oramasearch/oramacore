use std::collections::{HashMap, HashSet};
use std::iter::Peekable;

use anyhow::{bail, Result};
use ordered_float::NotNan;
use tracing::info;

use super::index::Index;
use super::GroupValue;
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

async fn sort_and_truncate_documents_by_field2(
    doc_ids: HashSet<DocumentId>,
    sort_by: &SortBy,
    relevant_indexes: &[&Index],
    token_scores: &HashMap<DocumentId, f32>,
    top_count: usize,
) -> Result<Vec<DocumentId>> {
    fn process_sort_iterator(
        iter: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_>,
        desiderata: usize,
        doc_ids: &HashSet<DocumentId>,
    ) -> Vec<DocumentId> {
        iter.flat_map(|(_, dd)| dd.into_iter())
            .filter(|id| doc_ids.contains(id))
            .take(desiderata)
            .collect()
    }

    let v = if relevant_indexes.len() == 1 {
        // If there's only one index, use the existing optimized path
        let output = relevant_indexes[0]
            .get_sort_iterator(&sort_by.property, sort_by.order, |iter| {
                process_sort_iterator(iter, top_count, &doc_ids)
            })
            .await?;

        // truncate(token_scores, output.into_iter(), top_count)
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
                    process_sort_iterator(iter, top_count, &doc_ids)
                })
                .await?;

            all_results.extend(index_results);
        }

        // Sort the merged results by their numeric values to ensure global ordering
        // all_results.sort_by_key(|(n, _)| *n);

        // Apply descending order if requested by reversing the sorted results
        // if sort_by.order == SortOrder::Descending {
        //     all_results.reverse();
        // }

        // Get top
        // truncate(token_scores, all_results.into_iter(), top_count)
    };
    panic!()
    // Ok(v)
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

pub async fn sort_documents_in_groups(
    groups: HashMap<Vec<GroupValue>, HashSet<DocumentId>>,
    token_scores: &HashMap<DocumentId, f32>,
    sort_by: Option<&SortBy>,
    relevant_indexes: &[&Index],
    max_results: usize,
) -> Result<HashMap<Vec<GroupValue>, Vec<DocumentId>>> {
    let mut sorted_groups = HashMap::with_capacity(groups.len());

    for (group_key, doc_set) in groups {
        let docs = if let Some(sort_by) = sort_by {
            top_n_documents_by_field(
                doc_set,
                sort_by,
                relevant_indexes,
                token_scores,
                max_results,
            )
            .await?
        } else {
            top_n_documents_by_score(doc_set, token_scores, max_results)
        };

        sorted_groups.insert(group_key, docs);
    }

    Ok(sorted_groups)
}

fn top_n_documents_by_score(
    docs: HashSet<DocumentId>,
    token_scores: &HashMap<DocumentId, f32>,
    max_results: usize,
) -> Vec<DocumentId> {
    let mut capped_heap = CappedHeap::new(max_results);

    for doc_id in docs {
        if let Some(score) = token_scores.get(&doc_id) {
            if let Ok(k) = NotNan::new(*score) {
                capped_heap.insert(k, doc_id);
            }
        }
    }

    capped_heap.into_top().map(|(_, doc_id)| doc_id).collect()
}

async fn top_n_documents_by_field(
    docs: HashSet<DocumentId>,
    sort_by: &SortBy,
    relevant_indexes: &[&Index],
    token_scores: &HashMap<DocumentId, f32>,
    max_results: usize,
) -> Result<Vec<DocumentId>> {
    panic!()
    /*
    sort_and_truncate_documents_by_field(

        sort_by,
        relevant_indexes,
        token_scores,
        max_results,
    ).await
     */
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

mod sort_iter {
    use crate::types::{DocumentId, SortOrder};
    use itertools::Itertools;
    use std::collections::HashSet;
    use std::iter::Peekable;

    pub trait OrderedKey: Ord + Eq + Clone {
        fn min_value() -> Self;
        fn max_value() -> Self;
    }

    pub struct MergeSortedIterator<'iter, T: OrderedKey> {
        iters: Vec<Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 'iter>>>,
        order: SortOrder,
        min: T,

        inner_cache: Vec<(usize, T)>,
    }

    impl<'iter, T: OrderedKey> MergeSortedIterator<'iter, T> {
        pub fn new(order: SortOrder) -> Self {
            Self {
                iters: vec![],
                order,
                min: T::min_value(),
                inner_cache: vec![],
            }
        }

        pub fn add(&mut self, iter: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 'iter>) {
            self.iters.push(iter.peekable());
        }
    }

    impl<T: OrderedKey> Iterator for MergeSortedIterator<'_, T> {
        type Item = (T, HashSet<DocumentId>);

        fn next(&mut self) -> Option<Self::Item> {
            let min = self.min.clone();

            // Eq case
            let el = self
                .iters
                .iter_mut()
                .filter_map(|iter| match iter.peek() {
                    Some((n, _)) if *n == min => Some(iter.next().unwrap()),
                    _ => None,
                })
                .next();
            if let Some(el) = el {
                return Some(el);
            }

            self.inner_cache.clear();
            self.inner_cache
                .extend(
                    self.iters
                        .iter_mut()
                        .enumerate()
                        .filter_map(|(index, iter)| match iter.peek() {
                            Some((n, _)) => Some((index, n.clone())),
                            _ => None,
                        }),
                );

            // All iters returns None
            if self.inner_cache.is_empty() {
                return None;
            }

            let index = if self.order == SortOrder::Ascending {
                self.inner_cache
                    .iter()
                    .min_by(|(_, key1), (_, key2)| key1.cmp(key2))
            } else {
                self.inner_cache
                    .iter()
                    .max_by(|(_, key1), (_, key2)| key1.cmp(key2))
            };

            if let Some((index, _)) = index {
                let iter = self.iters.get_mut(*index).unwrap();
                iter.next()
            } else {
                None
            }
        }
    }

    #[cfg(test)]
    mod sort_iter_test {
        use super::*;

        impl OrderedKey for u8 {
            fn min_value() -> Self {
                0
            }
            fn max_value() -> Self {
                u8::MAX
            }
        }

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
    }
}
