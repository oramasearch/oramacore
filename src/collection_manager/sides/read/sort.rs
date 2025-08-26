use std::collections::{HashMap, HashSet};
use std::iter::Peekable;

use anyhow::{anyhow, Result};
use ordered_float::NotNan;

use crate::{
    capped_heap::CappedHeap,
    types::{DocumentId, Limit, Number, SearchOffset, SortBy, SortOrder, TokenScore},
};
use crate::pin_rules::Consequence;
use super::index::Index;

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

/// Truncate results based on limit and offset, applying token scores
fn truncate<I: Iterator<Item = (Number, HashSet<DocumentId>)>>(
    token_scores: &HashMap<DocumentId, f32>,
    output: I,
    limit: Limit,
    offset: SearchOffset,
) -> Vec<TokenScore> {
    let mut res = Vec::with_capacity(limit.0 + offset.0);
    let mut k = 0_usize;
    for (_, docs) in output {
        for doc in docs {
            let score = token_scores.get(&doc);
            if let Some(score) = score {
                if k >= offset.0 && k < limit.0 + offset.0 {
                    res.push(TokenScore {
                        document_id: doc,
                        score: *score,
                    });
                }

                k += 1;

                if k > limit.0 + offset.0 {
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
pub async fn sort_and_truncate_documents(
    relevant_indexes: &[&Index],
    pins: Vec<Consequence<DocumentId>>,
    token_scores: &HashMap<DocumentId, f32>,
    limit: Limit,
    offset: SearchOffset,
    sort_by: Option<&SortBy>,
) -> Result<Vec<TokenScore>> {
    if let Some(sort_by) = sort_by {
        if relevant_indexes.is_empty() {
            return Err(anyhow!(
                "Cannot sort by \"{}\": no index has that field",
                sort_by.property
            ));
        }

        // Calculate total number of documents needed (limit + offset for pagination)
        let desiderata = limit.0 + offset.0;

        // If there's only one index, use the existing optimized path
        // Single-index sorting is more efficient as it avoids collecting and re-sorting all results
        if relevant_indexes.len() == 1 {
            let output = relevant_indexes[0]
                .get_sort_iterator(&sort_by.property, sort_by.order, |iter| {
                    process_sort_iterator(iter, desiderata)
                })
                .await?;

            return Ok(truncate(token_scores, output.into_iter(), limit, offset));
        }

        // - For the relevant_indexes, we obtain the first `desiderata` items.
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
        // Note: Each index only returns up to 'desiderata' results, not all documents
        let mut all_results = Vec::new();

        for index in relevant_indexes {
            let index_results = index
                .get_sort_iterator(&sort_by.property, sort_by.order, |iter| {
                    process_sort_iterator(iter, desiderata)
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

        // Apply pagination (limit/offset) and convert to final TokenScore format
        Ok(truncate(
            token_scores,
            all_results.into_iter(),
            limit,
            offset,
        ))
    } else {
        // No sorting requested - fall back to simple top-N selection based on token scores
        // This is the most efficient path when only relevance ranking is needed
        let top = top_n(token_scores.iter(), limit.0 + offset.0);

        let top = if !pins.is_empty() {
            apply_pin_rules(pins, token_scores, top)
        } else {
            top
        };

        Ok(top)
    }
}

fn apply_pin_rules(
    pins: Vec<Consequence<DocumentId>>,
    token_scores: &HashMap<DocumentId, f32>,
    top: Vec<TokenScore>
) -> Vec<TokenScore> {
    unimplemented!("implement `apply_pin_rules`")
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
