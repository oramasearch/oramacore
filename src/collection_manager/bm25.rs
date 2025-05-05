/** The structure of data needed for BM25 scoring:
 * ```text
 * (coll_id, field_id) => {
 *    average_field_length: f32,
 *    total_documents_with_field: usize,
 *    
 *    (, doc_id) => {
 *      document_length
 *    }
 *
 *    (, term) => {
 *      total_documents_with_term_in_field: usize
 *      (, doc_id) => {
 *       term_occurrence_in_document: usize
 *      }
 *    }
 * }
 * ```
 *
 * RAW data:
 * ```text
 * (coll_id, field_id) => [average_field_length, total_documents_with_field]
 *
 * (coll_id, field_id, term) => [total_documents_with_term_in_field]
 *
 * (coll_id, field_id, doc_id) => [document_length]
 *
 * (coll_id, field_id, doc_id, term) => [term_occurrence_in_document]
 * ```
 */
use std::{collections::HashMap, fmt::Debug, hash::Hash};
use tracing::error;

/// BM25 scoring function
///
/// # Arguments
///
/// * `term_occurrence_in_document` - occurrence of the term in the field in the document
///   (coll_id, doc_id, field_id, term_id)
/// * `document_length` - length of the field of the document in words
///   (coll_id, field_id, doc_id)
/// * `average_field_length` - average field length in the collection
///   (coll_id, field_id)
/// * `total_documents_with_field` - number of documents that has that field in the collection
///   (coll_id, field_id)
/// * `total_documents_with_term_in_field` - number of documents that has that term in the field in the collection
///   (coll_id, field_id, term_id)
/// * `k` - k parameter
/// * `b` - b parameter
///
/// # Returns
///
/// * `f32` - BM25 score
fn bm25_score(
    term_occurrence_in_document: usize,
    document_length: u32,
    average_field_length: f32,
    total_documents_with_field: f32,
    total_documents_with_term_in_field: usize,
    k: f32,
    b: f32,
) -> f32 {
    let f = term_occurrence_in_document as f32;
    let l = document_length as f32;
    let avgdl = average_field_length;

    let ni = total_documents_with_term_in_field as f32;

    let idf = ((total_documents_with_field - ni + 0.5_f32) / (ni + 0.5_f32)).ln_1p();

    idf * (f * (k + 1.0)) / (f + k * (1.0 - b + b * (l / avgdl)))
}

pub enum BM25Scorer<K: Eq + Hash> {
    Plain(BM25ScorerPlain<K>),
    WithThreshold(BM25ScorerWithThreshold<K>),
}
impl<K: Eq + Hash + Debug> BM25Scorer<K> {
    pub fn plain() -> Self {
        Self::Plain(BM25ScorerPlain::new())
    }

    pub fn with_threshold(threshold: u32) -> Self {
        Self::WithThreshold(BM25ScorerWithThreshold {
            threshold,
            scores: Default::default(),
            term_index: 0,
        })
    }

    pub fn next_term(&mut self) {
        if let Self::WithThreshold(scorer) = self {
            scorer.next_term();
        }
    }

    pub fn reset_term(&mut self) {
        if let Self::WithThreshold(scorer) = self {
            scorer.reset_term();
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        key: K,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        b: f32,
        boost: f32,
        token_indexes: u32,
    ) {
        match self {
            Self::Plain(scorer) => scorer.add(
                key,
                term_occurrence_in_field,
                field_length,
                average_field_length,
                total_documents_with_field,
                total_documents_with_term_in_field,
                k,
                b,
                boost,
            ),
            Self::WithThreshold(scorer) => {
                scorer.add(
                    key,
                    term_occurrence_in_field,
                    field_length,
                    average_field_length,
                    total_documents_with_field,
                    total_documents_with_term_in_field,
                    k,
                    b,
                    boost,
                    token_indexes,
                );
            }
        }
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        match self {
            Self::Plain(scorer) => scorer.get_scores(),
            Self::WithThreshold(scorer) => scorer.get_scores(),
        }
    }
}

pub struct BM25ScorerWithThreshold<K: Eq + Hash> {
    threshold: u32,
    term_index: usize,
    scores: HashMap<K, (u32, f32)>,
}
impl<K: Eq + Hash + Debug> BM25ScorerWithThreshold<K> {
    pub fn next_term(&mut self) {
        self.term_index += 1;
    }
    pub fn reset_term(&mut self) {
        self.term_index = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        key: K,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        b: f32,
        boost: f32,
        token_indexes: u32,
    ) {
        let score = bm25_score(
            term_occurrence_in_field as usize,
            field_length,
            average_field_length,
            total_documents_with_field,
            total_documents_with_term_in_field,
            k,
            b,
        );

        if score.is_nan() {
            error!(
                ?term_occurrence_in_field,
                ?field_length,
                ?average_field_length,
                ?total_documents_with_field,
                ?total_documents_with_term_in_field,
                ?k,
                ?b,
                "score is NaN. Skipping item"
            );
            return;
        }

        let score = score * boost;

        let old_score = self.scores.entry(key).or_default();

        old_score.0 |= if token_indexes > 0 {
            token_indexes
        } else {
            1 << self.term_index
        };

        // This "+" operation doesn't distinguish between the FieldId.
        // This means that if a document matches on the same field
        // or on different fields, it is the same.
        // But this is not good for the ranking.
        // TODO: we should distinguish between the FieldId.
        old_score.1 += score;
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.scores
            .into_iter()
            .filter_map(|(key, (count, score))| {
                if count.count_ones() >= self.threshold {
                    Some((key, score))
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct BM25ScorerPlain<K: Eq + Hash> {
    scores: HashMap<K, f32>,
}

impl<K: Eq + Hash + Debug> Default for BM25ScorerPlain<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Debug> BM25ScorerPlain<K> {
    pub fn new() -> Self {
        Self {
            scores: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        key: K,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        b: f32,
        boost: f32,
    ) {
        let score = bm25_score(
            term_occurrence_in_field as usize,
            field_length,
            average_field_length,
            total_documents_with_field,
            total_documents_with_term_in_field,
            k,
            b,
        );

        if score.is_nan() {
            error!(
                ?term_occurrence_in_field,
                ?field_length,
                ?average_field_length,
                ?total_documents_with_field,
                ?total_documents_with_term_in_field,
                ?k,
                ?b,
                "score is NaN. Skipping item"
            );
            return;
        }

        let score = score * boost;

        let old_score = self.scores.entry(key).or_default();

        // This "+" operation doesn't distinguish between the FieldId.
        // This means that if a document matches on the same field
        // or on different fields, it is the same.
        // But this is not good for the ranking.
        // TODO: we should distinguish between the FieldId.
        *old_score += score;
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.scores
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_indexes_string_scorer_bm25() {
        let mut scorer = BM25Scorer::plain();

        scorer.add("doc1", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 1.0, 0);

        let scores = scorer.get_scores();
        assert_eq!(scores.len(), 1);
        assert_approx_eq!(scores["doc1"], 1.2297773);
    }

    #[test]
    fn test_indexes_string_scorer_bm25_boost() {
        let mut scorer = BM25Scorer::plain();
        scorer.add("doc1", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 1.0, 0);
        scorer.add("doc2", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 2.0, 0);
        scorer.add("doc3", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 0.5, 0);
        let scores = scorer.get_scores();

        assert!(scores["doc2"] > scores["doc1"]);
        assert!(scores["doc3"] < scores["doc1"]);
    }
}
