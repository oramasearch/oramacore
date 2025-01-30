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

#[derive(Debug, Default)]
pub struct BM25Scorer<K: Eq + Hash> {
    scores: HashMap<K, f32>,
}

impl<K: Eq + Hash + Debug> BM25Scorer<K> {
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
        let mut scorer = BM25Scorer::new();

        scorer.add("doc1", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 1.0);

        let scores = scorer.get_scores();
        assert_eq!(scores.len(), 1);
        assert_approx_eq!(scores["doc1"], 1.2297773);
    }

    #[test]
    fn test_indexes_string_scorer_bm25_boost() {
        let mut scorer = BM25Scorer::new();
        scorer.add("doc1", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 1.0);
        scorer.add("doc2", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 2.0);
        scorer.add("doc3", 5, 100, 100.0, 10.0, 5, 1.2, 0.75, 0.5);
        let scores = scorer.get_scores();

        assert!(scores["doc2"] > scores["doc1"]);
        assert!(scores["doc3"] < scores["doc1"]);
    }
}
