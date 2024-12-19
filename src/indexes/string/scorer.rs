use std::{collections::HashMap, hash::Hash};

/**
 *
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
 *
 */

/*
 * (coll_id, field_id) => [average_field_length, total_documents_with_field]
 *
 * (coll_id, field_id, term) => [total_documents_with_term_in_field]
 *
 * (coll_id, field_id, doc_id) => [document_length]
 *
 * (coll_id, field_id, doc_id, term) => [term_occurrence_in_document]
 */

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

#[derive(Debug)]
pub struct BM25Scorer<K: Eq + Hash> {
    scores: HashMap<K, f32>,
}

impl<K: Eq + Hash> BM25Scorer<K> {
    pub fn new() -> Self {
        Self {
            scores: Default::default(),
        }
    }

    pub fn add(
        &mut self,
        key: K,
        term_occurrence_in_document: u32,
        document_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        b: f32,
        boost: f32,
    ) {
        let score = bm25_score(
            term_occurrence_in_document as usize,
            document_length,
            average_field_length,
            total_documents_with_field,
            total_documents_with_term_in_field,
            k,
            b,
        );
        let score = score * boost;

        let old_score = self.scores.entry(key).or_default();
        *old_score += score;
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.scores
    }
}

/*

#[cfg(test)]
mod tests {

    use crate::collection_manager::dto::FieldId;

    use super::*;

    #[test]
    fn test_bm25_basic_scoring() {
        let scorer = bm25::BM25Score::new();
        let global_info = GlobalInfo {
            total_documents: 10,
            total_document_length: 1000,
        };

        let posting = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            occurrence: 5,
            field_length: 100,
            positions: vec![1, 2, 3, 4, 5],
        };

        scorer.add_entry(&global_info, &posting, 1.0, 1.0);
        let scores = scorer.get_scores();
        assert!(scores.contains_key(&DocumentId(1)));
        assert!(scores[&DocumentId(1)] > 0.0);
    }

    #[test]
    fn test_bm25_empty_document() {
        let scorer = bm25::BM25Score::new();
        let global_info = GlobalInfo {
            total_documents: 1,
            total_document_length: 0,
        };

        let posting = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            occurrence: 0,
            field_length: 0,
            positions: vec![],
        };

        scorer.add_entry(&global_info, &posting, 1.0, 1.0);
        let scores = scorer.get_scores();
        assert_eq!(scores[&DocumentId(1)], 0.0);
    }

    #[test]
    fn test_bm25_boost_effect() {
        let scorer = bm25::BM25Score::new();
        let global_info = GlobalInfo {
            total_documents: 10,
            total_document_length: 1000,
        };

        let posting = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            occurrence: 5,
            field_length: 100,
            positions: vec![1, 2, 3, 4, 5],
        };

        // Test with different boost values
        scorer.add_entry(&global_info, &posting, 1.0, 1.0);
        let normal_scores = scorer.get_scores();

        let scorer = bm25::BM25Score::new();
        scorer.add_entry(&global_info, &posting, 1.0, 2.0);
        let boosted_scores = scorer.get_scores();

        assert!(boosted_scores[&DocumentId(1)] > normal_scores[&DocumentId(1)]);
    }
}

*/
