/** The structure of data needed for BM25F scoring:
 *
 * BM25F extends BM25 to handle multiple fields with field-specific weights and normalization.
 * Each field contributes to the overall document score based on its weight and normalization factor.
 *
 * ```text
 * (coll_id, field_id) => {
 *    average_field_length: f32,
 *    total_documents_with_field: usize,
 *    field_weight: f32,        // Field-specific weight (boost)
 *    field_b: f32,             // Field-specific normalization parameter
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
 * (coll_id, field_id) => [average_field_length, total_documents_with_field, field_weight, field_b]
 *
 * (coll_id, field_id, term) => [total_documents_with_term_in_field]
 *
 * (coll_id, field_id, doc_id) => [document_length]
 *
 * (coll_id, field_id, doc_id, term) => [term_occurrence_in_document]
 * ```
 */
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
};
use tracing::error;

use crate::types::FieldId;

/// Create a consistent FieldId from a field path for BM25F scoring
/// This allows backward compatibility with existing code that uses field_path
pub fn field_path_to_field_id(field_path: &[String]) -> FieldId {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    for segment in field_path {
        segment.hash(&mut hasher);
    }
    let hash = hasher.finish();
    // Use lower 16 bits for FieldId(u16)
    FieldId((hash & 0xFFFF) as u16)
}

/// BM25F field parameters for scoring
///
/// Contains field-specific parameters that allow BM25F to weight
/// and normalize different fields independently
#[derive(Debug, Clone, PartialEq)]
pub struct BM25FFieldParams {
    /// Field-specific weight (boost factor)
    pub weight: f32,
    /// Field-specific normalization parameter (typically between 0.0 and 1.0)
    pub b: f32,
}

impl Default for BM25FFieldParams {
    fn default() -> Self {
        Self {
            weight: 1.0,
            b: 0.75,
        }
    }
}

/// BM25F scoring function for a single field
///
/// # Arguments
///
/// * `term_occurrence_in_document` - occurrence of the term in the field in the document
/// * `document_length` - length of the field of the document in words
/// * `average_field_length` - average field length in the collection
/// * `total_documents_with_field` - number of documents that has that field in the collection
/// * `total_documents_with_term_in_field` - number of documents that has that term in the field in the collection
/// * `k` - k parameter (shared across fields)
/// * `field_params` - field-specific parameters (weight and b)
///
/// # Returns
///
/// * `f32` - BM25F field score (before IDF multiplication)
fn bm25f_field_score(
    term_occurrence_in_document: usize,
    document_length: u32,
    average_field_length: f32,
    k: f32,
    field_params: &BM25FFieldParams,
) -> f32 {
    let f = term_occurrence_in_document as f32;
    let l = document_length as f32;
    let avgdl = average_field_length;

    // BM25F field-specific normalization
    let normalization = 1.0 - field_params.b + field_params.b * (l / avgdl);

    // Apply field weight and normalization
    field_params.weight * (f * (k + 1.0)) / (f + k * normalization)
}

/// Calculate IDF (Inverse Document Frequency) component
///
/// # Arguments
///
/// * `total_documents_with_field` - total documents in collection that have this field
/// * `total_documents_with_term_in_field` - documents that contain this term in this field
///
/// # Returns
///
/// * `f32` - IDF score
fn calculate_idf(
    total_documents_with_field: f32,
    total_documents_with_term_in_field: usize,
) -> f32 {
    let ni = total_documents_with_term_in_field as f32;
    ((total_documents_with_field - ni + 0.5_f32) / (ni + 0.5_f32)).ln_1p()
}

/// Complete BM25F score calculation combining field score and IDF
///
/// # Arguments
///
/// * `field_score` - Pre-calculated field score from bm25f_field_score
/// * `idf` - Pre-calculated IDF score
///
/// # Returns
///
/// * `f32` - Final BM25F score
fn bm25f_score(field_score: f32, idf: f32) -> f32 {
    idf * field_score
}

pub enum BM25Scorer<K: Eq + Hash> {
    Plain(BM25FScorerPlain<K>),
    WithThreshold(BM25FScorerWithThreshold<K>),
}
impl<K: Eq + Hash + Debug> BM25Scorer<K> {
    pub fn plain() -> Self {
        Self::Plain(BM25FScorerPlain::new())
    }

    pub fn with_threshold(threshold: u32) -> Self {
        Self::WithThreshold(BM25FScorerWithThreshold {
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
        field_id: FieldId,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        field_params: &BM25FFieldParams,
        boost: f32,
        token_indexes: u32,
    ) {
        match self {
            Self::Plain(scorer) => scorer.add(
                key,
                field_id,
                term_occurrence_in_field,
                field_length,
                average_field_length,
                total_documents_with_field,
                total_documents_with_term_in_field,
                k,
                field_params,
                boost,
            ),
            Self::WithThreshold(scorer) => {
                scorer.add(
                    key,
                    field_id,
                    term_occurrence_in_field,
                    field_length,
                    average_field_length,
                    total_documents_with_field,
                    total_documents_with_term_in_field,
                    k,
                    field_params,
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

pub struct BM25FScorerWithThreshold<K: Eq + Hash> {
    threshold: u32,
    term_index: usize,
    // Track field-specific scores: (token_indexes, field_scores_sum)
    scores: HashMap<K, (u32, HashMap<FieldId, f32>)>,
}
impl<K: Eq + Hash + Debug> BM25FScorerWithThreshold<K> {
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
        field_id: FieldId,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        field_params: &BM25FFieldParams,
        boost: f32,
        token_indexes: u32,
    ) {
        // Calculate field score component
        let field_score = bm25f_field_score(
            term_occurrence_in_field as usize,
            field_length,
            average_field_length,
            k,
            field_params,
        );

        // Calculate IDF component
        let idf = calculate_idf(
            total_documents_with_field,
            total_documents_with_term_in_field,
        );

        // Combine field score and IDF
        let score = bm25f_score(field_score, idf);

        if score.is_nan() {
            error!(
                ?term_occurrence_in_field,
                ?field_length,
                ?average_field_length,
                ?total_documents_with_field,
                ?total_documents_with_term_in_field,
                ?k,
                ?field_params,
                "BM25F score is NaN. Skipping item"
            );
            return;
        }

        let final_score = score * boost;

        let entry = self
            .scores
            .entry(key)
            .or_insert_with(|| (0, HashMap::new()));

        // Update token indexes
        entry.0 |= if token_indexes > 0 {
            token_indexes
        } else {
            1 << self.term_index
        };

        // Accumulate field-specific scores
        *entry.1.entry(field_id).or_insert(0.0) += final_score;
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.scores
            .into_iter()
            .filter_map(|(key, (count, field_scores))| {
                if count.count_ones() >= self.threshold {
                    // Sum all field scores for this document
                    let total_score: f32 = field_scores.values().sum();
                    Some((key, total_score))
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct BM25FScorerPlain<K: Eq + Hash> {
    // Track field-specific scores for proper BM25F combination
    scores: HashMap<K, HashMap<FieldId, f32>>,
}

impl<K: Eq + Hash + Debug> Default for BM25FScorerPlain<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Debug> BM25FScorerPlain<K> {
    pub fn new() -> Self {
        Self {
            scores: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        key: K,
        field_id: FieldId,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        total_documents_with_field: f32,
        total_documents_with_term_in_field: usize,
        k: f32,
        field_params: &BM25FFieldParams,
        boost: f32,
    ) {
        // Calculate field score component
        let field_score = bm25f_field_score(
            term_occurrence_in_field as usize,
            field_length,
            average_field_length,
            k,
            field_params,
        );

        // Calculate IDF component
        let idf = calculate_idf(
            total_documents_with_field,
            total_documents_with_term_in_field,
        );

        // Combine field score and IDF
        let score = bm25f_score(field_score, idf);

        if score.is_nan() {
            error!(
                ?term_occurrence_in_field,
                ?field_length,
                ?average_field_length,
                ?total_documents_with_field,
                ?total_documents_with_term_in_field,
                ?k,
                ?field_params,
                "BM25F score is NaN. Skipping item"
            );
            return;
        }

        let final_score = score * boost;

        // Accumulate field-specific scores properly
        let field_scores = self.scores.entry(key).or_default();
        *field_scores.entry(field_id).or_insert(0.0) += final_score;
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.scores
            .into_iter()
            .map(|(key, field_scores)| {
                // Sum all field scores for this document
                let total_score: f32 = field_scores.values().sum();
                (key, total_score)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_bm25f_scorer_basic() {
        let mut scorer = BM25Scorer::plain();
        let field_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.75,
        };

        scorer.add(
            "doc1",
            FieldId(0),
            5,
            100,
            100.0,
            10.0,
            5,
            1.2,
            &field_params,
            1.0,
            0,
        );

        let scores = scorer.get_scores();
        assert_eq!(scores.len(), 1);
        // Expected score should be similar to original BM25 since we're using default params
        assert_approx_eq!(scores["doc1"], 1.2297773, 1e-6);
    }

    #[test]
    fn test_bm25f_scorer_boost() {
        let mut scorer = BM25Scorer::plain();
        let field_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.75,
        };

        scorer.add(
            "doc1",
            FieldId(0),
            5,
            100,
            100.0,
            10.0,
            5,
            1.2,
            &field_params,
            1.0,
            0,
        );
        scorer.add(
            "doc2",
            FieldId(0),
            5,
            100,
            100.0,
            10.0,
            5,
            1.2,
            &field_params,
            2.0,
            0,
        );
        scorer.add(
            "doc3",
            FieldId(0),
            5,
            100,
            100.0,
            10.0,
            5,
            1.2,
            &field_params,
            0.5,
            0,
        );
        let scores = scorer.get_scores();

        assert!(scores["doc2"] > scores["doc1"]);
        assert!(scores["doc3"] < scores["doc1"]);
    }

    #[test]
    fn test_bm25f_field_weights() {
        let mut scorer = BM25Scorer::plain();

        // Different field weights
        let field1_params = BM25FFieldParams {
            weight: 2.0, // Higher weight for field 1
            b: 0.75,
        };
        let field2_params = BM25FFieldParams {
            weight: 1.0, // Standard weight for field 2
            b: 0.75,
        };

        // Same document, different fields, same term
        scorer.add(
            "doc1",
            FieldId(1),
            5,
            100,
            100.0,
            10.0,
            5,
            1.2,
            &field1_params,
            1.0,
            0,
        );
        scorer.add(
            "doc1",
            FieldId(2),
            5,
            100,
            100.0,
            10.0,
            5,
            1.2,
            &field2_params,
            1.0,
            0,
        );

        let scores = scorer.get_scores();
        assert_eq!(scores.len(), 1);

        // Score should be higher than a single field due to field combination
        let expected_single_field_score = 1.2297773;
        assert!(scores["doc1"] > expected_single_field_score);
    }

    #[test]
    fn test_bm25f_field_normalization() {
        let mut scorer = BM25Scorer::plain();

        // Different normalization parameters
        let low_b_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.2, // Less length normalization
        };
        let high_b_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.9, // More length normalization
        };

        // Test with longer document
        scorer.add(
            "doc1",
            FieldId(0),
            5,
            200,
            100.0,
            10.0,
            5,
            1.2,
            &low_b_params,
            1.0,
            0,
        );
        scorer.add(
            "doc2",
            FieldId(0),
            5,
            200,
            100.0,
            10.0,
            5,
            1.2,
            &high_b_params,
            1.0,
            0,
        );

        let scores = scorer.get_scores();

        // Document with lower b should score higher (less penalized for length)
        assert!(scores["doc1"] > scores["doc2"]);
    }
}
