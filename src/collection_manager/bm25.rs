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
use std::{collections::HashMap, fmt::Debug, hash::Hash};
use tracing::error;

use crate::types::FieldId;

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

/// Calculate IDF (Inverse Document Frequency) component using Lucene-style formula
///
/// # Arguments
///
/// * `total_documents` - N: total documents in the collection (corpus size)  
/// * `corpus_document_frequency` - df: documents that contain this term across any searched field
///
/// # Returns
///
/// * `f32` - IDF score using Lucene formula: ln(1 + (N - df + 0.5) / (df + 0.5))
/// This ensures positive scores for all term frequencies, avoiding negative scores for common terms
fn calculate_idf(total_documents: f32, corpus_document_frequency: usize) -> f32 {
    let df = corpus_document_frequency as f32;
    let ratio = (total_documents - df + 0.5) / (df + 0.5);
    ratio.ln_1p() // Lucene-style: always positive
}

/// Calculate normalized term frequency for a field in BM25F
///
/// This is tf'_{t,f} = tf_{t,f} / (1 − b_f + b_f · len_f/avglen_f)
///
/// # Arguments
///
/// * `term_occurrence_in_field` - Raw term frequency in the field
/// * `field_length` - Length of the field in the document
/// * `average_field_length` - Average field length in the collection
/// * `b` - Field-specific normalization parameter
///
/// # Returns
///
/// * `f32` - Normalized term frequency
fn bm25f_normalized_tf(
    term_occurrence_in_field: u32,
    field_length: u32,
    average_field_length: f32,
    b: f32,
) -> f32 {
    let tf = term_occurrence_in_field as f32;
    let len = field_length as f32;
    let avglen = average_field_length;

    tf / (1.0 - b + b * (len / avglen))
}

/// Complete BM25F score calculation using canonical formulation
///
/// # Arguments
///
/// * `aggregated_score` - S_t = Σ_f w_f · tf'_{t,f} (sum of weighted normalized tf across fields)
/// * `k` - k parameter (typically 1.2)
/// * `idf` - Corpus-level IDF score
///
/// # Returns
///
/// * `f32` - Final BM25F score for this term
fn bm25f_score(aggregated_score: f32, k: f32, idf: f32) -> f32 {
    idf * (k + 1.0) * aggregated_score / (k + aggregated_score)
}

/// Represents a field contribution to a term in BM25F scoring
#[derive(Debug, Clone)]
struct BM25FFieldContribution {
    normalized_tf: f32,
    weight: f32,
}

pub enum BM25Scorer<K: Eq + Hash> {
    Plain(BM25FScorerPlain<K>),
    WithThreshold(BM25FScorerWithThreshold<K>),
}

impl<K: Eq + Hash + Debug + Clone> BM25Scorer<K> {
    pub fn plain() -> Self {
        Self::Plain(BM25FScorerPlain::new())
    }

    pub fn with_threshold(threshold: u32) -> Self {
        Self::WithThreshold(BM25FScorerWithThreshold::new(threshold))
    }

    pub fn next_term(&mut self) {
        match self {
            Self::Plain(scorer) => scorer.next_term(),
            Self::WithThreshold(scorer) => scorer.next_term(),
        }
    }

    pub fn reset_term(&mut self) {
        match self {
            Self::Plain(scorer) => scorer.reset_term(),
            Self::WithThreshold(scorer) => scorer.reset_term(),
        }
    }

    /// Add a field contribution for the current term
    #[allow(clippy::too_many_arguments)]
    pub fn add_field(
        &mut self,
        key: K,
        _field_id: FieldId,
        term_occurrence_in_field: u32,
        field_length: u32,
        average_field_length: f32,
        field_params: &BM25FFieldParams,
    ) {
        let normalized_tf = bm25f_normalized_tf(
            term_occurrence_in_field,
            field_length,
            average_field_length,
            field_params.b,
        );

        match self {
            Self::Plain(scorer) => {
                scorer.add_field(key, _field_id, normalized_tf, field_params.weight)
            }
            Self::WithThreshold(scorer) => {
                scorer.add_field(key, _field_id, normalized_tf, field_params.weight)
            }
        }
    }

    /// Finalize the current term and compute BM25F scores
    pub fn finalize_term(
        &mut self,
        corpus_term_frequency: usize,
        total_documents: f32,
        k: f32,
        phrase_boost: f32,
        token_indexes: u32,
    ) {
        match self {
            Self::Plain(scorer) => {
                scorer.finalize_term(corpus_term_frequency, total_documents, k, phrase_boost)
            }
            Self::WithThreshold(scorer) => scorer.finalize_term(
                corpus_term_frequency,
                total_documents,
                k,
                phrase_boost,
                token_indexes,
            ),
        }
    }

    /// Finalize term for plain scorer (no token tracking needed)
    pub fn finalize_term_plain(
        &mut self,
        corpus_term_frequency: usize,
        total_documents: f32,
        k: f32,
        phrase_boost: f32,
    ) {
        match self {
            Self::Plain(scorer) => {
                scorer.finalize_term(corpus_term_frequency, total_documents, k, phrase_boost)
            }
            Self::WithThreshold(_) => {
                panic!("Use finalize_term with token_indexes for threshold scorer")
            }
        }
    }

    /// Legacy method for backward compatibility - approximates canonical BM25F per field
    ///
    /// Note: This doesn't provide true BM25F benefits since it processes one field at a time,
    /// but it's closer to canonical BM25F than the old approach of applying field weight as
    /// an external multiplier after BM25 calculation.
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        key: K,
        _field_id: FieldId,
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
        // Calculate normalized tf using canonical BM25F approach
        let normalized_tf = bm25f_normalized_tf(
            term_occurrence_in_field,
            field_length,
            average_field_length,
            field_params.b,
        );

        // Apply field weight (this is where user boost should be integrated)
        let weighted_tf = field_params.weight * normalized_tf;

        // Calculate corpus-level IDF using canonical formula
        let idf = calculate_idf(
            total_documents_with_field,
            total_documents_with_term_in_field,
        );

        // Apply BM25F saturation to the weighted tf
        let term_score = bm25f_score(weighted_tf, k, idf);

        if !term_score.is_nan() {
            let final_score = term_score * boost; // phrase boost

            match self {
                Self::Plain(scorer) => {
                    *scorer.document_scores.entry(key).or_insert(0.0) += final_score;
                }
                Self::WithThreshold(scorer) => {
                    *scorer.document_scores.entry(key.clone()).or_insert(0.0) += final_score;

                    let token_mask = if token_indexes > 0 {
                        token_indexes
                    } else {
                        1 << scorer.term_index
                    };
                    *scorer.document_token_masks.entry(key).or_insert(0) |= token_mask;
                }
            }
        } else if term_score.is_nan() {
            error!(
                ?weighted_tf,
                ?k,
                ?idf,
                "BM25F score is NaN in legacy add method. Skipping term contribution"
            );
        }
    }

    /// Get the number of distinct documents that have contributions for the current term
    pub fn current_term_document_count(&self) -> usize {
        match self {
            Self::Plain(scorer) => scorer.current_term_contributions.len(),
            Self::WithThreshold(scorer) => scorer.current_term_contributions.len(),
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
    // Document scores and token tracking
    document_scores: HashMap<K, f32>,
    document_token_masks: HashMap<K, u32>,
    // Current term data being accumulated
    current_term_contributions: HashMap<K, Vec<BM25FFieldContribution>>,
}
impl<K: Eq + Hash + Debug + Clone> BM25FScorerWithThreshold<K> {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            term_index: 0,
            document_scores: HashMap::new(),
            document_token_masks: HashMap::new(),
            current_term_contributions: HashMap::new(),
        }
    }

    pub fn next_term(&mut self) {
        self.term_index += 1;
        self.current_term_contributions.clear();
    }

    pub fn reset_term(&mut self) {
        self.term_index = 0;
        self.current_term_contributions.clear();
    }

    /// Add a field contribution for the current term
    pub fn add_field(&mut self, key: K, _field_id: FieldId, normalized_tf: f32, weight: f32) {
        let contribution = BM25FFieldContribution {
            normalized_tf,
            weight,
        };

        self.current_term_contributions
            .entry(key)
            .or_default()
            .push(contribution);
    }

    /// Finalize the current term using canonical BM25F
    pub fn finalize_term(
        &mut self,
        corpus_term_frequency: usize,
        total_documents: f32,
        k: f32,
        phrase_boost: f32,
        token_indexes: u32,
    ) {
        // Calculate corpus-level IDF
        let idf = calculate_idf(total_documents, corpus_term_frequency);

        for (key, contributions) in self.current_term_contributions.drain() {
            // Calculate S_t = Σ_f w_f · tf'_{t,f} (canonical BM25F aggregation)
            let aggregated_score: f32 = contributions
                .iter()
                .map(|contrib| contrib.weight * contrib.normalized_tf)
                .sum();

            if aggregated_score.is_normal() {
                // Apply canonical BM25F saturation: score_t = idf(t) · (k + 1) · S_t / (k + S_t)
                let term_score = bm25f_score(aggregated_score, k, idf);

                if !term_score.is_nan() {
                    let final_score = term_score * phrase_boost;

                    // Add to document's total score
                    *self.document_scores.entry(key.clone()).or_insert(0.0) += final_score;

                    // Update token mask
                    let token_mask = if token_indexes > 0 {
                        token_indexes
                    } else {
                        1 << self.term_index
                    };
                    *self.document_token_masks.entry(key).or_insert(0) |= token_mask;
                } else {
                    error!(
                        ?aggregated_score,
                        ?k,
                        ?idf,
                        "BM25F score is NaN. Skipping term contribution"
                    );
                }
            }
        }
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.document_scores
            .into_iter()
            .filter_map(|(key, score)| {
                let token_count = self.document_token_masks.get(&key)?.count_ones();
                if token_count >= self.threshold {
                    Some((key, score))
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct BM25FScorerPlain<K: Eq + Hash> {
    // Document scores
    document_scores: HashMap<K, f32>,
    // Current term data being accumulated
    current_term_contributions: HashMap<K, Vec<BM25FFieldContribution>>,
}

impl<K: Eq + Hash + Debug + Clone> Default for BM25FScorerPlain<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Debug + Clone> BM25FScorerPlain<K> {
    pub fn new() -> Self {
        Self {
            document_scores: HashMap::new(),
            current_term_contributions: HashMap::new(),
        }
    }

    pub fn next_term(&mut self) {
        self.current_term_contributions.clear();
    }

    pub fn reset_term(&mut self) {
        self.current_term_contributions.clear();
    }

    /// Add a field contribution for the current term
    pub fn add_field(&mut self, key: K, _field_id: FieldId, normalized_tf: f32, weight: f32) {
        let contribution = BM25FFieldContribution {
            normalized_tf,
            weight,
        };

        self.current_term_contributions
            .entry(key)
            .or_default()
            .push(contribution);
    }

    /// Finalize the current term using canonical BM25F
    pub fn finalize_term(
        &mut self,
        corpus_term_frequency: usize,
        total_documents: f32,
        k: f32,
        phrase_boost: f32,
    ) {
        // Calculate corpus-level IDF
        let idf = calculate_idf(total_documents, corpus_term_frequency);

        for (key, contributions) in self.current_term_contributions.drain() {
            // Calculate S_t = Σ_f w_f · tf'_{t,f} (canonical BM25F aggregation)
            let aggregated_score: f32 = contributions
                .iter()
                .map(|contrib| contrib.weight * contrib.normalized_tf)
                .sum();

            if aggregated_score.is_normal() {
                // Apply canonical BM25F saturation: score_t = idf(t) · (k + 1) · S_t / (k + S_t)
                let term_score = bm25f_score(aggregated_score, k, idf);

                if !term_score.is_nan() {
                    let final_score = term_score * phrase_boost;

                    // Add to document's total score
                    *self.document_scores.entry(key).or_insert(0.0) += final_score;
                } else {
                    error!(
                        ?aggregated_score,
                        ?k,
                        ?idf,
                        "BM25F score is NaN. Skipping term contribution"
                    );
                }
            }
        }
    }

    pub fn get_scores(self) -> HashMap<K, f32> {
        self.document_scores
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
            5,     // term_occurrence_in_field
            100,   // field_length
            100.0, // average_field_length
            100.0, // total_documents_with_field
            10,    // total_documents_with_term_in_field (less than half for positive IDF)
            1.2,   // k parameter
            &field_params,
            1.0, // boost
            0,   // token_indexes
        );

        let scores = scorer.get_scores();
        assert_eq!(scores.len(), 1);
        // With Lucene IDF formula: ln(1 + (100 - 10 + 0.5) / (10 + 0.5)) ≈ 3.065
        // Expected score will be different from the original due to canonical IDF
        let ratio = (100.0_f32 - 10.0 + 0.5) / (10.0 + 0.5);
        let expected_idf = ratio.ln_1p(); // ≈ 3.065
        let normalized_tf = 5.0; // tf / (1 - b + b * (len/avglen)) = 5/(1-0.75+0.75*1) = 5
        let expected_score = expected_idf * (1.2 + 1.0) * normalized_tf / (1.2 + normalized_tf);
        assert_approx_eq!(scores["doc1"], expected_score, 1e-6);
    }

    #[test]
    fn test_bm25f_scorer_boost() {
        let mut scorer = BM25Scorer::plain();
        let field_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.75,
        };

        // Use parameters that give positive IDF: 100 total docs, 10 with term
        let (total_docs, term_docs) = (100.0, 10);

        scorer.add(
            "doc1",
            FieldId(0),
            5,
            100,
            100.0,
            total_docs,
            term_docs,
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
            total_docs,
            term_docs,
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
            total_docs,
            term_docs,
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

        let field1_params = BM25FFieldParams {
            weight: 2.0,
            b: 0.75,
        };
        let field2_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.75,
        };

        let (total_docs, term_docs) = (100.0, 10);

        scorer.add(
            "doc1",
            FieldId(1),
            5,
            100,
            100.0,
            total_docs,
            term_docs,
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
            total_docs,
            term_docs,
            1.2,
            &field2_params,
            1.0,
            0,
        );

        let scores = scorer.get_scores();
        assert_eq!(scores.len(), 1);

        let ratio = (100.0_f32 - 10.0 + 0.5) / (10.0 + 0.5);
        let expected_idf = ratio.ln_1p();
        let normalized_tf = 5.0;
        let expected_single_field_score =
            expected_idf * (1.2 + 1.0) * normalized_tf / (1.2 + normalized_tf);

        assert!(scores["doc1"] > expected_single_field_score);
    }

    #[test]
    fn test_bm25f_field_normalization() {
        let mut scorer = BM25Scorer::plain();

        let low_b_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.2,
        };
        let high_b_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.9,
        };

        let (total_docs, term_docs) = (100.0, 10);

        scorer.add(
            "doc1",
            FieldId(0),
            5,
            200,
            100.0,
            total_docs,
            term_docs,
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
            total_docs,
            term_docs,
            1.2,
            &high_b_params,
            1.0,
            0,
        );

        let scores = scorer.get_scores();
        assert!(scores["doc1"] > scores["doc2"]);
    }

    #[test]
    fn test_bm25f_boost_integration_single_field() {
        let mut scorer_no_boost = BM25Scorer::plain();
        let mut scorer_with_boost = BM25Scorer::plain();

        // Test same document/field/term with different boost values
        let no_boost_params = BM25FFieldParams {
            weight: 1.0, // No boost
            b: 0.75,
        };
        let with_boost_params = BM25FFieldParams {
            weight: 2.0, // 2x boost
            b: 0.75,
        };

        // Same document, same conditions
        let field_id = FieldId(0);
        let term_occurrence = 5_u32;
        let field_length = 100_u32;
        let average_field_length = 100.0_f32;
        let total_documents = 100.0_f32;
        let term_documents = 10_usize;
        let k = 1.2_f32;

        scorer_no_boost.add(
            "doc1",
            field_id,
            term_occurrence,
            field_length,
            average_field_length,
            total_documents,
            term_documents,
            k,
            &no_boost_params,
            1.0, // No additional boost
            0,
        );

        scorer_with_boost.add(
            "doc1",
            field_id,
            term_occurrence,
            field_length,
            average_field_length,
            total_documents,
            term_documents,
            k,
            &with_boost_params,
            1.0, // No additional boost (boost is in field weight)
            0,
        );

        let no_boost_scores = scorer_no_boost.get_scores();
        let with_boost_scores = scorer_with_boost.get_scores();

        // Boosted field should score higher
        assert!(with_boost_scores["doc1"] > no_boost_scores["doc1"]);

        // The boost should be significant (not just a tiny difference)
        let score_ratio = with_boost_scores["doc1"] / no_boost_scores["doc1"];
        println!("Score ratio (2x boost): {:.6}", score_ratio);
        println!("No boost score: {:.6}", no_boost_scores["doc1"]);
        println!("With boost score: {:.6}", with_boost_scores["doc1"]);
        assert!(
            score_ratio > 1.05,
            "Expected ratio > 1.05, got {:.6}",
            score_ratio
        ); // Should be meaningfully higher due to 2x boost (canonical BM25F has diminishing returns)
    }

    #[test]
    fn test_bm25f_boost_integration_multi_field() {
        let mut scorer = BM25Scorer::plain();

        // Test document with multiple fields having different boost values
        let title_params = BM25FFieldParams {
            weight: 3.0, // High boost for title
            b: 0.75,
        };
        let content_params = BM25FFieldParams {
            weight: 1.0, // Standard boost for content
            b: 0.75,
        };
        let tags_params = BM25FFieldParams {
            weight: 1.5, // Medium boost for tags
            b: 0.75,
        };

        let title_field = FieldId(1);
        let content_field = FieldId(2);
        let tags_field = FieldId(3);

        // Add same term occurrence across different fields
        scorer.add(
            "doc1",
            title_field,
            3,  // term occurs 3 times in title
            50, // title is shorter
            50.0,
            100.0,
            10,
            1.2,
            &title_params,
            1.0,
            0,
        );

        scorer.add(
            "doc1",
            content_field,
            3,   // same term occurrence in content
            200, // content is longer
            200.0,
            100.0,
            10,
            1.2,
            &content_params,
            1.0,
            0,
        );

        scorer.add(
            "doc1",
            tags_field,
            3,  // same term occurrence in tags
            20, // tags are shortest
            20.0,
            100.0,
            10,
            1.2,
            &tags_params,
            1.0,
            0,
        );

        let scores = scorer.get_scores();

        // Document should have a combined score from all fields
        // The title field with highest boost should contribute most significantly
        assert!(scores["doc1"] > 0.0);

        // Test individual field contributions by running separate scorers
        let mut title_only = BM25Scorer::plain();
        title_only.add(
            "doc1",
            title_field,
            3,
            50,
            50.0,
            100.0,
            10,
            1.2,
            &title_params,
            1.0,
            0,
        );
        let title_score = title_only.get_scores()["doc1"];

        let mut content_only = BM25Scorer::plain();
        content_only.add(
            "doc1",
            content_field,
            3,
            200,
            200.0,
            100.0,
            10,
            1.2,
            &content_params,
            1.0,
            0,
        );
        let content_score = content_only.get_scores()["doc1"];

        // Title should score higher than content due to boost (3.0 vs 1.0)
        assert!(title_score > content_score);
    }

    #[test]
    fn test_bm25f_boost_values_comparison() {
        // Test different boost values to ensure they create the expected score differences
        let boost_values = [0.5, 1.0, 1.5, 2.0, 3.0];
        let mut scores = Vec::new();

        for &boost_value in &boost_values {
            let mut scorer = BM25Scorer::plain();
            let field_params = BM25FFieldParams {
                weight: boost_value,
                b: 0.75,
            };

            scorer.add(
                "doc1",
                FieldId(0),
                5,
                100,
                100.0,
                100.0,
                10,
                1.2,
                &field_params,
                1.0,
                0,
            );

            let score = scorer.get_scores()["doc1"];
            scores.push(score);
        }

        // Scores should increase with boost values
        for i in 1..scores.len() {
            assert!(
                scores[i] > scores[i - 1],
                "Score with boost {} ({}) should be higher than boost {} ({})",
                boost_values[i],
                scores[i],
                boost_values[i - 1],
                scores[i - 1]
            );
        }

        // Specific ratio checks - canonical BM25F has diminishing returns
        let ratio_1x_to_2x = scores[3] / scores[1]; // 2.0 boost vs 1.0 boost
        println!(
            "Boost comparison - 1.0 boost: {:.6}, 2.0 boost: {:.6}, ratio: {:.6}",
            scores[1], scores[3], ratio_1x_to_2x
        );
        assert!(
            ratio_1x_to_2x > 1.0 && ratio_1x_to_2x < 1.5,
            "2x boost should provide some increase but with diminishing returns in canonical BM25F, got ratio: {}",
            ratio_1x_to_2x
        );
    }

    #[test]
    fn test_canonical_bm25f_single_term_two_fields() {
        let mut scorer = BM25Scorer::plain();

        // Hand-calculated canonical BM25F example
        // Term appears in two fields with different lengths and weights
        let title_params = BM25FFieldParams {
            weight: 2.0, // Title has higher weight
            b: 0.75,
        };
        let content_params = BM25FFieldParams {
            weight: 1.0, // Content has standard weight
            b: 0.75,
        };

        let doc_key = "test_doc";
        let k = 1.2_f32;
        let corpus_docs = 100_f32;
        let term_docs = 10_usize;

        // Field 1: Title - short field, term appears twice
        let title_tf = 2_u32;
        let title_len = 10_u32;
        let title_avglen = 8.0_f32;

        // Field 2: Content - longer field, term appears once
        let content_tf = 1_u32;
        let content_len = 200_u32;
        let content_avglen = 150.0_f32;

        // Add field contributions for the same term
        scorer.add_field(
            doc_key,
            FieldId(1),
            title_tf,
            title_len,
            title_avglen,
            &title_params,
        );
        scorer.add_field(
            doc_key,
            FieldId(2),
            content_tf,
            content_len,
            content_avglen,
            &content_params,
        );

        // Finalize with canonical BM25F
        scorer.finalize_term_plain(term_docs, corpus_docs, k, 1.0);

        let scores = scorer.get_scores();
        let actual_score = scores[doc_key];

        // Hand calculate expected canonical BM25F score
        // tf'_{title} = tf / (1 - b + b * (len / avglen)) = 2 / (1 - 0.75 + 0.75 * (10/8)) = 2 / (0.25 + 0.9375) = 1.6842
        let title_normalized_tf = 2.0 / (1.0 - 0.75 + 0.75 * (10.0 / 8.0));
        // tf'_{content} = 1 / (1 - 0.75 + 0.75 * (200/150)) = 1 / (0.25 + 1.0) = 0.8
        let content_normalized_tf = 1.0 / (1.0 - 0.75 + 0.75 * (200.0 / 150.0));

        // S_t = w_title * tf'_{title} + w_content * tf'_{content} = 2.0 * 1.6842 + 1.0 * 0.8
        let aggregated_s = title_params.weight * title_normalized_tf
            + content_params.weight * content_normalized_tf;

        // IDF = ln(1 + (N - df + 0.5) / (df + 0.5)) = ln(1 + (100 - 10 + 0.5) / (10 + 0.5))
        let ratio = (corpus_docs - term_docs as f32 + 0.5) / (term_docs as f32 + 0.5);
        let idf = ratio.ln_1p();

        // Final score = idf * (k + 1) * S_t / (k + S_t)
        let expected_score = idf * (k + 1.0) * aggregated_s / (k + aggregated_s);

        assert_approx_eq!(actual_score, expected_score, 1e-5);

        println!("Canonical BM25F test:");
        println!("  Title tf': {:.6}", title_normalized_tf);
        println!("  Content tf': {:.6}", content_normalized_tf);
        println!("  Aggregated S_t: {:.6}", aggregated_s);
        println!("  IDF: {:.6}", idf);
        println!("  Expected score: {:.6}", expected_score);
        println!("  Actual score: {:.6}", actual_score);
    }

    #[test]
    fn test_canonical_bm25f_vs_sum_of_per_field_bm25() {
        let mut canonical_scorer = BM25Scorer::plain();

        // Field parameters
        let field1_params = BM25FFieldParams {
            weight: 2.0,
            b: 0.75,
        };
        let field2_params = BM25FFieldParams {
            weight: 1.0,
            b: 0.75,
        };

        let doc_key = "test_doc";
        let k = 1.2_f32;
        let corpus_docs = 100_f32;
        let term_docs = 10_usize;

        // Add contributions using canonical BM25F
        canonical_scorer.add_field(doc_key, FieldId(1), 3, 50, 40.0, &field1_params);
        canonical_scorer.add_field(doc_key, FieldId(2), 2, 100, 80.0, &field2_params);
        canonical_scorer.finalize_term_plain(term_docs, corpus_docs, k, 1.0);

        let canonical_score = canonical_scorer.get_scores()[doc_key];

        // Calculate what the sum of individual BM25 scores would be
        let field1_tf_normalized = 3.0 / (1.0 - 0.75 + 0.75 * (50.0 / 40.0));
        let field2_tf_normalized = 2.0 / (1.0 - 0.75 + 0.75 * (100.0 / 80.0));

        // Individual BM25 scores (what old approach would do)
        let idf = ((corpus_docs - term_docs as f32 + 0.5) / (term_docs as f32 + 0.5)).ln_1p();
        let field1_individual = field1_params.weight * idf * (k + 1.0) * field1_tf_normalized
            / (k + field1_tf_normalized);
        let field2_individual = field2_params.weight * idf * (k + 1.0) * field2_tf_normalized
            / (k + field2_tf_normalized);
        let sum_of_individual = field1_individual + field2_individual;

        // Canonical BM25F should be different (typically lower due to single saturation)
        println!("Canonical BM25F: {:.6}", canonical_score);
        println!("Sum of individual BM25: {:.6}", sum_of_individual);
        println!(
            "Ratio (canonical/sum): {:.6}",
            canonical_score / sum_of_individual
        );

        // The canonical score should be less than or equal to sum of individual scores
        // (this catches double-saturation issues)
        assert!(
            canonical_score <= sum_of_individual + 1e-6,
            "Canonical BM25F ({}) should be <= sum of individual BM25 scores ({})",
            canonical_score,
            sum_of_individual
        );

        // But it should still be a meaningful score
        assert!(
            canonical_score > 0.0,
            "Canonical BM25F should produce positive scores"
        );
    }
}
