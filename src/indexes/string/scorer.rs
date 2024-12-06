use std::collections::HashMap;

use crate::document_storage::DocumentId;

use super::{GlobalInfo, Posting};

pub trait Scorer {
    fn add_entry(
        &self,
        global_info: &GlobalInfo,
        posting: Posting,
        total_token_count: f32,
        boost_per_field: f32,
    );
    fn get_scores(self) -> HashMap<DocumentId, f32>;
}

pub mod bm25 {
    use std::collections::HashMap;

    use dashmap::DashMap;

    use crate::{
        document_storage::DocumentId,
        indexes::string::{GlobalInfo, Posting},
    };

    use super::Scorer;

    #[derive(Debug, Default)]
    pub struct BM25Score {
        scores: DashMap<DocumentId, f32>,
    }
    impl BM25Score {
        pub fn new() -> Self {
            Self {
                scores: DashMap::new(),
            }
        }

        #[inline]
        fn calculate_score(
            tf: f32,
            idf: f32,
            doc_length: f32,
            avg_doc_length: f32,
            boost: f32,
        ) -> f32 {
            if tf == 0.0 || doc_length == 0.0 || avg_doc_length == 0.0 {
                return 0.0;
            }

            let k1 = 1.5;
            let b = 0.75;
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length));

            // @todo: find a better way to avoid division by 0
            if denominator == 0.0 {
                return 0.0;
            }

            idf * (numerator / denominator) * boost
        }
    }

    impl Scorer for BM25Score {
        #[inline]
        fn add_entry(
            &self,
            global_info: &GlobalInfo,
            posting: Posting,
            total_token_count: f32,
            boost_per_field: f32,
        ) {
            let term_frequency = posting.term_frequency;
            let doc_length = posting.doc_length as f32;
            let total_documents = global_info.total_documents as f32;

            if total_documents == 0.0 {
                self.scores.insert(posting.document_id, 0.0);
                return;
            }

            let avg_doc_length = global_info.total_document_length as f32 / total_documents;
            let idf =
                ((total_documents - total_token_count + 0.5) / (total_token_count + 0.5)).ln_1p();

            let score = Self::calculate_score(
                term_frequency,
                idf,
                doc_length,
                avg_doc_length,
                boost_per_field,
            );

            self.scores
                .entry(posting.document_id)
                .and_modify(|e| *e += score)
                .or_insert(score);
        }

        fn get_scores(self) -> HashMap<DocumentId, f32> {
            self.scores.into_iter().collect()
        }
    }
}

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
            term_frequency: 5.0,
            doc_length: 100,
            positions: vec![1, 2, 3, 4, 5],
        };

        scorer.add_entry(&global_info, posting, 1.0, 1.0);
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
            term_frequency: 0.0,
            doc_length: 0,
            positions: vec![],
        };

        scorer.add_entry(&global_info, posting, 1.0, 1.0);
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
            term_frequency: 5.0,
            doc_length: 100,
            positions: vec![1, 2, 3, 4, 5],
        };

        // Test with different boost values
        scorer.add_entry(&global_info, posting.clone(), 1.0, 1.0);
        let normal_scores = scorer.get_scores();

        let scorer = bm25::BM25Score::new();
        scorer.add_entry(&global_info, posting, 1.0, 2.0);
        let boosted_scores = scorer.get_scores();

        assert!(boosted_scores[&DocumentId(1)] > normal_scores[&DocumentId(1)]);
    }
}
