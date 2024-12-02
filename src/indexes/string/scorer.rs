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

pub mod code {
    use std::collections::HashMap;

    use dashmap::DashMap;

    use crate::{
        document_storage::DocumentId,
        indexes::string::{GlobalInfo, Posting},
    };

    use super::Scorer;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
    struct Position(usize);

    #[derive(Debug)]
    struct DocumentLength(u16);

    #[derive(Debug, Default)]
    pub struct CodeScore {
        scores: DashMap<DocumentId, (DashMap<Position, usize>, f32, DocumentLength)>,
    }

    impl CodeScore {
        pub fn new() -> Self {
            Self {
                scores: DashMap::new(),
            }
        }

        #[inline]
        fn calculate_score(pos: DashMap<Position, usize>, _boost: f32, doc_length: u16) -> f32 {
            let mut foo: Vec<_> = pos.into_iter().map(|(p, v)| (p.0, v)).collect();

            if foo.is_empty() || doc_length == 0 {
                return 0.0;
            }

            foo.sort_by_key(|(p, _)| (*p as isize));

            let pos_len = foo.len();

            let mut score = 0.0;
            for i in 0..foo.len() {
                let before_before = if i > 1 { foo.get(i - 2) } else { None };
                let before = if i > 0 { foo.get(i - 1) } else { None };
                let current = foo.get(i).unwrap();
                let after = foo.get(i + 1);
                let after_after = foo.get(i + 2);

                let before_before_current_distance = match before_before {
                    Some((before_before_pos, _)) => current.0 - before_before_pos,
                    None => 0,
                };
                let before_current_distance = match before {
                    Some((before_pos, _)) => current.0 - before_pos,
                    None => 0,
                };
                let current_after_distance = match after {
                    Some((after_pos, _)) => after_pos - current.0,
                    None => 0,
                };
                let after_after_current_distance = match after_after {
                    Some((after_after_pos, _)) => after_after_pos - current.0,
                    None => 0,
                };

                let before_before_current = 1.0 / (before_before_current_distance as f32 + 1.0);
                let before_current = 1.0 / (before_current_distance as f32 + 1.0);
                let current_after = 1.0 / (current_after_distance as f32 + 1.0);
                let after_after_current = 1.0 / (after_after_current_distance as f32 + 1.0);

                let score_for_position =
                    before_before_current + before_current + current_after + after_after_current;

                score += score_for_position;
            }

            let denominator = (pos_len * (doc_length as usize)) as f32;
            if denominator == 0.0 {
                0.0
            } else {
                score / denominator
            }
        }
    }

    impl Scorer for CodeScore {
        #[inline]
        fn add_entry(
            &self,
            _global_info: &GlobalInfo,
            posting: Posting,
            _total_token_count: f32,
            boost_per_field: f32,
        ) {
            let document_id = posting.document_id;
            let previous = self.scores.entry(document_id).or_insert_with(|| {
                (
                    DashMap::new(),
                    boost_per_field,
                    DocumentLength(posting.doc_length),
                )
            });

            for position in posting.positions {
                previous
                    .0
                    .entry(Position(position))
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
            }
        }

        fn get_scores(self) -> HashMap<DocumentId, f32> {
            self.scores
                .into_iter()
                // .par_bridge()
                .map(|(document_id, (pos, boost, doc_length))| {
                    let score = Self::calculate_score(pos, boost, doc_length.0);
                    (document_id, score)
                })
                .collect()
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

    #[test]
    fn test_code_score_basic() {
        let scorer = code::CodeScore::new();
        let global_info = GlobalInfo {
            total_documents: 10,
            total_document_length: 1000,
        };

        let posting = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            term_frequency: 3.0,
            doc_length: 100,
            positions: vec![1, 3, 5],
        };

        scorer.add_entry(&global_info, posting, 1.0, 1.0);
        let scores = scorer.get_scores();
        assert!(scores.contains_key(&DocumentId(1)));
        assert!(scores[&DocumentId(1)] > 0.0);
    }

    #[test]
    fn test_code_score_adjacent_positions() {
        let scorer = code::CodeScore::new();
        let global_info = GlobalInfo {
            total_documents: 10,
            total_document_length: 1000,
        };

        let posting_adjacent = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            term_frequency: 3.0,
            doc_length: 100,
            positions: vec![1, 2, 3],
        };

        let posting_spread = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(2),
            term_frequency: 3.0,
            doc_length: 100,
            positions: vec![1, 10, 20],
        };

        scorer.add_entry(&global_info, posting_adjacent, 1.0, 1.0);
        scorer.add_entry(&global_info, posting_spread, 1.0, 1.0);
        let scores = scorer.get_scores();

        assert!(scores[&DocumentId(1)] > scores[&DocumentId(2)]);
    }

    #[test]
    fn test_code_score_empty_positions() {
        let scorer = code::CodeScore::new();
        let global_info = GlobalInfo {
            total_documents: 1,
            total_document_length: 100,
        };

        let posting = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            term_frequency: 0.0,
            doc_length: 100,
            positions: vec![],
        };

        scorer.add_entry(&global_info, posting, 1.0, 1.0);
        let scores = scorer.get_scores();
        assert_eq!(scores[&DocumentId(1)], 0.0);
    }

    #[test]
    fn test_code_score_single_position() {
        let scorer = code::CodeScore::new();
        let global_info = GlobalInfo {
            total_documents: 1,
            total_document_length: 100,
        };

        let posting = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            term_frequency: 1.0,
            doc_length: 100,
            positions: vec![1],
        };

        scorer.add_entry(&global_info, posting, 1.0, 1.0);
        let scores = scorer.get_scores();
        assert!(scores[&DocumentId(1)] > 0.0);
    }

    #[test]
    fn test_multiple_entries_same_document() {
        let scorer = code::CodeScore::new();
        let global_info = GlobalInfo {
            total_documents: 1,
            total_document_length: 100,
        };

        let posting1 = Posting {
            field_id: FieldId(1),
            document_id: DocumentId(1),
            term_frequency: 2.0,
            doc_length: 100,
            positions: vec![1, 2],
        };

        let posting2 = Posting {
            field_id: FieldId(2),
            document_id: DocumentId(1),
            term_frequency: 2.0,
            doc_length: 100,
            positions: vec![3, 4],
        };

        scorer.add_entry(&global_info, posting1, 1.0, 1.0);
        scorer.add_entry(&global_info, posting2, 1.0, 1.0);

        let scores = scorer.get_scores();
        assert!(scores[&DocumentId(1)] > 0.0);
    }
}
