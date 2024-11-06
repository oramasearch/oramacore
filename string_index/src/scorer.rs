use std::collections::HashMap;

use types::DocumentId;

use crate::{GlobalInfo, Posting};

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
    use types::DocumentId;

    use super::Scorer;

    #[derive(Debug, Default)]
    pub struct BM25Score {
        scores: DashMap<DocumentId, f32>,
    }
    impl BM25Score {
        fn new() -> Self {
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
            let k1 = 1.5;
            let b = 0.75;
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length));
            idf * (numerator / denominator) * boost
        }
    }

    impl Scorer for BM25Score {
        #[inline]
        fn add_entry(
            &self,
            _global_info: &crate::GlobalInfo,
            posting: crate::Posting,
            _total_token_count: f32,
            _boost_per_field: f32,
        ) {
            let term_frequency = posting.term_frequency;
            let doc_length = posting.doc_length as f32;
            let freq = 1.0;

            let total_documents = 1.0;
            let avg_doc_length = total_documents / 1.0;

            let idf = ((total_documents - freq + 0.5_f32) / (freq + 0.5_f32)).ln_1p();
            let score = Self::calculate_score(
                term_frequency,
                idf,
                doc_length,
                avg_doc_length,
                _boost_per_field,
            );

            let mut previous = self.scores.entry(posting.document_id)
                .or_insert(0.0);
            *previous += score;
        }

        fn get_scores(self) -> HashMap<types::DocumentId, f32> {
            self.scores.into_iter().collect()
        }
    }
}

pub mod code {
    use std::collections::HashMap;

    use dashmap::DashMap;
    use rayon::iter::{ParallelBridge, ParallelIterator};
    use types::DocumentId;

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
        fn new() -> Self {
            Self {
                scores: DashMap::new(),
            }
        }

        fn calculate_score(pos: DashMap<Position, usize>, boost: f32, doc_lenth: u16) -> f32 {
            let mut foo: Vec<_> = pos.into_iter().map(|(p, v)| (p.0, v)).collect();
            foo.sort_by_key(|(p, _) | (*p as isize));

            let pos_len = foo.len();

            let mut score = 0.0;
            for i in 0..foo.len() {
                let before_before = if i > 1 {
                    foo.get(i - 2)
                } else {
                    None
                };
                let before = if i > 0 {
                    foo.get(i - 1)
                } else {
                    None
                };
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

                let score_for_position = before_before_current + before_current + current_after + after_after_current;

                score += score_for_position;
            }

            score / (pos_len * (doc_lenth as usize)) as f32
        }
    }

    impl Scorer for CodeScore {
        #[inline]
        fn add_entry(
            &self,
            _global_info: &crate::GlobalInfo,
            posting: crate::Posting,
            _total_token_count: f32,
            boost_per_field: f32,
        ) {
            let document_id = posting.document_id;
            let previous = self.scores.entry(document_id)
                .or_insert_with(|| (DashMap::new(), boost_per_field, DocumentLength(posting.doc_length)));

            for position in posting.positions {
                previous.0.entry(Position(position))
                    .and_modify(|e| *e += 1)
                    .or_insert(1);
            }
        }

        fn get_scores(self) -> HashMap<types::DocumentId, f32> {
            self.scores.into_iter()
                // .par_bridge()
                .map(|(document_id, (pos, boost, doc_length))| {
                    let score = Self::calculate_score(pos, boost, doc_length.0);
                    (document_id, score)
                })
                .collect()
        }
    }
}
