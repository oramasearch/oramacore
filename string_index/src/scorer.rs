use crate::{GlobalInfo, Posting};

pub trait Scorer {
    fn update_score(
        &self,
        previous: &mut f32,
        global_info: &GlobalInfo,
        posting: Posting,
        total_token_count: f32,
        boost_per_field: f32,
    );
}

pub mod bm25 {
    use super::Scorer;

    pub struct BM25Score;
    impl BM25Score {
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
        fn update_score(
            &self,
            previous: &mut f32,
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

            *previous += score;
        }
    }
}

pub mod counter {
    use super::Scorer;

    pub struct CounterScore;
    impl Scorer for CounterScore {
        #[inline]
        fn update_score(
            &self,
            previous: &mut f32,
            _global_info: &crate::GlobalInfo,
            posting: crate::Posting,
            _total_token_count: f32,
            boost_per_field: f32,
        ) {
            *previous += posting.term_frequency * boost_per_field;
        }
    }
}
