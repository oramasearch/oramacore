use std::collections::{HashMap, HashSet};

use anyhow::Result;
use fst::{Automaton, IntoStreamer, Map, Streamer};
use tracing::warn;

use crate::{document_storage::DocumentId, nlp::stop_words::fi};

use super::{scorer::BM25Scorer, GlobalInfo};

#[derive(Default, Debug)]
pub struct CommittedStringFieldIndex {
    fst_map: Map<Vec<u8>>,
    document_lengths_per_document: HashMap<DocumentId, u32>,
    sum_of_total_field_length: usize,
    storage: HashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
    number_of_documents: usize,
}

impl CommittedStringFieldIndex {
    pub fn new(
        fst_map: Map<Vec<u8>>,
        document_lengths_per_document: HashMap<DocumentId, u32>,
        sum_of_total_field_length: usize,
        storage: HashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
        number_of_documents: usize,
    ) -> Self {
        Self {
            fst_map,
            document_lengths_per_document,
            sum_of_total_field_length,
            storage,
            number_of_documents,
        }
    }

    pub fn get_global_info(&self) -> GlobalInfo {
        GlobalInfo {
            total_document_length: self.sum_of_total_field_length,
            total_documents: self.number_of_documents,
        }
    }

    pub fn search(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        if tokens.len() == 1 {
            self.search_without_phrase_match(tokens, boost, scorer, filtered_doc_ids, global_info)
        } else {
            self.search_with_phrase_match(tokens, boost, scorer, filtered_doc_ids, global_info)
        }
    }

    pub fn search_with_phrase_match(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        let total_field_length = global_info.total_document_length as f32;
        let total_documents_with_field = global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        struct Foo {
            positions: HashSet<usize>,
            matches: Vec<(u32, usize, usize)>,
        }
        let mut foo: HashMap<DocumentId, Foo> = HashMap::new();

        for token in tokens {
            let automaton = fst::automaton::Str::new(token).starts_with();
            let mut stream = self.fst_map.search(automaton).into_stream();

            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            while let Some((_, posting_list_id)) = stream.next() {
                let postings = match self.storage.get(&posting_list_id) {
                    Some(postings) => postings,
                    None => {
                        warn!("posting list not found: skipping");
                        continue;
                    }
                };

                let total_documents_with_term_in_field = postings.len();

                for (doc_id, positions) in postings {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(doc_id) {
                            continue;
                        }
                    }

                    let v = foo.entry(*doc_id).or_insert_with(|| Foo {
                        positions: Default::default(),
                        matches: Default::default(),
                    });
                    v.positions.extend(positions);

                    let field_length =
                        *self.document_lengths_per_document.get(doc_id).unwrap_or(&1);
                    v.matches.push((
                        field_length,
                        positions.len(),
                        total_documents_with_term_in_field,
                    ));
                }
            }
        }

        for (doc_id, Foo { matches, positions }) in foo {
            let mut ordered_positions: Vec<_> = positions.iter().copied().collect();
            ordered_positions.sort_unstable(); // asc order

            let sequences_count = ordered_positions
                .windows(2)
                .filter(|window| {
                    let first = window[0];
                    let second = window[1];

                    // TODO: make this "1" configurable
                    (second - first) < 1
                })
                .count();

            let total_boost = (sequences_count as f32 * 2.0) + boost;

            for (field_length, term_occurrence_in_field, total_documents_with_term_in_field) in
                matches
            {
                scorer.add(
                    doc_id,
                    term_occurrence_in_field as u32,
                    field_length,
                    average_field_length,
                    global_info.total_documents as f32,
                    total_documents_with_term_in_field,
                    1.2,
                    0.75,
                    total_boost,
                );
            }
        }

        /*
        scorer.add(
            *doc_id,
            positions.len() as u32,
            field_length,
            average_field_length,
            global_info.total_documents as f32,
            total_documents_with_term_in_field,
            1.2,
            0.75,
            boost,
        );
        */

        Ok(())
    }

    pub fn search_without_phrase_match(
        &self,
        tokens: &[String],
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        global_info: &GlobalInfo,
    ) -> Result<()> {
        let total_field_length = global_info.total_document_length as f32;
        let total_documents_with_field = global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        for token in tokens {
            let automaton = fst::automaton::Str::new(token).starts_with();
            let mut stream = self.fst_map.search(automaton).into_stream();

            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            while let Some((_, posting_list_id)) = stream.next() {
                let postings = match self.storage.get(&posting_list_id) {
                    Some(postings) => postings,
                    None => {
                        warn!("posting list not found: skipping");
                        continue;
                    }
                };

                let total_documents_with_term_in_field = postings.len();

                for (doc_id, positions) in postings {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(doc_id) {
                            continue;
                        }
                    }

                    let field_length =
                        *self.document_lengths_per_document.get(doc_id).unwrap_or(&1);
                    let term_occurrence_in_field = positions.len() as u32;

                    scorer.add(
                        *doc_id,
                        term_occurrence_in_field,
                        field_length,
                        average_field_length,
                        global_info.total_documents as f32,
                        total_documents_with_term_in_field,
                        1.2,
                        0.75,
                        boost,
                    );
                }
            }
        }

        Ok(())
    }
}
/*
impl CommittedStringFieldIndex {
    /*
    pub async fn search(
        &self,
        tokens: &[String],
        global_info: &GlobalInfo,
        search_on: Option<&[FieldId]>,
        boost: &HashMap<FieldId, f32>,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
    ) -> Result<()> {

        let mut all_postings: Vec<(Vec<Posting>, usize)> = Vec::new();

        if let Some(fst) = self.fst_map.read().await.as_ref() {

            // We threat the current token as the postfixes
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            for token in tokens {
                let automaton = fst::automaton::Str::new(token).starts_with();
                let mut stream = fst.search(automaton).into_stream();

                while let Some(result) = stream.next() {
                    let (_, packed_value) = result;
                    let posting_list_id = PostingListId((packed_value >> 32) as u32);
                    let term_frequency = (packed_value & 0xFFFFFFFF) as usize;

                    // if let Ok(postings) = self.posting_storage.get(posting_list_id) {
                    //     all_postings.push((postings, term_frequency));
                    // }
                }
            }
        } else {
            warn!("fst map is empty: returning empty search results");
            return Ok(());
        }

        let in_filtered = move |posting: &Posting| -> bool {
            if let Some(filtered_doc_ids) = filtered_doc_ids {
                filtered_doc_ids.contains(&posting.document_id)
            } else {
                true
            }
        };

        let mut filtered_postings: HashMap<DocumentId, Vec<(Posting, usize)>> = HashMap::new();
        for (postings, term_freq) in all_postings {
            for posting in postings.into_iter().filter(in_filtered) {
                if search_on
                    .as_ref()
                    .map(|search_on| search_on.contains(&posting.field_id))
                    .unwrap_or(true)
                {
                    filtered_postings
                        .entry(posting.document_id)
                        .or_default()
                        .push((posting, term_freq));
                }
            }
        }

        for (doc_id, postings) in filtered_postings {
            let mut token_positions: Vec<Vec<usize>> = postings
                .iter()
                .map(|(posting, _)| posting.positions.clone())
                .collect();

            token_positions
                .iter_mut()
                .for_each(|positions| positions.sort_unstable());

            let is_phrase_match =  self.is_phrase_match(&token_positions);

            for (posting, term_freq) in postings {
                let boost_per_field = *boost.get(&posting.field_id).unwrap_or(&1.0);
                let boost_per_field = if is_phrase_match {
                    // If the query token positions matches the document token positions, we boost the score
                    // The `25` here is arbitrary and we should probably make it configurable
                    // TODO: Make this `25` configurable
                    boost_per_field * 25.0
                } else {
                    boost_per_field
                };
                // scorer.add_entry(global_info, &posting, term_freq as f32, boost_per_field);

                /*
                scorer.add(
                    doc_id,
                    posting.field_id,

                );
                */
            }
        }


        Ok(())
    }
    */

}
*/

pub mod merge {
    use std::{
        collections::HashMap,
        iter::Peekable,
        sync::{atomic::AtomicU64, Arc},
    };

    use anyhow::Result;
    use fst::{Map, Streamer};

    use crate::{
        document_storage::DocumentId,
        indexes::string::{
            uncommitted::{Positions, TotalDocumentsWithTermInField},
            UncommittedStringFieldIndex,
        },
    };

    use super::CommittedStringFieldIndex;

    struct MergeIterator<
        'a,
        UncommittedIterType: Iterator<
            Item = (
                Vec<u8>,
                (
                    TotalDocumentsWithTermInField,
                    HashMap<DocumentId, Positions>,
                ),
            ),
        >,
        CommittedIterType: Iterator<Item = (Vec<u8>, u64)>,
    > {
        posting_id_generator: Arc<AtomicU64>,
        committed_storage: &'a mut HashMap<u64, Vec<(DocumentId, Vec<usize>)>>,
        committed_iter: Peekable<CommittedIterType>,
        uncommitted_iter: Peekable<UncommittedIterType>,
    }
    impl<
            'a,
            UncommittedIterType: Iterator<
                Item = (
                    Vec<u8>,
                    (
                        TotalDocumentsWithTermInField,
                        HashMap<DocumentId, Positions>,
                    ),
                ),
            >,
            CommittedIterType: Iterator<Item = (Vec<u8>, u64)>,
        > Iterator for MergeIterator<'a, UncommittedIterType, CommittedIterType>
    {
        type Item = (Vec<u8>, u64);

        fn next(&mut self) -> Option<Self::Item> {
            let uncommitted = self.uncommitted_iter.peek();
            let committed = self.committed_iter.peek();

            match (uncommitted, committed) {
                (Some((uncommitted_key, _)), Some((committed_key, _))) => {
                    let cmp = uncommitted_key.cmp(committed_key);
                    match cmp {
                        std::cmp::Ordering::Less => {
                            let (uncommitted_key, (_, positions_per_document_id)) =
                                self.uncommitted_iter.next().unwrap();

                            let new_posting_list_id = self
                                .posting_id_generator
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                            self.committed_storage.insert(
                                new_posting_list_id,
                                positions_per_document_id
                                    .into_iter()
                                    .map(|(doc_id, positions)| (doc_id, positions.0))
                                    .collect(),
                            );

                            Some((uncommitted_key, new_posting_list_id))
                        }
                        std::cmp::Ordering::Equal => {
                            // Merge
                            let (_, (_, positions_per_document_id)) =
                                self.uncommitted_iter.next().unwrap();
                            let (key, committed_posting_id) = self.committed_iter.next().unwrap();

                            let committed_positions_per_doc = self
                                .committed_storage
                                .entry(committed_posting_id)
                                .or_default();

                            committed_positions_per_doc.extend(
                                positions_per_document_id
                                    .into_iter()
                                    .map(|(doc_id, positions)| (doc_id, positions.0)),
                            );

                            Some((key, committed_posting_id))
                        }
                        std::cmp::Ordering::Greater => self.committed_iter.next(),
                    }
                }
                (Some(_), None) => {
                    let (uncommitted_key, (_, positions_per_document_id)) =
                        self.uncommitted_iter.next().unwrap();

                    let a = self
                        .posting_id_generator
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    self.committed_storage.insert(
                        a,
                        positions_per_document_id
                            .into_iter()
                            .map(|(doc_id, positions)| (doc_id, positions.0))
                            .collect(),
                    );

                    Some((uncommitted_key, a))
                }
                (None, Some(_)) => self.committed_iter.next(),
                // Both are `None`: we are done
                (None, None) => None,
            }
        }
    }

    struct FTSIter<'stream> {
        stream: fst::map::Stream<'stream>,
    }
    impl<'stream> Iterator for FTSIter<'stream> {
        // The Item allocate memory, but we could avoid it by using a reference
        // TODO: resolve lifetime issue with reference here
        type Item = (Vec<u8>, u64);

        fn next(&mut self) -> Option<Self::Item> {
            self.stream.next().map(|(key, value)| (key.to_vec(), value))
        }
    }

    pub fn merge(
        posting_id_generator: Arc<AtomicU64>,
        uncommitted: UncommittedStringFieldIndex,
        committed: CommittedStringFieldIndex,
    ) -> Result<CommittedStringFieldIndex> {
        let UncommittedStringFieldIndex {
            inner,
            field_length_per_doc: document_length_per_doc,
            total_field_length,
            ..
        } = uncommitted;

        let inner = inner.read().unwrap();
        let iter = inner.iter();
        let mut uncommitted_data: Vec<_> = iter.collect();
        uncommitted_data.sort_by(|a, b| a.0.cmp(&b.0));

        let uncommitted_iter = uncommitted_data.into_iter().peekable();

        let mut committed_document_lengths_per_document = committed.document_lengths_per_document;
        let mut committed_sum_of_total_field_length = committed.sum_of_total_field_length as u64;
        let mut committed_storage = committed.storage;
        let mut committed_document_count = committed.number_of_documents;

        let merge_iterator = MergeIterator {
            posting_id_generator,

            committed_storage: &mut committed_storage,
            committed_iter: FTSIter {
                stream: committed.fst_map.stream(),
            }
            .peekable(),

            uncommitted_iter,
        };

        let fst_map: Map<Vec<u8>> = Map::from_iter(merge_iterator)?;

        committed_document_count += document_length_per_doc.len();
        committed_document_lengths_per_document.extend(document_length_per_doc);
        committed_sum_of_total_field_length +=
            total_field_length.load(std::sync::atomic::Ordering::Relaxed);

        let new_committed = CommittedStringFieldIndex::new(
            fst_map,
            committed_document_lengths_per_document,
            committed_sum_of_total_field_length as usize,
            committed_storage,
            committed_document_count,
        );

        Ok(new_committed)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::indexes::string::scorer::BM25Scorer;
        use crate::test_utils::{
            create_committed_string_field_index, create_uncommitted_string_field_index,
            create_uncommitted_string_field_index_from,
        };
        use serde_json::json;

        #[test]
        fn test_indexes_string_committed_from_uncommitted() -> Result<()> {
            let index = create_uncommitted_string_field_index(vec![
                json!({
                    "field": "hello hello world",
                })
                .try_into()?,
                json!({
                    "field": "hello tom",
                })
                .try_into()?,
            ])?;

            let mut scorer = BM25Scorer::new();
            index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )?;
            let before_output = scorer.get_scores();

            let new_committed = merge(
                Arc::new(AtomicU64::new(0)),
                index,
                CommittedStringFieldIndex {
                    fst_map: Map::default(),
                    document_lengths_per_document: HashMap::new(),
                    sum_of_total_field_length: 0,
                    storage: HashMap::new(),
                    number_of_documents: 0,
                },
            )?;

            let mut scorer = BM25Scorer::new();
            new_committed.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &new_committed.get_global_info(),
            )?;
            let after_output = scorer.get_scores();

            assert_eq!(before_output, after_output);

            Ok(())
        }

        #[test]
        fn test_indexes_string_merge_equal() -> Result<()> {
            let _ = tracing_subscriber::fmt::try_init();

            let (committed_index, posting_id_generator) =
                create_committed_string_field_index(vec![
                    json!({
                        "field": "hello hello world",
                    })
                    .try_into()?,
                    json!({
                        "field": "hello tom",
                    })
                    .try_into()?,
                ])?;
            let uncommitted_index = create_uncommitted_string_field_index_from(
                vec![
                    json!({
                        "field": "hello hello world",
                    })
                    .try_into()?,
                    json!({
                        "field": "hello tom",
                    })
                    .try_into()?,
                ],
                2,
            )?;

            let new_committed_index =
                merge(posting_id_generator, uncommitted_index, committed_index)?;

            let mut scorer = BM25Scorer::new();
            new_committed_index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &new_committed_index.get_global_info(),
            )?;
            let new_committed_output = scorer.get_scores();

            assert_eq!(new_committed_output.len(), 4);

            Ok(())
        }

        #[test]
        fn test_indexes_string_merge_no_uncommitted() -> Result<()> {
            let _ = tracing_subscriber::fmt::try_init();

            let (committed_index, posting_id_generator) =
                create_committed_string_field_index(vec![
                    json!({
                        "field": "hello hello world",
                    })
                    .try_into()?,
                    json!({
                        "field": "hello tom",
                    })
                    .try_into()?,
                ])?;
            let uncommitted_index = create_uncommitted_string_field_index_from(vec![], 2)?;

            let new_committed_index =
                merge(posting_id_generator, uncommitted_index, committed_index)?;

            let mut scorer = BM25Scorer::new();
            new_committed_index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &new_committed_index.get_global_info(),
            )?;
            let new_committed_output = scorer.get_scores();

            assert_eq!(new_committed_output.len(), 2);

            Ok(())
        }

        #[test]
        fn test_indexes_string_merge_both_empty() -> Result<()> {
            let _ = tracing_subscriber::fmt::try_init();

            let (committed_index, posting_id_generator) =
                create_committed_string_field_index(vec![])?;
            let uncommitted_index = create_uncommitted_string_field_index_from(vec![], 0)?;

            let new_committed_index =
                merge(posting_id_generator, uncommitted_index, committed_index)?;

            let mut scorer = BM25Scorer::new();
            new_committed_index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &new_committed_index.get_global_info(),
            )?;
            let new_committed_output = scorer.get_scores();

            assert_eq!(new_committed_output.len(), 0);

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::indexes::string::scorer::BM25Scorer;
    use crate::test_utils::create_committed_string_field_index;

    use super::*;

    #[test]
    fn test_indexes_string_committed() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let (index, _) = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        // Exact match
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let exact_match_output = scorer.get_scores();
        assert_eq!(
            exact_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );
        assert!(exact_match_output[&DocumentId(0)] > exact_match_output[&DocumentId(1)]);

        // Prefix match
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hel".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let prefix_match_output = scorer.get_scores();
        assert_eq!(
            prefix_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );

        Ok(())
    }

    #[test]
    fn test_indexes_string_committed_boost() -> Result<()> {
        let (index, _) = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        // 1.0
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let base_output = scorer.get_scores();

        // 0.5
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            0.5,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let half_boost_output = scorer.get_scores();

        // 2.0
        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            2.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let twice_boost_output = scorer.get_scores();

        assert!(base_output[&DocumentId(0)] > half_boost_output[&DocumentId(0)]);
        assert!(base_output[&DocumentId(0)] < twice_boost_output[&DocumentId(0)]);

        assert!(base_output[&DocumentId(1)] > half_boost_output[&DocumentId(1)]);
        assert!(base_output[&DocumentId(1)] < twice_boost_output[&DocumentId(1)]);

        Ok(())
    }

    #[test]
    fn test_indexes_string_committed_nonexistent_term() -> Result<()> {
        let (index, _) = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        let mut scorer = BM25Scorer::new();
        index.search(
            &["nonexistent".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let output = scorer.get_scores();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );

        Ok(())
    }

    #[test]
    fn test_indexes_string_committed_field_filter() -> Result<()> {
        let (index, _) = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])?;

        // Exclude a doc
        {
            let mut scorer = BM25Scorer::new();
            index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                Some(&HashSet::from_iter([DocumentId(0)])),
                &index.get_global_info(),
            )?;
            let output = scorer.get_scores();
            assert!(output.contains_key(&DocumentId(0)),);
            assert!(!output.contains_key(&DocumentId(1)),);
        }

        // Exclude all docs
        {
            let mut scorer = BM25Scorer::new();
            index.search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                Some(&HashSet::new()),
                &index.get_global_info(),
            )?;
            let output = scorer.get_scores();
            assert!(output.is_empty(),);
        }

        Ok(())
    }

    #[test]
    fn test_indexes_string_committed_on_empty_index() -> Result<()> {
        let (index, _) = create_committed_string_field_index(vec![])?;

        let mut scorer = BM25Scorer::new();
        index.search(
            &["hello".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let output = scorer.get_scores();
        assert!(output.is_empty(),);

        Ok(())
    }

    #[test]
    fn test_indexes_string_committed_large_text() -> Result<()> {
        let (index, _) = create_committed_string_field_index(vec![json!({
            "field": "word ".repeat(10000),
        })
        .try_into()?])?;

        let mut scorer = BM25Scorer::new();
        index.search(
            &["word".to_string()],
            1.0,
            &mut scorer,
            None,
            &index.get_global_info(),
        )?;
        let output = scorer.get_scores();
        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );

        Ok(())
    }
}
