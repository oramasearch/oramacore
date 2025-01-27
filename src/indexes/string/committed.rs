use std::{
    collections::{HashMap, HashSet},
    fs::File,
    sync::{atomic::AtomicU64, Arc},
};

use anyhow::{Context, Result};
use fst::{Automaton, IntoStreamer, Map, Streamer};
use memmap::Mmap;
use tracing::{info, warn};

use crate::{file_utils::BufferedFile, types::DocumentId};

use super::{
    document_lengths::DocumentLengthsPerDocument, posting_storage::PostingIdStorage,
    scorer::BM25Scorer, GlobalInfo, StringIndexFieldInfo,
};

#[derive(Debug)]
pub struct CommittedStringFieldIndex {
    fst_map: Map<memmap::Mmap>,
    pub document_lengths_per_document: DocumentLengthsPerDocument,

    pub storage: PostingIdStorage,
    pub posting_id_generator: Arc<AtomicU64>,

    global_info: GlobalInfo,

    info: StringIndexFieldInfo,
}

impl CommittedStringFieldIndex {
    pub fn try_new(string_index_field_info: StringIndexFieldInfo) -> Result<Self> {
        // Reload field
        let global_info: GlobalInfo =
            BufferedFile::open(string_index_field_info.global_info_path.clone())
                .context("Cannot open global info file")?
                .read_json_data()
                .context("Cannot deserialize global info")?;

        let posting_id = std::fs::read(string_index_field_info.posting_id_path.clone()).unwrap();
        let posting_id: u64 = serde_json::from_slice(&posting_id).unwrap();

        let file = File::open(string_index_field_info.fst_path.clone())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let fst_map = Map::new(mmap)?;

        let document_lengths_per_document = DocumentLengthsPerDocument::try_new(
            string_index_field_info.document_length_path.clone(),
        )?;
        let storage = PostingIdStorage::try_new(string_index_field_info.posting_path.clone())?;

        Ok(CommittedStringFieldIndex {
            fst_map,
            document_lengths_per_document,
            storage,
            posting_id_generator: Arc::new(AtomicU64::new(posting_id)),
            global_info,
            info: string_index_field_info,
        })
    }

    pub fn get_info(&self) -> StringIndexFieldInfo {
        self.info.clone()
    }

    pub fn get_global_info(&self) -> GlobalInfo {
        self.global_info.clone()
    }

    pub async fn search(
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
                .await
        } else {
            self.search_with_phrase_match(tokens, boost, scorer, filtered_doc_ids, global_info)
                .await
        }
    }

    pub async fn search_with_phrase_match(
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

        struct PhraseMatchStorage {
            positions: HashSet<usize>,
            matches: Vec<(u32, usize, usize)>,
        }
        let mut storage: HashMap<DocumentId, PhraseMatchStorage> = HashMap::new();

        let fst_map = &self.fst_map;

        for token in tokens {
            let automaton = fst::automaton::Str::new(token).starts_with();
            let mut stream = fst_map.search(automaton).into_stream();

            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            while let Some((_, posting_list_id)) = stream.next() {
                let postings = match self.storage.get_posting(&posting_list_id).await? {
                    Some(postings) => postings,
                    None => {
                        warn!("posting list not found: skipping");
                        continue;
                    }
                };

                let total_documents_with_term_in_field = postings.len();

                for (doc_id, positions) in postings {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(&doc_id) {
                            continue;
                        }
                    }

                    let v = storage.entry(doc_id).or_insert_with(|| PhraseMatchStorage {
                        positions: Default::default(),
                        matches: Default::default(),
                    });
                    let position_len = positions.len();
                    v.positions.extend(positions);

                    let field_length = self
                        .document_lengths_per_document
                        .get_length(&doc_id)
                        .await
                        .context("Failed to get document length")?;
                    v.matches.push((
                        field_length,
                        position_len,
                        total_documents_with_term_in_field,
                    ));
                }
            }
        }

        let mut total_matches = 0_usize;
        for (doc_id, PhraseMatchStorage { matches, positions }) in storage {
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

            // We have different kind of boosting:
            // 1. Boost for the exact match: not implemented
            // 2. Boost for the phrase match when the terms appear in sequence (without holes): implemented
            // 3. Boost for the phrase match when the terms appear in sequence (with holes): not implemented
            // 4. Boost for the phrase match when the terms appear in any order: implemented
            // 5. Boost defined by the user: implemented
            // We should allow the user to configure which boost to use and how much it impacts the score.
            // TODO: think about this
            let boost_any_order = positions.len() as f32;
            let boost_sequence = sequences_count as f32 * 2.0;
            let total_boost = boost_any_order + boost_sequence + boost;

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

                total_matches += 1;
            }
        }

        info!(total_matches = total_matches, "Committed total matches");

        Ok(())
    }

    pub async fn search_without_phrase_match(
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

        let fst_map = &self.fst_map;

        for token in tokens {
            let automaton = fst::automaton::Str::new(token).starts_with();
            let mut stream = fst_map.search(automaton).into_stream();

            // We don't "boost" the exact match at all.
            // Should we boost if the match is "perfect"?
            // TODO: think about this

            while let Some((_, posting_list_id)) = stream.next() {
                let postings = match self.storage.get_posting(&posting_list_id).await? {
                    Some(postings) => postings,
                    None => {
                        warn!("posting list not found: skipping");
                        continue;
                    }
                };

                let total_documents_with_term_in_field = postings.len();

                for (doc_id, positions) in postings {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(&doc_id) {
                            continue;
                        }
                    }

                    let field_length = self
                        .document_lengths_per_document
                        .get_length(&doc_id)
                        .await
                        .context("Failed to get document length")?;
                    let term_occurrence_in_field = positions.len() as u32;

                    scorer.add(
                        doc_id,
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        indexes::string::scorer::BM25Scorer, test_utils::create_committed_string_field_index,
    };

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_string_committed() -> Result<()> {
        let _ = tracing_subscriber::fmt::try_init();

        let index = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?
        .unwrap();

        // Exact match
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let exact_match_output = scorer.get_scores();
        assert_eq!(
            exact_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );
        assert!(exact_match_output[&DocumentId(0)] > exact_match_output[&DocumentId(1)]);

        // Prefix match
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["hel".to_string()],
                1.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let prefix_match_output = scorer.get_scores();
        assert_eq!(
            prefix_match_output.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from_iter([DocumentId(0), DocumentId(1)])
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_string_committed_boost() -> Result<()> {
        let index = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?
        .unwrap();

        // 1.0
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["hello".to_string()],
                1.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let base_output = scorer.get_scores();

        // 0.5
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["hello".to_string()],
                0.5,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let half_boost_output = scorer.get_scores();

        // 2.0
        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["hello".to_string()],
                2.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let twice_boost_output = scorer.get_scores();

        assert!(base_output[&DocumentId(0)] > half_boost_output[&DocumentId(0)]);
        assert!(base_output[&DocumentId(0)] < twice_boost_output[&DocumentId(0)]);

        assert!(base_output[&DocumentId(1)] > half_boost_output[&DocumentId(1)]);
        assert!(base_output[&DocumentId(1)] < twice_boost_output[&DocumentId(1)]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_string_committed_nonexistent_term() -> Result<()> {
        let index = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?
        .unwrap();

        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["nonexistent".to_string()],
                1.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let output = scorer.get_scores();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_string_committed_field_filter() -> Result<()> {
        let index = create_committed_string_field_index(vec![
            json!({
                "field": "hello hello world",
            })
            .try_into()?,
            json!({
                "field": "hello tom",
            })
            .try_into()?,
        ])
        .await?
        .unwrap();

        // Exclude a doc
        {
            let mut scorer = BM25Scorer::new();
            index
                .search(
                    &["hello".to_string()],
                    1.0,
                    &mut scorer,
                    Some(&HashSet::from_iter([DocumentId(0)])),
                    &index.get_global_info(),
                )
                .await?;
            let output = scorer.get_scores();
            assert!(output.contains_key(&DocumentId(0)),);
            assert!(!output.contains_key(&DocumentId(1)),);
        }

        // Exclude all docs
        {
            let mut scorer = BM25Scorer::new();
            index
                .search(
                    &["hello".to_string()],
                    1.0,
                    &mut scorer,
                    Some(&HashSet::new()),
                    &index.get_global_info(),
                )
                .await?;
            let output = scorer.get_scores();
            assert!(output.is_empty(),);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_indexes_string_committed_large_text() -> Result<()> {
        let index = create_committed_string_field_index(vec![json!({
            "field": "word ".repeat(10000),
        })
        .try_into()?])
        .await?
        .unwrap();

        let mut scorer = BM25Scorer::new();
        index
            .search(
                &["word".to_string()],
                1.0,
                &mut scorer,
                None,
                &index.get_global_info(),
            )
            .await?;
        let output = scorer.get_scores();
        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );

        Ok(())
    }
}
