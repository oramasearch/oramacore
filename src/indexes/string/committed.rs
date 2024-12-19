
use std::collections::{HashMap, HashSet};

use anyhow::Result;
use fst::{Automaton, IntoStreamer, Map, Streamer};
use tokio::sync::RwLock;
use tracing::warn;

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

use super::{posting_storage::{PostingListId, PostingStorage}, scorer::BM25Scorer, GlobalInfo, Posting};

#[derive(Debug)]
pub struct CommittedStringIndex {
    fst_map: RwLock<Option<Map<Vec<u8>>>>,
    posting_storage: PostingStorage,
}

impl CommittedStringIndex {
    pub fn new(posting_storage: PostingStorage) -> Self {
        Self {
            fst_map: RwLock::new(None),
            posting_storage,
        }
    }

    pub async fn search(&self,
        tokens: &[String],
        global_info: &GlobalInfo,
        search_on: Option<&[FieldId]>,
        boost: &HashMap<FieldId, f32>,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>
    ) -> Result<()> {
        let mut all_postings = Vec::new();

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

                    if let Ok(postings) = self.posting_storage.get(posting_list_id) {
                        all_postings.push((postings, term_frequency));
                    }
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

                scorer.add(
                    doc_id,
                    posting.field_id,

                );
            }
        }

        Ok(())

    }


    fn is_phrase_match(&self, token_positions: &[Vec<usize>]) -> bool {
        if token_positions.is_empty() {
            return false;
        }

        let position_sets: Vec<std::collections::HashSet<usize>> = token_positions
            .iter()
            .skip(1)
            .map(|positions| positions.iter().copied().collect())
            .collect();

        for &start_pos in &token_positions[0] {
            let mut current_pos = start_pos;
            let mut matched = true;

            for positions in &position_sets {
                let next_pos = current_pos + 1;
                if positions.contains(&next_pos) {
                    current_pos = next_pos;
                } else {
                    matched = false;
                    break;
                }
            }

            if matched {
                return true;
            }
        }

        false
    }
}
