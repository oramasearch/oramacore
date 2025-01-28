use std::{collections::{HashMap, HashSet}, hash::Hash, path::PathBuf};

use anyhow::{Context, Result};
use fst::MapBuilder;
use tracing::info;

use crate::{
    collection_manager::sides::read::collection::uncommitted::{Positions, TotalDocumentsWithTermInField}, file_utils::{create_if_not_exists, BufferedFile}, indexes::{
        fst::FSTIndex,
        map::Map,
        string::{document_lengths, BM25Scorer, GlobalInfo},
    }, types::DocumentId
};

#[derive(Debug)]
pub struct StringField {
    index: FSTIndex,

    posting_storage: PostingIdStorage,
    document_lengths_per_document: DocumentLengthsPerDocument,
}

impl StringField {
    pub fn from_iter(
        uncommitted_iter: impl Iterator<Item = (Vec<u8>, (TotalDocumentsWithTermInField, HashMap<DocumentId, Positions>))>,
        length_per_documents: HashMap<DocumentId, u32>,
        data_dir: PathBuf,
    ) -> Result<Self> {
        let mut posting_id_generator = 0;

        create_if_not_exists(&data_dir)?;

        let mut delta_committed_storage: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> =
            Default::default();
        let iter = uncommitted_iter
            .map(|(key, value)| {
                let new_posting_list_id = posting_id_generator;
                posting_id_generator += 1;

                delta_committed_storage.insert(
                    new_posting_list_id,
                    value
                        .1
                        .into_iter()
                        .map(|(doc_id, positions)| (doc_id, positions.0))
                        .collect(),
                );

                (key, new_posting_list_id)
            });

        let index = FSTIndex::from_iter(iter, data_dir, |k, v| {

        })?;

        Ok(Self {
            index,
            posting_storage: PostingIdStorage {
                inner: Map::from_hash_map(delta_committed_storage),
            },
            document_lengths_per_document: DocumentLengthsPerDocument {
                inner: Map::from_hash_map(length_per_documents),
            },
        })
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
            let matches = self
                .index
                .search(token)
                .flat_map(|posting_list_id| self.posting_storage.get_posting(&posting_list_id))
                .flat_map(|postings| {
                    let total_documents_with_term_in_field = postings.len();

                    postings
                        .iter()
                        .filter(|(doc_id, _)| {
                            filtered_doc_ids
                                .map_or(true, |filtered_doc_ids| filtered_doc_ids.contains(doc_id))
                        })
                        .map(move |(doc_id, positions)| {
                            let field_length =
                                self.document_lengths_per_document.get_length(doc_id);
                            let term_occurrence_in_field = positions.len() as u32;
                            (
                                doc_id,
                                term_occurrence_in_field,
                                field_length,
                                total_documents_with_term_in_field,
                            )
                        })
                });

            for (
                doc_id,
                term_occurrence_in_field,
                field_length,
                total_documents_with_term_in_field,
            ) in matches
            {
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

        Ok(())
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

        struct PhraseMatchStorage {
            positions: HashSet<usize>,
            matches: Vec<(u32, usize, usize)>,
        }
        let mut storage: HashMap<DocumentId, PhraseMatchStorage> = HashMap::new();

        for token in tokens {
            let iter = self
                .index
                .search(token)
                .filter_map(|posting_id| self.posting_storage.get_posting(&posting_id))
                .flat_map(|postings| {
                    let total_documents_with_term_in_field = postings.len();

                    postings
                        .iter()
                        .filter(|(doc_id, _)| {
                            filtered_doc_ids
                                .map_or(true, |filtered_doc_ids| filtered_doc_ids.contains(doc_id))
                        })
                        .map(move |(doc_id, positions)| {
                            let field_lenght =
                                self.document_lengths_per_document.get_length(&doc_id);
                            (
                                doc_id,
                                field_lenght,
                                positions,
                                total_documents_with_term_in_field,
                            )
                        })
                });

            for (doc_id, field_length, positions, total_documents_with_term_in_field) in iter {
                let v = storage
                    .entry(*doc_id)
                    .or_insert_with(|| PhraseMatchStorage {
                        positions: Default::default(),
                        matches: Default::default(),
                    });
                v.positions.extend(positions);
                v.matches.push((
                    field_length,
                    positions.len(),
                    total_documents_with_term_in_field,
                ));
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
}

#[derive(Debug)]
struct PostingIdStorage {
    inner: Map<u64, Vec<(DocumentId, Vec<usize>)>>,
}
impl PostingIdStorage {
    fn get_posting(&self, posting_id: &u64) -> Option<&Vec<(DocumentId, Vec<usize>)>> {
        self.inner.get(posting_id)
    }
}

#[derive(Debug)]
struct DocumentLengthsPerDocument {
    inner: Map<DocumentId, u32>,
}
impl DocumentLengthsPerDocument {
    fn get_length(&self, doc_id: &DocumentId) -> u32 {
        self.inner.get(doc_id).copied().unwrap_or(1)
    }
}
