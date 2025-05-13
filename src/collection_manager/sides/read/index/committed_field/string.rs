use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context, Result};
use filters::FilterResult;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    collection_manager::{
        bm25::BM25Scorer,
        global_info::GlobalInfo,
        sides::read::index::uncommitted_field::{Positions, TotalDocumentsWithTermInField},
    },
    file_utils::create_if_not_exists,
    indexes::{fst::FSTIndex, map::Map},
    merger::MergedIterator,
    types::DocumentId,
};

#[derive(Debug)]
pub struct CommittedStringField {
    field_path: Box<[String]>,
    index: FSTIndex,

    posting_storage: PostingIdStorage,
    document_lengths_per_document: DocumentLengthsPerDocument,

    data_dir: PathBuf,
}

impl CommittedStringField {
    pub fn from_iter(
        field_path: Box<[String]>,
        uncommitted_iter: impl Iterator<
            Item = (
                Vec<u8>,
                (
                    TotalDocumentsWithTermInField,
                    HashMap<DocumentId, (Positions, Positions)>,
                ),
            ),
        >,
        mut length_per_documents: HashMap<DocumentId, u32>,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
    ) -> Result<Self> {
        let mut posting_id_generator = 0;

        create_if_not_exists(&data_dir)?;

        let mut delta_committed_storage: HashMap<u64, Vec<(DocumentId, (Vec<usize>, Vec<usize>))>> =
            Default::default();
        let iter = uncommitted_iter.map(|(key, value)| {
            let new_posting_list_id = posting_id_generator;
            posting_id_generator += 1;

            delta_committed_storage.insert(
                new_posting_list_id,
                value
                    .1
                    .into_iter()
                    .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id))
                    .map(|(doc_id, positions)| (doc_id, (positions.0 .0, positions.1 .0)))
                    .collect(),
            );

            (key, new_posting_list_id)
        });

        uncommitted_document_deletions.iter().for_each(|doc_id| {
            length_per_documents.remove(doc_id);
        });

        let posting_id_storage_file_path = data_dir.join("posting_id_storage.map");
        let length_per_documents_file_path = data_dir.join("length_per_documents.map");
        let fst_file_path = data_dir.join("fst.map");
        let index = FSTIndex::from_iter(iter, fst_file_path).context("Cannot commit fst")?;
        let document_lengths_per_document = DocumentLengthsPerDocument::from_map(
            Map::from_hash_map(length_per_documents, length_per_documents_file_path)
                .context("Cannot commit document lengths per document storage")?,
        );
        let posting_storage = PostingIdStorage::from_map(
            Map::from_hash_map(delta_committed_storage, posting_id_storage_file_path)
                .context("Cannot commit posting id storage")?,
        );
        Ok(Self {
            field_path,
            index,
            posting_storage,
            document_lengths_per_document,
            data_dir,
        })
    }

    pub fn from_iter_and_committed(
        field_path: Box<[String]>,
        uncommitted_iter: impl Iterator<
            Item = (
                Vec<u8>,
                (
                    TotalDocumentsWithTermInField,
                    HashMap<DocumentId, (Positions, Positions)>,
                ),
            ),
        >,
        committed: &Self,
        length_per_documents: HashMap<DocumentId, u32>,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
    ) -> Result<Self> {
        create_if_not_exists(&data_dir)
            .context("Cannot create data directory for committed string field")?;

        let new_posting_storage_file = data_dir.join("posting_id_storage.map");
        let old_posting_storage_file = committed.posting_storage.get_backed_file();

        if old_posting_storage_file != new_posting_storage_file {
            std::fs::copy(&old_posting_storage_file, &new_posting_storage_file).with_context(
                || {
                    format!(
                        "Cannot copy posting storage file: old {:?} -> new {:?}",
                        old_posting_storage_file, new_posting_storage_file
                    )
                },
            )?;
        }
        let posting_storage = PostingIdStorage::load(new_posting_storage_file)
            .context("Cannot load posting storage")?;
        let mut posting_id_generator = posting_storage.get_max_posting_id() + 1;
        let posting_storage = RefCell::new(posting_storage);

        let committed_iter = committed.index.iter();
        let merged_iterator = MergedIterator::new(
            uncommitted_iter,
            committed_iter,
            |_, (_, positions_per_document_id)| {
                let new_posting_list_id = posting_id_generator;
                posting_id_generator += 1;

                let mut lock = posting_storage.borrow_mut();
                lock.insert(
                    new_posting_list_id,
                    positions_per_document_id
                        .into_iter()
                        .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id))
                        .map(|(doc_id, positions)| (doc_id, (positions.0 .0, positions.1 .0)))
                        .collect(),
                );

                new_posting_list_id
            },
            |_, (_, positions_per_document_id), committed_posting_id| {
                let mut lock = posting_storage.borrow_mut();
                lock.merge(
                    committed_posting_id,
                    positions_per_document_id
                        .into_iter()
                        .map(|(doc_id, positions)| (doc_id, (positions.0 .0, positions.1 .0))),
                    uncommitted_document_deletions,
                );

                committed_posting_id
            },
        );
        let fst_file_path = data_dir.join("fst.map");
        let index =
            FSTIndex::from_iter(merged_iterator, fst_file_path).context("Cannot commit fst")?;

        let new_document_lengths_per_document_file = data_dir.join("length_per_documents.map");
        let old_document_lengths_per_document_file =
            committed.document_lengths_per_document.get_backed_file();
        if old_document_lengths_per_document_file != new_document_lengths_per_document_file {
            std::fs::copy(
                old_document_lengths_per_document_file,
                &new_document_lengths_per_document_file,
            )
            .context("Cannot copy posting storage file")?;
        }
        let mut document_lengths_per_document =
            DocumentLengthsPerDocument::load(new_document_lengths_per_document_file)
                .context("Cannot load document lengths per document")?;
        for (k, v) in length_per_documents {
            document_lengths_per_document.insert(k, v);
        }

        let mut posting_storage = posting_storage.into_inner();

        posting_storage.remove_doc_ids(uncommitted_document_deletions);
        document_lengths_per_document.remove_doc_ids(uncommitted_document_deletions);

        posting_storage.commit()?;
        document_lengths_per_document.commit()?;

        Ok(Self {
            field_path,
            index,
            posting_storage,
            document_lengths_per_document,
            data_dir,
        })
    }

    pub fn try_load(info: StringFieldInfo) -> Result<Self> {
        let index = FSTIndex::load(info.data_dir.join("fst.map"))?;
        let posting_storage = PostingIdStorage::load(info.data_dir.join("posting_id_storage.map"))?;
        let document_lengths_per_document =
            DocumentLengthsPerDocument::load(info.data_dir.join("length_per_documents.map"))?;

        Ok(Self {
            field_path: info.field_path,
            index,
            posting_storage,
            document_lengths_per_document,
            data_dir: info.data_dir,
        })
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn get_field_info(&self) -> StringFieldInfo {
        StringFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn global_info(&self) -> GlobalInfo {
        self.document_lengths_per_document.global_info.clone()
    }

    pub fn search(
        &self,
        tokens: &[String],
        exact: bool,
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        global_info: &GlobalInfo,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }

        if tokens.len() == 1 {
            self.search_without_phrase_match(
                tokens,
                exact,
                boost,
                scorer,
                filtered_doc_ids,
                global_info,
                uncommitted_deleted_documents,
            )
        } else {
            self.search_with_phrase_match(
                tokens,
                exact,
                boost,
                scorer,
                filtered_doc_ids,
                global_info,
                uncommitted_deleted_documents,
            )
        }
    }

    fn search_without_phrase_match(
        &self,
        tokens: &[String],
        exact: bool,
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        global_info: &GlobalInfo,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        let total_field_length = global_info.total_document_length as f32;
        let total_documents_with_field = global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        for token in tokens {
            let iter: Box<dyn Iterator<Item = u64>> = if exact {
                if let Some(posting_list_id) = self.index.search_exact(token) {
                    Box::new(vec![posting_list_id].into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                Box::new(self.index.search(token))
            };

            let matches = iter
                .flat_map(|posting_list_id| self.posting_storage.get_posting(&posting_list_id))
                .flat_map(|postings| {
                    let total_documents_with_term_in_field = postings.len();

                    postings
                        .iter()
                        .filter(|(doc_id, _)| {
                            filtered_doc_ids
                                .is_none_or(|filtered_doc_ids| filtered_doc_ids.contains(doc_id))
                        })
                        .filter(|(doc_id, _)| !uncommitted_deleted_documents.contains(doc_id))
                        .filter_map(move |(doc_id, positions)| {
                            let field_length =
                                self.document_lengths_per_document.get_length(doc_id);
                            let term_occurrence_in_field = if exact {
                                positions.0.len() as u32
                            } else {
                                (positions.0.len() + positions.1.len()) as u32
                            };

                            if term_occurrence_in_field == 0 {
                                return None;
                            }

                            Some((
                                doc_id,
                                term_occurrence_in_field,
                                field_length,
                                total_documents_with_term_in_field,
                            ))
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
                    0,
                );
            }
        }

        Ok(())
    }

    fn search_with_phrase_match(
        &self,
        tokens: &[String],
        exact: bool,
        boost: f32,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        global_info: &GlobalInfo,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        let total_field_length = global_info.total_document_length as f32;
        let total_documents_with_field = global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        struct PhraseMatchStorage {
            positions: HashSet<usize>,
            matches: Vec<(u32, usize, usize)>,
            token_indexes: u32,
        }
        let mut storage: HashMap<DocumentId, PhraseMatchStorage> = HashMap::new();

        for (token_index, token) in tokens.iter().enumerate() {
            let iter: Box<dyn Iterator<Item = u64>> = if exact {
                if let Some(posting_list_id) = self.index.search_exact(token) {
                    Box::new(vec![posting_list_id].into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                Box::new(self.index.search(token))
            };

            let iter = iter
                .filter_map(|posting_id| self.posting_storage.get_posting(&posting_id))
                .flat_map(|postings| {
                    let total_documents_with_term_in_field = postings.len();

                    postings
                        .iter()
                        .filter(|(doc_id, _)| {
                            filtered_doc_ids
                                .is_none_or(|filtered_doc_ids| filtered_doc_ids.contains(doc_id))
                        })
                        .filter(|(doc_id, _)| !uncommitted_deleted_documents.contains(doc_id))
                        .filter_map(move |(doc_id, positions)| {
                            let field_lenght =
                                self.document_lengths_per_document.get_length(doc_id);

                            let positions: Vec<usize> = if exact {
                                positions.0.to_vec()
                            } else {
                                positions
                                    .0
                                    .iter()
                                    .copied()
                                    .chain(positions.1.iter().copied())
                                    .collect()
                            };

                            if positions.is_empty() {
                                return None;
                            }

                            Some((
                                doc_id,
                                field_lenght,
                                positions,
                                total_documents_with_term_in_field,
                            ))
                        })
                });

            for (doc_id, field_length, positions, total_documents_with_term_in_field) in iter {
                let v = storage
                    .entry(*doc_id)
                    .or_insert_with(|| PhraseMatchStorage {
                        positions: Default::default(),
                        matches: Default::default(),
                        token_indexes: 0,
                    });
                v.matches.push((
                    field_length,
                    positions.len(),
                    total_documents_with_term_in_field,
                ));
                v.positions.extend(positions);
                v.token_indexes |= 1 << token_index;
            }
        }

        let mut total_matches = 0_usize;
        for (
            doc_id,
            PhraseMatchStorage {
                matches,
                positions,
                token_indexes,
            },
        ) in storage
        {
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
                    token_indexes,
                );

                total_matches += 1;
            }
        }

        info!(total_matches = total_matches, "Committed total matches");

        Ok(())
    }

    pub fn stats(&self) -> Result<CommittedStringFieldStats> {
        let key_count = self.index.len();
        Ok(CommittedStringFieldStats {
            key_count,
            global_info: self.global_info(),
        })
    }
}

#[derive(Debug)]
struct PostingIdStorage {
    // id -> (doc_id, (exact positions, stemmed positions))
    inner: Map<u64, Vec<(DocumentId, (Vec<usize>, Vec<usize>))>>,
}
impl PostingIdStorage {
    fn from_map(inner: Map<u64, Vec<(DocumentId, (Vec<usize>, Vec<usize>))>>) -> Self {
        Self { inner }
    }

    fn load(file_path: PathBuf) -> Result<Self> {
        Ok(Self {
            inner: Map::load(file_path)?,
        })
    }

    fn commit(&self) -> Result<()> {
        self.inner.commit()
    }

    fn get_posting(
        &self,
        posting_id: &u64,
    ) -> Option<&Vec<(DocumentId, (Vec<usize>, Vec<usize>))>> {
        self.inner.get(posting_id)
    }

    fn get_backed_file(&self) -> PathBuf {
        self.inner.file_path()
    }

    fn get_max_posting_id(&self) -> u64 {
        self.inner.get_max_key().copied().unwrap_or(0)
    }

    fn insert(&mut self, posting_id: u64, posting: Vec<(DocumentId, (Vec<usize>, Vec<usize>))>) {
        self.inner.insert(posting_id, posting);
    }

    fn remove_doc_ids(&mut self, doc_ids: &HashSet<DocumentId>) {
        self.inner.remove_inner_keys(doc_ids);
    }

    fn merge(
        &mut self,
        posting_id: u64,
        posting: impl Iterator<Item = (DocumentId, (Vec<usize>, Vec<usize>))>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
    ) {
        self.inner
            .merge(posting_id, posting, uncommitted_document_deletions);
    }
}

#[derive(Debug)]
struct DocumentLengthsPerDocument {
    inner: Map<DocumentId, u32>,
    global_info: GlobalInfo,
}
impl DocumentLengthsPerDocument {
    fn from_map(inner: Map<DocumentId, u32>) -> Self {
        let total_documents = inner.len();
        let total_document_length = inner.values().map(|v| *v as usize).sum();
        Self {
            inner,
            global_info: GlobalInfo {
                total_documents,
                total_document_length,
            },
        }
    }

    fn load(file_path: PathBuf) -> Result<Self> {
        let inner = Map::load(file_path)?;
        Ok(Self::from_map(inner))
    }

    fn commit(&self) -> Result<()> {
        self.inner.commit()
    }

    fn get_length(&self, doc_id: &DocumentId) -> u32 {
        self.inner.get(doc_id).copied().unwrap_or(1)
    }

    fn get_backed_file(&self) -> PathBuf {
        self.inner.file_path()
    }

    fn remove_doc_ids(&mut self, doc_ids: &HashSet<DocumentId>) {
        for doc_id in doc_ids {
            self.global_info.total_documents -= 1;
            if let Some(a) = self.inner.remove(doc_id) {
                self.global_info.total_document_length -= a as usize;
            }
        }
    }

    fn insert(&mut self, doc_id: DocumentId, len: u32) {
        self.global_info.total_documents += 1;
        self.global_info.total_document_length += len as usize;
        self.inner.insert(doc_id, len);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StringFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct CommittedStringFieldStats {
    pub key_count: usize,
    pub global_info: GlobalInfo,
}
