use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::RwLock,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::{
    collection_manager::{
        bm25::{BM25FFieldParams, BM25Scorer},
        global_info::GlobalInfo,
        sides::read::{
            index::{
                committed_field::offload_utils::{InnerCommittedField, LoadedField},
                search_context::FullTextSearchContext,
                uncommitted_field::{Positions, TotalDocumentsWithTermInField},
            },
            OffloadFieldConfig,
        },
    },
    indexes::{fst::FSTIndex, map::Map},
    merger::MergedIterator,
    types::DocumentId,
};
use fs::create_if_not_exists;

#[derive(Debug)]
pub struct CommittedStringField {
    inner: RwLock<
        InnerCommittedField<LoadedCommittedStringField, CommittedStringFieldStats, StringFieldInfo>,
    >,
}

impl CommittedStringField {
    pub fn try_load(info: StringFieldInfo, offload_config: OffloadFieldConfig) -> Result<Self> {
        let loaded = LoadedCommittedStringField::try_load(info)?;
        let inner = InnerCommittedField::new_loaded(loaded, offload_config);
        let inner = RwLock::new(inner);
        Ok(Self { inner })
    }

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
        length_per_documents: HashMap<DocumentId, u32>,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let loaded = LoadedCommittedStringField::from_iter(
            field_path,
            uncommitted_iter,
            length_per_documents,
            data_dir,
            uncommitted_document_deletions,
        )?;

        let inner = InnerCommittedField::new_loaded(loaded, offload_config);
        let inner = RwLock::new(inner);
        Ok(Self { inner })
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
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        committed.load(); // enforce loading of the committed field
        let lock = committed.inner.read().unwrap();
        let prev_loaded = match &*lock {
            InnerCommittedField::Loaded { field, .. } => field,
            InnerCommittedField::Unloaded { field_info, .. } => {
                return Err(anyhow::anyhow!(
                    "Cannot commit to an unloaded committed string field: {:?}",
                    field_info
                ));
            }
        };

        let loaded = LoadedCommittedStringField::from_iter_and_committed(
            field_path,
            uncommitted_iter,
            prev_loaded,
            length_per_documents,
            data_dir,
            uncommitted_document_deletions,
        )?;

        let inner = InnerCommittedField::new_loaded(loaded, offload_config);

        Ok(Self {
            inner: RwLock::new(inner),
        })
    }

    fn loaded(&self) -> bool {
        self.inner.read().unwrap().loaded()
    }

    fn load(&self) {
        if self.loaded() {
            return;
        }

        let mut inner = self.inner.write().unwrap();
        if let InnerCommittedField::Unloaded {
            offload_config,
            field_info,
            ..
        } = &*inner
        {
            let loaded = LoadedCommittedStringField::try_load(field_info.clone())
                .expect("Cannot load committed string field");
            *inner = InnerCommittedField::new_loaded(loaded, *offload_config)
        }
    }

    pub fn unload_if_not_used(&self) {
        let lock = self.inner.read().unwrap();
        if lock.should_unload() {
            drop(lock); // Release the read lock before unloading
            self.unload();
        }
    }

    fn unload(&self) {
        let mut inner = self.inner.write().unwrap();
        if let InnerCommittedField::Loaded {
            field,
            offload_config,
            ..
        } = &*inner
        {
            let field_path = field.field_path();
            debug!("Unloading committed string field {:?}", field_path,);
            let mut stats = field.stats();
            stats.loaded = false; // Mark field as unloaded
            *inner = InnerCommittedField::unloaded(*offload_config, stats, field.info());

            info!("Committed string field {:?} unloaded", field_path,);
        }
    }

    pub fn field_path(&self) -> Box<[String]> {
        let inner = self.inner.read().unwrap();
        inner.info().field_path.clone()
    }

    pub fn global_info(&self) -> GlobalInfo {
        if !self.loaded() {
            self.load();
        }

        let inner = self.inner.read().unwrap();
        match &*inner {
            InnerCommittedField::Loaded { field, .. } => field.global_info(),
            InnerCommittedField::Unloaded { .. } => GlobalInfo::default(),
        }
    }

    pub fn get_field_info(&self) -> StringFieldInfo {
        self.inner.read().unwrap().info()
    }

    pub fn stats(&self) -> CommittedStringFieldStats {
        let inner = self.inner.read().unwrap();
        inner.stats()
    }

    pub fn search(
        &self,
        context: &mut FullTextSearchContext<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
        tolerance: Option<u8>,
    ) -> Result<()> {
        self.load();

        let inner = self.inner.read().unwrap();

        if let Some(field) = inner.get_load_unchecked() {
            field.search(context, scorer, tolerance)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct LoadedCommittedStringField {
    field_path: Box<[String]>,
    index: FSTIndex,

    posting_storage: PostingIdStorage,
    document_lengths_per_document: DocumentLengthsPerDocument,

    data_dir: PathBuf,
}

impl LoadedCommittedStringField {
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

        let mut delta_committed_storage: HashMap<u64, Vec<(DocumentId, PostingIdPosition)>> =
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
                        "Cannot copy posting storage file: old {old_posting_storage_file:?} -> new {new_posting_storage_file:?}"
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

    pub fn field_path(&self) -> Box<[String]> {
        self.field_path.clone()
    }

    pub fn global_info(&self) -> GlobalInfo {
        self.document_lengths_per_document.global_info.clone()
    }

    pub fn search(
        &self,
        context: &mut FullTextSearchContext<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
        tolerance: Option<u8>,
    ) -> Result<()> {
        if context.tokens.is_empty() {
            return Ok(());
        }

        if context.tokens.len() == 1 {
            self.search_without_phrase_match(context, scorer, tolerance)
        } else {
            self.search_with_phrase_match(context, scorer, tolerance)
        }
    }

    fn search_without_phrase_match(
        &self,
        context: &mut FullTextSearchContext<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
        tolerance: Option<u8>,
    ) -> Result<()> {
        let total_field_length = context.global_info.total_document_length as f32;
        let total_documents_with_field = context.global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        for token in context.tokens {
            context.increment_term_count();

            let iter: Box<dyn Iterator<Item = u64>> = if context.exact_match {
                if let Some(posting_list_id) = self.index.search_exact(token) {
                    Box::new(vec![posting_list_id].into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                self.index
                    .search(token, tolerance)
                    .context("Cannot search in index")?
            };

            let matches = iter
                .flat_map(|posting_list_id| self.posting_storage.get_posting(&posting_list_id))
                .flat_map(|postings| {
                    let total_documents_with_term_in_field = postings.len();

                    let uncommitted_deleted_documents = context.uncommitted_deleted_documents;
                    let exact_match = context.exact_match;
                    let filtered_doc_ids = context.filtered_doc_ids;

                    postings
                        .iter()
                        .filter(move |(doc_id, _)| {
                            filtered_doc_ids
                                .is_none_or(|filtered_doc_ids| filtered_doc_ids.contains(doc_id))
                        })
                        .filter(|(doc_id, _)| !uncommitted_deleted_documents.contains(doc_id))
                        .filter_map(move |(doc_id, positions)| {
                            let field_length =
                                self.document_lengths_per_document.get_length(doc_id);
                            let term_occurrence_in_field = if exact_match {
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
                _total_documents_with_term_in_field,
            ) in matches
            {
                let field_id = context.field_id;
                let field_params = BM25FFieldParams {
                    weight: context.boost, // User-defined field boost as BM25F weight
                    b: 0.75,               // Default normalization parameter
                };

                scorer.add_field(
                    *doc_id,
                    field_id,
                    term_occurrence_in_field,
                    field_length,
                    average_field_length,
                    &field_params,
                );
            }
        }

        Ok(())
    }

    fn search_with_phrase_match(
        &self,
        context: &mut FullTextSearchContext<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
        tolerance: Option<u8>,
    ) -> Result<()> {
        let total_field_length = context.global_info.total_document_length as f32;
        let total_documents_with_field = context.global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        struct PhraseMatchStorage {
            positions: HashSet<usize>,
            matches: Vec<(u32, usize, usize)>,
            token_indexes: u32,
        }
        let mut storage: HashMap<DocumentId, PhraseMatchStorage> = HashMap::new();

        for (token_index, token) in context.tokens.iter().enumerate() {
            context.increment_term_count();

            let iter: Box<dyn Iterator<Item = u64>> = if context.exact_match {
                if let Some(posting_list_id) = self.index.search_exact(token) {
                    Box::new(vec![posting_list_id].into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                self.index
                    .search(token, tolerance)
                    .context("Cannot search in index")?
            };

            let iter = iter
                .filter_map(|posting_id| self.posting_storage.get_posting(&posting_id))
                .flat_map(|postings| {
                    let total_documents_with_term_in_field = postings.len();

                    let uncommitted_deleted_documents = context.uncommitted_deleted_documents;
                    let exact_match = context.exact_match;
                    let filtered_doc_ids = context.filtered_doc_ids;

                    postings
                        .iter()
                        .filter(move |(doc_id, _)| {
                            filtered_doc_ids
                                .is_none_or(|filtered_doc_ids| filtered_doc_ids.contains(doc_id))
                        })
                        .filter(|(doc_id, _)| !uncommitted_deleted_documents.contains(doc_id))
                        .filter_map(move |(doc_id, positions)| {
                            let field_lenght =
                                self.document_lengths_per_document.get_length(doc_id);

                            let positions: Vec<usize> = if exact_match {
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
                token_indexes: _,
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
            // 5. Boost defined by the user: implemented as BM25F field weight
            // We should allow the user to configure which boost to use and how much it impacts the score.
            // TODO: think about this
            let boost_any_order = positions.len() as f32;
            let boost_sequence = sequences_count as f32 * 2.0;
            let _phrase_boost = boost_any_order + boost_sequence;

            let field_id = context.field_id;

            let field_params = BM25FFieldParams {
                weight: context.boost, // User-defined field boost as BM25F weight
                b: 0.75, // Default normalization parameter @todo: make this configurable?
            };

            for (field_length, term_occurrence_in_field, _total_documents_with_term_in_field) in
                matches
            {
                scorer.add_field(
                    doc_id,
                    field_id,
                    term_occurrence_in_field as u32,
                    field_length,
                    average_field_length,
                    &field_params,
                );

                total_matches += 1;
            }
        }

        info!(total_matches = total_matches, "Committed total matches");

        Ok(())
    }
}

impl LoadedField for LoadedCommittedStringField {
    type Stats = CommittedStringFieldStats;
    type Info = StringFieldInfo;

    fn stats(&self) -> Self::Stats {
        let key_count = self.index.len();
        CommittedStringFieldStats {
            key_count,
            global_info: self.global_info(),
            loaded: true,
        }
    }

    fn info(&self) -> Self::Info {
        StringFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

// (exact positions, stemmed positions)
type PostingIdPosition = (Vec<usize>, Vec<usize>);

#[derive(Debug)]
struct PostingIdStorage {
    // id -> (doc_id, (exact positions, stemmed positions))
    inner: Map<u64, Vec<(DocumentId, PostingIdPosition)>>,
}
impl PostingIdStorage {
    fn from_map(inner: Map<u64, Vec<(DocumentId, PostingIdPosition)>>) -> Self {
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

    fn get_posting(&self, posting_id: &u64) -> Option<&Vec<(DocumentId, PostingIdPosition)>> {
        self.inner.get(posting_id)
    }

    fn get_backed_file(&self) -> PathBuf {
        self.inner.file_path()
    }

    fn get_max_posting_id(&self) -> u64 {
        self.inner.get_max_key().copied().unwrap_or(0)
    }

    fn insert(&mut self, posting_id: u64, posting: Vec<(DocumentId, PostingIdPosition)>) {
        self.inner.insert(posting_id, posting);
    }

    fn remove_doc_ids(&mut self, doc_ids: &HashSet<DocumentId>) {
        self.inner.remove_inner_keys(doc_ids);
    }

    fn merge(
        &mut self,
        posting_id: u64,
        posting: impl Iterator<Item = (DocumentId, PostingIdPosition)>,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StringFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug, Clone)]
pub struct CommittedStringFieldStats {
    pub key_count: usize,
    pub global_info: GlobalInfo,
    pub loaded: bool,
}

#[cfg(test)]
mod tests {
    use crate::{
        collection_manager::sides::{
            read::index::uncommitted_field::UncommittedStringField, Term, TermStringField,
        },
        tests::utils::generate_new_path,
    };

    use super::*;

    #[test]
    fn test_offload_string_field() {
        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        uncommitted.insert(
            DocumentId(1),
            5,
            HashMap::from([
                (
                    Term("term1".to_string()),
                    TermStringField {
                        exact_positions: vec![0, 1],
                        positions: vec![0, 1],
                    },
                ),
                (
                    Term("term2".to_string()),
                    TermStringField {
                        exact_positions: vec![2, 3],
                        positions: vec![2, 3],
                    },
                ),
            ]),
        );
        uncommitted.insert(
            DocumentId(2),
            3,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0],
                    positions: vec![0],
                },
            )]),
        );

        let mut search_context = FullTextSearchContext {
            tokens: &["term1".to_string()],
            exact_match: false,
            boost: 1.0,
            field_id: FieldId(1), // Use test field ID
            filtered_doc_ids: None,
            global_info: uncommitted.global_info(),
            uncommitted_deleted_documents: &HashSet::new(),
            total_term_count: 0,
        };
        use crate::types::FieldId;
        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::plain();
        uncommitted
            .search(&mut search_context, &mut scorer)
            .unwrap();

        assert_eq!(scorer.get_scores().len(), 2);

        let field_length_per_doc = uncommitted.field_length_per_doc();

        let data_dir = generate_new_path();

        let committed = CommittedStringField::from_iter(
            uncommitted.field_path().to_vec().into_boxed_slice(),
            uncommitted.iter().map(|(n, v)| (n, v.clone())),
            field_length_per_doc,
            data_dir.clone(),
            &HashSet::new(),
            crate::collection_manager::sides::read::OffloadFieldConfig {
                unload_window: std::time::Duration::from_secs(30 * 60).into(), // 30 minutes
                slot_count_exp: 8, // 2^8 = 256 time slots
                slot_size_exp: 4,  // 2^4 = 16 seconds per slot
            },
        )
        .unwrap();

        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::plain();
        committed
            .search(&mut search_context, &mut scorer, None)
            .unwrap();

        assert_eq!(scorer.get_scores().len(), 2);

        committed.unload(); // force unloading
        assert!(!committed.loaded());

        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::plain();
        committed
            .search(&mut search_context, &mut scorer, None)
            .unwrap();
        assert_eq!(scorer.get_scores().len(), 2);
    }
}
