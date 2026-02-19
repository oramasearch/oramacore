use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use invocation_counter::InvocationCounter;
use oramacore_lib::data_structures::fst::FSTIndex;
use oramacore_lib::{data_structures::map::Map, filters::FilterResult};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    collection_manager::{
        bm25::{BM25FFieldParams, BM25Scorer},
        global_info::GlobalInfo,
        sides::read::{
            index::{
                committed_field::offload_utils::{
                    create_counter, should_offload, update_invocation_counter, Cow,
                },
                merge::{CommittedField, CommittedFieldMetadata, Field},
                uncommitted_field::UncommittedStringField,
            },
            OffloadFieldConfig,
        },
    },
    lock::OramaSyncLock,
    merger::MergedIterator,
    types::{DocumentId, FieldId},
};

const EXACT_MATCH_BOOST_MULTIPLIER: f32 = 3.0;

pub struct StringSearchParams<'index, 'search> {
    pub global_info: GlobalInfo,
    pub exact_match: bool,
    pub filtered_doc_ids: Option<&'index FilterResult<DocumentId>>,
    pub field_id: FieldId,
    pub boost: f32,
    pub tokens: &'search [String],
    pub tolerance: Option<u8>,
}

enum StringLayout {
    Fst(Box<(FSTIndex, PostingIdStorage, DocumentLengthsPerDocument)>),
}
impl StringLayout {
    fn key_count(&self) -> usize {
        match self {
            StringLayout::Fst(boxed) => boxed.0.len(),
        }
    }

    fn global_info(&self) -> GlobalInfo {
        match self {
            StringLayout::Fst(boxed) => boxed.2.get_global_info().clone(),
        }
    }

    fn search_without_phrase_match(
        &self,
        params: &StringSearchParams<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
    ) -> Result<()> {
        let total_field_length = params.global_info.total_document_length as f32;
        let total_documents_with_field = params.global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;

        let (index, posting_storage, document_lengths_per_document) = match self {
            StringLayout::Fst(boxed) => (&boxed.0, &boxed.1, &boxed.2),
        };

        for token in params.tokens {
            let iter: Box<dyn Iterator<Item = (bool, u64)>> = if params.exact_match {
                if let Some(posting_list_id) = index.search_exact(token) {
                    Box::new(vec![(true, posting_list_id)].into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                index
                    .search(token, params.tolerance)
                    .context("Cannot search in index")?
            };

            let matches = iter
                .flat_map(|(is_exact, posting_list_id)| {
                    posting_storage
                        .get_posting(&posting_list_id)
                        .into_iter()
                        .map(move |posting| (is_exact, posting))
                })
                .flat_map(|(is_exact, postings)| {
                    let exact_match = params.exact_match;

                    postings
                        .iter()
                        .filter(move |(doc_id, _)| {
                            params
                                .filtered_doc_ids
                                .is_none_or(|filtered| filtered.contains(doc_id))
                        })
                        .filter_map(move |(doc_id, positions)| {
                            let field_length = document_lengths_per_document.get_length(doc_id);
                            let term_occurrence_in_field = if exact_match {
                                positions.0.len() as u32
                            } else {
                                (positions.0.len() + positions.1.len()) as u32
                            };

                            if term_occurrence_in_field == 0 {
                                return None;
                            }

                            Some((doc_id, term_occurrence_in_field, field_length, is_exact))
                        })
                });

            for (doc_id, term_occurrence_in_field, field_length, is_exact) in matches {
                let field_id = params.field_id;

                // User-defined field boost as BM25F weight
                // But we boost more if the match is exact
                let weight = if is_exact {
                    params.boost * EXACT_MATCH_BOOST_MULTIPLIER
                } else {
                    params.boost
                };
                let field_params = BM25FFieldParams {
                    weight,
                    b: 0.75, // Default normalization parameter
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
        params: &StringSearchParams<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
    ) -> Result<()> {
        let total_field_length = params.global_info.total_document_length as f32;
        let total_documents_with_field = params.global_info.total_documents as f32;
        let average_field_length = total_field_length / total_documents_with_field;
        let (index, posting_storage, document_lengths_per_document) = match self {
            StringLayout::Fst(boxed) => (&boxed.0, &boxed.1, &boxed.2),
        };

        struct PhraseMatchStorage {
            positions: HashSet<usize>,
            matches: Vec<(u32, usize, usize, bool)>,
            token_indexes: u32,
        }
        let mut storage: HashMap<DocumentId, PhraseMatchStorage> = HashMap::new();

        for (token_index, token) in params.tokens.iter().enumerate() {
            let iter: Box<dyn Iterator<Item = (bool, u64)>> = if params.exact_match {
                if let Some(posting_list_id) = index.search_exact(token) {
                    Box::new(vec![(true, posting_list_id)].into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            } else {
                index
                    .search(token, params.tolerance)
                    .context("Cannot search in index")?
            };

            let iter = iter
                .filter_map(|(is_exact, posting_id)| {
                    posting_storage
                        .get_posting(&posting_id)
                        .map(move |posting| (is_exact, posting))
                })
                .flat_map(|(is_exact, postings)| {
                    let total_documents_with_term_in_field = postings.len();

                    let exact_match = params.exact_match;

                    postings
                        .iter()
                        .filter(move |(doc_id, _)| {
                            params.filtered_doc_ids.is_none_or(|f| f.contains(doc_id))
                        })
                        .filter_map(move |(doc_id, positions)| {
                            let field_lenght = document_lengths_per_document.get_length(doc_id);

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
                                is_exact,
                            ))
                        })
                });

            for (doc_id, field_length, positions, total_documents_with_term_in_field, is_exact) in
                iter
            {
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
                    is_exact,
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
            // 1. Boost for the exact match: implemented
            // 2. Boost for the phrase match when the terms appear in sequence (without holes): implemented
            // 3. Boost for the phrase match when the terms appear in sequence (with holes): not implemented
            // 4. Boost for the phrase match when the terms appear in any order: implemented
            // 5. Boost defined by the user: implemented as BM25F field weight
            // We should allow the user to configure which boost to use and how much it impacts the score.
            // TODO: think about this
            let boost_any_order = positions.len() as f32;
            let boost_sequence = sequences_count as f32 * 2.0;
            let phrase_boost = boost_any_order + boost_sequence;

            let field_id = params.field_id;

            let mut field_params = BM25FFieldParams {
                weight: params.boost * phrase_boost, // User-defined field boost as BM25F weight
                b: 0.75, // Default normalization parameter @todo: make this configurable?
            };

            for (
                field_length,
                term_occurrence_in_field,
                _total_documents_with_term_in_field,
                is_exact,
            ) in matches
            {
                if is_exact {
                    field_params.weight *= EXACT_MATCH_BOOST_MULTIPLIER
                }

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

enum StringStatus {
    Loaded(StringLayout),
    Unloaded,
}

pub struct CommittedStringField {
    metadata: StringFieldInfo,
    stats: CommittedStringFieldStats,
    status: OramaSyncLock<StringStatus>,
    invocation_counter: InvocationCounter,
    unload_window: Duration,
}

impl CommittedStringField {
    fn unload(&self) {
        let lock = self.status.read("unload_if_not_used").unwrap();
        // This field is already unloaded. Skip.
        if let StringStatus::Unloaded = &**lock {
            return;
        }
        drop(lock); // Release the read lock before unloading

        let mut lock = self.status.write("unload").unwrap();
        // Double check if another thread unloaded the field meanwhile.
        if let StringStatus::Unloaded = &**lock {
            return;
        }

        self.stats
            .loaded
            .store(false, std::sync::atomic::Ordering::Release);

        **lock = StringStatus::Unloaded;
    }

    pub fn unload_if_not_used(&self) {
        // Some invocations happened recently, do not unload.
        if should_offload(&self.invocation_counter, self.unload_window) {
            self.unload();
        }
    }

    pub fn search(
        &self,
        params: &StringSearchParams<'_, '_>,
        scorer: &mut BM25Scorer<DocumentId>,
    ) -> Result<()> {
        update_invocation_counter(&self.invocation_counter);
        if params.tokens.is_empty() {
            return Ok(());
        }

        let lock = self.status.read("search").unwrap();
        let lock = if let StringStatus::Unloaded = &**lock {
            drop(lock); // Release the read lock before loading

            let layout = load_layout(&self.metadata.data_dir)?;
            let mut write_lock = self.status.write("load").unwrap();
            **write_lock = StringStatus::Loaded(layout);
            self.stats
                .loaded
                .store(true, std::sync::atomic::Ordering::Release);
            drop(write_lock); // Release the write lock

            self.status.read("search").unwrap()
        } else {
            lock
        };
        let vector_status = &**lock;
        let layout = match vector_status {
            StringStatus::Loaded(layout) => layout,
            StringStatus::Unloaded => {
                // This never happens because of the logic above.
                return Err(anyhow!("Cannot search unloaded vector field"));
            }
        };

        if params.tokens.len() == 1 {
            layout.search_without_phrase_match(params, scorer)
        } else {
            layout.search_with_phrase_match(params, scorer)
        }
    }
}

impl CommittedField for CommittedStringField {
    type FieldMetadata = StringFieldInfo;
    type Uncommitted = UncommittedStringField;

    fn try_load(metadata: StringFieldInfo, offload_config: OffloadFieldConfig) -> Result<Self> {
        let layout = load_layout(&metadata.data_dir).context("Cannot load vector layout")?;

        Ok(Self {
            metadata,
            stats: CommittedStringFieldStats {
                key_count: layout.key_count(),
                global_info: layout.global_info(),
                loaded: AtomicBool::new(true),
            },
            status: OramaSyncLock::new("string_inner", StringStatus::Loaded(layout)),
            invocation_counter: create_counter(offload_config),
            unload_window: offload_config.unload_window.into(),
        })
    }

    fn from_uncommitted(
        uncommitted: &UncommittedStringField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let mut length_per_documents = uncommitted.field_length_per_doc();
        let mut posting_id_generator = 0;

        let mut delta_committed_storage: HashMap<u64, Vec<(DocumentId, PostingIdPosition)>> =
            Default::default();
        let iter = uncommitted.iter().map(|(key, value)| {
            let new_posting_list_id = posting_id_generator;
            posting_id_generator += 1;

            delta_committed_storage.insert(
                new_posting_list_id,
                value
                    .1
                    .iter()
                    .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id))
                    .map(|(doc_id, positions)| {
                        (*doc_id, (positions.0 .0.clone(), positions.1 .0.clone()))
                    })
                    .collect(),
            );

            (key, new_posting_list_id)
        });

        for doc_id in uncommitted_document_deletions {
            length_per_documents.remove(doc_id);
        }

        let posting_id_storage_file_path = data_dir.join(POSTING_ID_INDEX_FILE_NAME);
        let length_per_documents_file_path = data_dir.join(DOCUMENT_LENGTHS_PER_DOCUMENT_FILE_NAME);
        let fst_file_path = data_dir.join(FST_INDEX_FILE_NAME);

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
            metadata: StringFieldInfo {
                field_path: uncommitted.field_path().to_vec().into_boxed_slice(),
                data_dir,
            },
            stats: CommittedStringFieldStats {
                key_count: index.len(),
                global_info: document_lengths_per_document.get_global_info().clone(),
                loaded: AtomicBool::new(true),
            },
            status: OramaSyncLock::new(
                "string_inner",
                StringStatus::Loaded(StringLayout::Fst(Box::new((
                    index,
                    posting_storage,
                    document_lengths_per_document,
                )))),
            ),
            unload_window: offload_config.unload_window.into(),
            invocation_counter: create_counter(offload_config),
        })
    }

    fn add_uncommitted(
        &self,
        uncommitted: &UncommittedStringField,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let length_per_documents = uncommitted.field_length_per_doc();

        let new_posting_storage_file = data_dir.join(POSTING_ID_INDEX_FILE_NAME);

        let lock = self.status.read("commit").unwrap();
        let vector_status = &**lock;
        let current_layout = match vector_status {
            StringStatus::Loaded(layout) => Cow::Borrowed(layout),
            StringStatus::Unloaded => Cow::Owned(load_layout(&self.metadata.data_dir)?),
        };

        let old_posting_storage_file = match &*current_layout {
            StringLayout::Fst(boxed) => boxed.1.get_backed_file(),
        };
        if old_posting_storage_file != new_posting_storage_file {
            // `std::fs::copy` has a problem when source and destination are the same file.
            // So we check first if they are different.
            // Otherwise, the file will be trunked.
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

        let committed_iter = match &*current_layout {
            StringLayout::Fst(boxed) => boxed.0.iter(),
        };

        let merged_iterator = MergedIterator::new(
            uncommitted.iter(),
            committed_iter,
            |_, (_, positions_per_document_id)| {
                let new_posting_list_id = posting_id_generator;
                posting_id_generator += 1;

                let mut lock = posting_storage.borrow_mut();
                lock.insert(
                    new_posting_list_id,
                    positions_per_document_id
                        .iter()
                        .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id))
                        .map(|(doc_id, positions)| {
                            (*doc_id, (positions.0 .0.clone(), positions.1 .0.clone()))
                        })
                        .collect(),
                );

                new_posting_list_id
            },
            |_, (_, positions_per_document_id), committed_posting_id| {
                let mut lock = posting_storage.borrow_mut();
                lock.merge(
                    committed_posting_id,
                    positions_per_document_id.iter().map(|(doc_id, positions)| {
                        (*doc_id, (positions.0 .0.clone(), positions.1 .0.clone()))
                    }),
                    uncommitted_document_deletions,
                );

                committed_posting_id
            },
        );
        let fst_file_path = data_dir.join("fst.map");
        let index =
            FSTIndex::from_iter(merged_iterator, fst_file_path).context("Cannot commit fst")?;

        let new_document_lengths_per_document_file = data_dir.join("length_per_documents.map");
        let old_document_lengths_per_document_file = match &*current_layout {
            StringLayout::Fst(boxed) => boxed.2.get_backed_file(),
        };
        if old_document_lengths_per_document_file != new_document_lengths_per_document_file {
            // `std::fs::copy` has a problem when source and destination are the same file.
            // So we check first if they are different.
            // Otherwise, the file will be trunked.
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
            metadata: StringFieldInfo {
                field_path: self.metadata.field_path.clone(),
                data_dir,
            },
            stats: CommittedStringFieldStats {
                key_count: index.len(),
                global_info: document_lengths_per_document.get_global_info().clone(),
                loaded: AtomicBool::new(true),
            },
            status: OramaSyncLock::new(
                "string_inner",
                StringStatus::Loaded(StringLayout::Fst(Box::new((
                    index,
                    posting_storage,
                    document_lengths_per_document,
                )))),
            ),
            unload_window: offload_config.unload_window.into(),
            invocation_counter: create_counter(offload_config),
        })
    }

    fn metadata(&self) -> StringFieldInfo {
        self.metadata.clone()
    }
}

impl Field for CommittedStringField {
    type FieldStats = CommittedStringFieldStats;

    fn field_path(&self) -> &[String] {
        self.metadata.field_path.as_ref()
    }

    fn stats(&self) -> CommittedStringFieldStats {
        self.stats.clone()
    }
}

// (exact positions, stemmed positions)
type PostingIdPosition = (Vec<usize>, Vec<usize>);

#[derive(Serialize, Debug)]
pub struct CommittedStringFieldStats {
    pub key_count: usize,
    pub global_info: GlobalInfo,
    pub loaded: AtomicBool,
}

impl Clone for CommittedStringFieldStats {
    fn clone(&self) -> Self {
        Self {
            key_count: self.key_count,
            global_info: self.global_info.clone(),
            loaded: AtomicBool::new(self.loaded.load(std::sync::atomic::Ordering::Acquire)),
        }
    }
}

const FST_INDEX_FILE_NAME: &str = "fst.map";
const POSTING_ID_INDEX_FILE_NAME: &str = "posting_id_storage.map";
const DOCUMENT_LENGTHS_PER_DOCUMENT_FILE_NAME: &str = "length_per_documents.map";

fn load_layout(data_dir: &Path) -> Result<StringLayout> {
    let index = FSTIndex::load(data_dir.join(FST_INDEX_FILE_NAME))?;
    let posting_storage = PostingIdStorage::load(data_dir.join(POSTING_ID_INDEX_FILE_NAME))?;
    let document_lengths_per_document =
        DocumentLengthsPerDocument::load(data_dir.join(DOCUMENT_LENGTHS_PER_DOCUMENT_FILE_NAME))?;

    Ok(StringLayout::Fst(Box::new((
        index,
        posting_storage,
        document_lengths_per_document,
    ))))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StringFilterFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Debug)]
pub struct DocumentLengthsPerDocument {
    inner: Map<DocumentId, u32>,
    global_info: GlobalInfo,
}
impl DocumentLengthsPerDocument {
    pub fn from_map(inner: Map<DocumentId, u32>) -> Self {
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

    pub fn load(file_path: PathBuf) -> Result<Self> {
        let inner = Map::load(file_path)?;
        Ok(Self::from_map(inner))
    }

    pub fn get_global_info(&self) -> &GlobalInfo {
        &self.global_info
    }

    pub fn commit(&self) -> Result<()> {
        self.inner.commit()
    }

    pub fn get_length(&self, doc_id: &DocumentId) -> u32 {
        self.inner.get(doc_id).copied().unwrap_or(1)
    }

    pub fn get_backed_file(&self) -> PathBuf {
        self.inner.file_path()
    }

    pub fn remove_doc_ids(&mut self, doc_ids: &HashSet<DocumentId>) {
        for doc_id in doc_ids {
            if let Some(length) = self.inner.remove(doc_id) {
                self.global_info.total_documents -= 1;
                self.global_info.total_document_length -= length as usize;
            }
        }
    }

    pub fn insert(&mut self, doc_id: DocumentId, len: u32) {
        self.global_info.total_documents += 1;
        self.global_info.total_document_length += len as usize;
        self.inner.insert(doc_id, len);
    }
}

#[derive(Debug)]
pub struct PostingIdStorage {
    // id -> (doc_id, (exact positions, stemmed positions))
    inner: Map<u64, Vec<(DocumentId, PostingIdPosition)>>,
}
impl PostingIdStorage {
    pub fn from_map(inner: Map<u64, Vec<(DocumentId, PostingIdPosition)>>) -> Self {
        Self { inner }
    }

    pub fn load(file_path: PathBuf) -> Result<Self> {
        Ok(Self {
            inner: Map::load(file_path)?,
        })
    }

    pub fn commit(&self) -> Result<()> {
        self.inner.commit()
    }

    pub fn get_posting(&self, posting_id: &u64) -> Option<&Vec<(DocumentId, PostingIdPosition)>> {
        self.inner.get(posting_id)
    }

    pub fn get_backed_file(&self) -> PathBuf {
        self.inner.file_path()
    }

    pub fn get_max_posting_id(&self) -> u64 {
        self.inner.get_max_key().copied().unwrap_or(0)
    }

    pub fn insert(&mut self, posting_id: u64, posting: Vec<(DocumentId, PostingIdPosition)>) {
        self.inner.insert(posting_id, posting);
    }

    pub fn remove_doc_ids(&mut self, doc_ids: &HashSet<DocumentId>) {
        self.inner.remove_inner_keys(doc_ids);
    }

    pub fn merge(
        &mut self,
        posting_id: u64,
        posting: impl Iterator<Item = (DocumentId, PostingIdPosition)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
    ) {
        self.inner
            .merge(posting_id, posting, uncommitted_document_deletions);
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StringFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

impl CommittedFieldMetadata for StringFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
    fn field_path(&self) -> &[String] {
        self.field_path.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collection_manager::sides::{
            read::index::uncommitted_field::UncommittedStringField, Term, TermStringField,
        },
        tests::utils::{generate_new_path, init_log},
    };

    use super::*;

    #[test]
    fn test_offload_string_field() {
        init_log();

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

        let search_params = StringSearchParams {
            tokens: &["term1".to_string()],
            exact_match: false,
            boost: 1.0,
            field_id: FieldId(1), // Use test field ID
            global_info: uncommitted.global_info(),
            filtered_doc_ids: None,
            tolerance: None,
        };

        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::plain();
        uncommitted.search(&search_params, &mut scorer).unwrap();

        // Finalize the BM25F scoring process
        scorer.finalize_term_plain(
            1,   // corpus_df - at least 1 document contains the term
            2.0, // total_documents in test data
            1.2, // k parameter
            1.0, // phrase boost
        );
        assert_eq!(scorer.get_scores().len(), 2);

        let data_dir = generate_new_path();

        let committed = CommittedStringField::from_uncommitted(
            &uncommitted,
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
        committed.search(&search_params, &mut scorer).unwrap();

        // Finalize the BM25F scoring process
        scorer.finalize_term_plain(
            1,   // corpus_df
            2.0, // total_documents
            1.2, // k parameter
            1.0, // phrase boost
        );

        assert_eq!(scorer.get_scores().len(), 2);

        committed.unload(); // force unloading

        let status_lock = committed.status.read("test").unwrap();
        let status = &**status_lock;
        assert!(matches!(status, StringStatus::Unloaded));
        drop(status_lock);

        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::plain();
        committed.search(&search_params, &mut scorer).unwrap();

        // Finalize the BM25F scoring process
        scorer.finalize_term_plain(
            1,   // corpus_df
            2.0, // total_documents
            1.2, // k parameter
            1.0, // phrase boost
        );

        assert_eq!(scorer.get_scores().len(), 2);
    }

    // ---- Scoring parity tests ----
    // These tests verify that BM25 scores from oramacore match those from oramacore_fields
    // when given identical data and parameters.
    //
    // IMPORTANT: oramacore derives per-doc field_length as max(stemmed_positions) + 1.
    // All test data satisfies: field_length_param = max(stemmed_positions) + 1.
    //
    // Run corresponding oramacore_fields tests:
    //   cd /path/to/oramacore_fields && cargo test test_scoring_parity -- --nocapture

    fn commit_and_search(
        uncommitted: &UncommittedStringField,
        tokens: &[String],
        exact_match: bool,
        boost: f32,
        tolerance: Option<u8>,
        corpus_df: usize,
    ) -> HashMap<DocumentId, f32> {
        let data_dir = generate_new_path();
        let committed = CommittedStringField::from_uncommitted(
            uncommitted,
            data_dir,
            &HashSet::new(),
            crate::collection_manager::sides::read::OffloadFieldConfig {
                unload_window: std::time::Duration::from_secs(30 * 60).into(),
                slot_count_exp: 8,
                slot_size_exp: 4,
            },
        )
        .unwrap();

        let global_info = committed.stats().global_info;
        let total_documents = global_info.total_documents as f32;

        let search_params = StringSearchParams {
            tokens,
            exact_match,
            boost,
            field_id: FieldId(1),
            global_info,
            filtered_doc_ids: None,
            tolerance,
        };

        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::plain();
        committed.search(&search_params, &mut scorer).unwrap();
        scorer.finalize_term_plain(corpus_df, total_documents, 1.2, 1.0);
        scorer.get_scores()
    }

    #[test]
    fn test_scoring_parity_basic_bm25() {
        init_log();

        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        // Doc1: field_length=5, "term1" exact=[0,1] stemmed=[0,4]
        // oramacore doc_length = max(stemmed=[0,4])+1 = 5
        uncommitted.insert(
            DocumentId(1),
            5,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0, 1],
                    positions: vec![0, 4],
                },
            )]),
        );
        // Doc2: field_length=3, "term1" exact=[0] stemmed=[2]
        // oramacore doc_length = max(stemmed=[2])+1 = 3
        uncommitted.insert(
            DocumentId(2),
            3,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0],
                    positions: vec![2],
                },
            )]),
        );

        let tokens = vec!["term1".to_string()];
        // corpus_df = 2 (both docs contain "term1")
        let scores = commit_and_search(&uncommitted, &tokens, false, 1.0, None, 2);

        let score1 = scores[&DocumentId(1)];
        let score2 = scores[&DocumentId(2)];

        println!("test_scoring_parity_basic_bm25:");
        println!("  doc1 score = {score1:.10}");
        println!("  doc2 score = {score2:.10}");

        assert!(
            score1 > score2,
            "Doc1 (tf=4) should rank higher than Doc2 (tf=2)"
        );
        // Cross-repo parity: these values must match oramacore_fields' test_scoring_parity_basic_bm25
        assert!(
            (score1 - 0.358_531_8).abs() < 1e-6,
            "doc1 score mismatch: {score1}"
        );
        assert!(
            (score2 - 0.34503868).abs() < 1e-6,
            "doc2 score mismatch: {score2}"
        );
    }

    #[test]
    fn test_scoring_parity_exact_match() {
        init_log();

        // NOTE: oramacore always applies 3x exact_match_boost when search_exact returns
        // is_exact=true. oramacore_fields with tolerance=Some(0) does NOT apply the boost.
        // Scores here will be higher than the oramacore_fields counterpart.
        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        uncommitted.insert(
            DocumentId(1),
            5,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0, 1],
                    positions: vec![0, 4],
                },
            )]),
        );
        uncommitted.insert(
            DocumentId(2),
            3,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0],
                    positions: vec![2],
                },
            )]),
        );

        let tokens = vec!["term1".to_string()];
        // exact_match=true: oramacore uses search_exact, tolerance is ignored
        let scores = commit_and_search(&uncommitted, &tokens, true, 1.0, None, 2);

        let score1 = scores[&DocumentId(1)];
        let score2 = scores[&DocumentId(2)];

        println!("test_scoring_parity_exact_match:");
        println!("  doc1 score = {score1:.10} (oramacore_fields will be lower, no 3x boost)");
        println!("  doc2 score = {score2:.10} (oramacore_fields will be lower, no 3x boost)");

        // Doc1: tf=2 (exact only), Doc2: tf=1 (exact only), with 3x boost
        assert!(score1 > score2);
        // These are HIGHER than oramacore_fields (0.2342, 0.2031) due to 3x exact_match_boost
        assert!(
            (score1 - 0.32412723).abs() < 1e-6,
            "doc1 score mismatch: {score1}"
        );
        assert!(
            (score2 - 0.302_722_6).abs() < 1e-6,
            "doc2 score mismatch: {score2}"
        );
    }

    #[test]
    fn test_scoring_parity_field_boost() {
        init_log();

        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        uncommitted.insert(
            DocumentId(1),
            5,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0, 1],
                    positions: vec![0, 4],
                },
            )]),
        );
        uncommitted.insert(
            DocumentId(2),
            3,
            HashMap::from([(
                Term("term1".to_string()),
                TermStringField {
                    exact_positions: vec![0],
                    positions: vec![2],
                },
            )]),
        );

        let tokens = vec!["term1".to_string()];
        // boost=2.0
        let scores = commit_and_search(&uncommitted, &tokens, false, 2.0, None, 2);

        let score1 = scores[&DocumentId(1)];
        let score2 = scores[&DocumentId(2)];

        println!("test_scoring_parity_field_boost:");
        println!("  doc1 score = {score1:.10}");
        println!("  doc2 score = {score2:.10}");

        assert!(score1 > score2);
        // Cross-repo parity: must match oramacore_fields' test_scoring_parity_field_boost
        assert!(
            (score1 - 0.37862647).abs() < 1e-6,
            "doc1 score mismatch: {score1}"
        );
        assert!(
            (score2 - 0.37096643).abs() < 1e-6,
            "doc2 score mismatch: {score2}"
        );
    }

    #[test]
    fn test_scoring_parity_single_doc() {
        init_log();

        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        // Doc1: field_length=3, "hello" exact=[0] stemmed=[2]
        // oramacore doc_length = max(stemmed=[2])+1 = 3
        uncommitted.insert(
            DocumentId(1),
            3,
            HashMap::from([(
                Term("hello".to_string()),
                TermStringField {
                    exact_positions: vec![0],
                    positions: vec![2],
                },
            )]),
        );

        let tokens = vec!["hello".to_string()];
        // corpus_df = 1
        let scores = commit_and_search(&uncommitted, &tokens, false, 1.0, None, 1);

        let score1 = scores[&DocumentId(1)];

        println!("test_scoring_parity_single_doc:");
        println!("  doc1 score = {score1:.10}");

        // Cross-repo parity: must match oramacore_fields' test_scoring_parity_single_doc
        assert!(
            (score1 - 0.527_417_2).abs() < 1e-6,
            "doc1 score mismatch: {score1}"
        );
    }

    #[test]
    fn test_scoring_parity_stemmed_only() {
        init_log();

        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        // Doc1: field_length=4, "run" exact=[] stemmed=[0,1,3]
        // oramacore doc_length = max(stemmed=[0,1,3])+1 = 4
        uncommitted.insert(
            DocumentId(1),
            4,
            HashMap::from([(
                Term("run".to_string()),
                TermStringField {
                    exact_positions: vec![],
                    positions: vec![0, 1, 3],
                },
            )]),
        );

        let tokens = vec!["run".to_string()];

        // exact_match=false → tf=3 (stemmed only), corpus_df=1
        let scores = commit_and_search(&uncommitted, &tokens, false, 1.0, None, 1);

        let score1 = scores[&DocumentId(1)];
        println!("test_scoring_parity_stemmed_only (exact_match=false):");
        println!("  doc1 score = {score1:.10}");
        // Cross-repo parity: must match oramacore_fields' test_scoring_parity_stemmed_only
        assert!(
            (score1 - 0.558_441_7).abs() < 1e-6,
            "doc1 score mismatch: {score1}"
        );

        // exact_match=true → tf=0, no results
        let scores = commit_and_search(&uncommitted, &tokens, true, 1.0, None, 0);

        println!("test_scoring_parity_stemmed_only (exact_match=true):");
        println!("  scores = {scores:?}");

        assert!(
            scores.is_empty(),
            "exact_match=true with stemmed-only should yield no results"
        );
    }

    #[test]
    fn test_scoring_parity_varying_lengths() {
        init_log();

        let path = Box::new(["test".to_string()]);
        let mut uncommitted = UncommittedStringField::empty(path);
        // Doc1: field_length=10, "word" exact=[0,1,2] stemmed=[9]
        // oramacore doc_length = max(stemmed=[9])+1 = 10, tf=4
        uncommitted.insert(
            DocumentId(1),
            10,
            HashMap::from([(
                Term("word".to_string()),
                TermStringField {
                    exact_positions: vec![0, 1, 2],
                    positions: vec![9],
                },
            )]),
        );
        // Doc2: field_length=2, "word" exact=[0] stemmed=[1]
        // oramacore doc_length = max(stemmed=[1])+1 = 2, tf=2
        uncommitted.insert(
            DocumentId(2),
            2,
            HashMap::from([(
                Term("word".to_string()),
                TermStringField {
                    exact_positions: vec![0],
                    positions: vec![1],
                },
            )]),
        );
        // Doc3: field_length=20, "word" exact=[0,1] stemmed=[0,19]
        // oramacore doc_length = max(stemmed=[0,19])+1 = 20, tf=4
        uncommitted.insert(
            DocumentId(3),
            20,
            HashMap::from([(
                Term("word".to_string()),
                TermStringField {
                    exact_positions: vec![0, 1],
                    positions: vec![0, 19],
                },
            )]),
        );

        let tokens = vec!["word".to_string()];
        // corpus_df = 3 (all docs contain "word")
        let scores = commit_and_search(&uncommitted, &tokens, false, 1.0, None, 3);

        let score1 = scores[&DocumentId(1)];
        let score2 = scores[&DocumentId(2)];
        let score3 = scores[&DocumentId(3)];

        println!("test_scoring_parity_varying_lengths:");
        println!("  doc1 (fl=10, tf=4) score = {score1:.10}");
        println!("  doc2 (fl=2,  tf=2) score = {score2:.10}");
        println!("  doc3 (fl=20, tf=4) score = {score3:.10}");

        // Cross-repo parity: must match oramacore_fields' test_scoring_parity_varying_lengths
        assert!(
            (score1 - 0.268_205_7).abs() < 1e-6,
            "doc1 score mismatch: {score1}"
        );
        assert!(
            (score2 - 0.27248147).abs() < 1e-6,
            "doc2 score mismatch: {score2}"
        );
        assert!(
            (score3 - 0.252_027_1).abs() < 1e-6,
            "doc3 score mismatch: {score3}"
        );
    }
}
