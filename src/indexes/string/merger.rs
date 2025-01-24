use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{atomic::AtomicU64, Arc};

use anyhow::{Context, Result};
use dashmap::DashMap;
use fst::{Map, MapBuilder, Streamer};
use memmap::Mmap;

use crate::collection_manager::sides::Offset;
use crate::file_utils::BufferedFile;
use crate::merger::MergedIterator;
use crate::types::DocumentId;

use super::document_lengths::DocumentLengthsPerDocument;
use super::posting_storage::PostingIdStorage;
use super::uncommitted::{Positions, TotalDocumentsWithTermInField};
use super::{
    CommittedStringFieldIndex, GlobalInfo, StringIndexFieldInfo, UncommittedStringFieldIndex,
};

pub struct DataToCommit<'uncommitted> {
    pub index: &'uncommitted UncommittedStringFieldIndex,
    pub state_to_clear: bool,

    // the state
    pub total_field_length: u64,
    pub document_ids: HashSet<DocumentId>,
    pub field_length_per_doc: HashMap<DocumentId, u32>,
    #[allow(clippy::type_complexity)]
    pub tree: Vec<(
        Vec<u8>,
        (
            TotalDocumentsWithTermInField,
            HashMap<DocumentId, Positions>,
        ),
    )>,

    pub current_offset: Offset,
}

impl DataToCommit<'_> {
    pub fn get_document_lengths(&self) -> &HashMap<DocumentId, u32> {
        &self.field_length_per_doc
    }

    pub fn global_info(&self) -> GlobalInfo {
        GlobalInfo {
            total_document_length: self.total_field_length as usize,
            total_documents: self.document_ids.len(),
        }
    }

    pub fn get_offset(&self) -> Offset {
        self.current_offset
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            Vec<u8>,
            (
                TotalDocumentsWithTermInField,
                HashMap<DocumentId, Positions>,
            ),
        ),
    > + '_ {
        self.tree.iter().map(|(k, v)| (k.to_vec(), v.clone()))
    }

    pub async fn done(self) {
        let mut lock = self.index.inner.write().await;
        let tree = if self.state_to_clear {
            &mut lock.1.left
        } else {
            &mut lock.1.right
        };
        tree.clear();
    }
}

struct FTSIter<'stream> {
    stream: Option<fst::map::Stream<'stream>>,
}
impl Iterator for FTSIter<'_> {
    // The Item allocate memory, but we could avoid it by using a reference
    // TODO: resolve lifetime issue with reference here
    type Item = (Vec<u8>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let stream = match &mut self.stream {
            Some(stream) => stream,
            None => return None,
        };
        stream.next().map(|(key, value)| (key.to_vec(), value))
    }
}

pub async fn merge(
    data_to_commit: DataToCommit<'_>,
    committed: &CommittedStringFieldIndex,
    string_index_field_info: StringIndexFieldInfo,
) -> Result<()> {
    let StringIndexFieldInfo {
        document_length_path,
        fst_path,
        posting_path,
        global_info_path,
        posting_id_path,
        ..
    } = string_index_field_info;

    committed
        .document_lengths_per_document
        .merge(data_to_commit.get_document_lengths(), document_length_path)
        .context("Cannot merge document lengths")?;

    let max_posting_id = committed.posting_id_generator.load(Ordering::Relaxed);
    let posting_id_generator = Arc::new(AtomicU64::new(max_posting_id + 1));

    let storage_updates = tokio::task::block_in_place(|| {
        let uncommitted_iter = data_to_commit.iter();
        merge_iter(
            posting_id_generator.clone(),
            uncommitted_iter,
            committed.get_info().fst_path,
            fst_path,
        )
    })
    .context("Cannot merge iterators")?;

    committed
        .storage
        .apply_delta(storage_updates, posting_path)
        .await?;

    let global_info = data_to_commit.global_info() + committed.get_global_info();
    BufferedFile::create(global_info_path)
        .context("Cannot create file for global info")?
        .write_json_data(&global_info)
        .context("Cannot serialize global info to file")?;

    let posting_id = posting_id_generator.load(Ordering::Relaxed);
    BufferedFile::create(posting_id_path)
        .context("Cannot create file for posting_id")?
        .write_json_data(&posting_id)
        .context("Cannot serialize posting_id to file")?;

    data_to_commit.done().await;

    Ok(())
}

pub async fn create(
    data_to_commit: DataToCommit<'_>,
    string_index_field_info: StringIndexFieldInfo,
) -> Result<()> {
    let StringIndexFieldInfo {
        document_length_path,
        fst_path,
        posting_path,
        global_info_path,
        posting_id_path,
        ..
    } = string_index_field_info;

    let posting_id_generator = AtomicU64::new(0);

    DocumentLengthsPerDocument::create(data_to_commit.get_document_lengths(), document_length_path)
        .context("Cannot create file for document lengths")?;

    let uncommitted_iter = data_to_commit.iter();

    let mut delta_committed_storage: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> =
        Default::default();

    let r: Result<()> = tokio::task::block_in_place(|| {
        let mut buf = BufferedFile::create(fst_path.clone()).context("Cannot create fst file")?;
        let mut build = MapBuilder::new(&mut buf)?;

        for (key, value) in uncommitted_iter {
            let new_posting_list_id =
                posting_id_generator.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            delta_committed_storage.insert(
                new_posting_list_id,
                value
                    .1
                    .into_iter()
                    .map(|(doc_id, positions)| (doc_id, positions.0))
                    .collect(),
            );

            build
                .insert(key, new_posting_list_id)
                .context("Cannot insert value to FST map")?;
        }

        build.finish().context("Cannot finish build of FST map")?;

        buf.close().context("Cannot close buffered file")?;

        Ok(())
    });
    r?;

    PostingIdStorage::create(delta_committed_storage, posting_path)
        .context("Cannot create posting id storage")?;

    let global_info = data_to_commit.global_info();
    BufferedFile::create(global_info_path)
        .context("Cannot create global_info file")?
        .write_json_data(&global_info)
        .context("Cannot write global info to file")?;

    let posting_id = posting_id_generator.load(Ordering::Relaxed);
    BufferedFile::create(posting_id_path)
        .context("Cannot create posting_id file")?
        .write_json_data(&posting_id)
        .context("Cannot write posting_id to file")?;

    data_to_commit.done().await;

    Ok(())
}

#[allow(clippy::type_complexity)]
fn merge_iter<UncommittedIter>(
    posting_id_generator: Arc<AtomicU64>,
    uncommitted_iter: UncommittedIter,
    committed_path: PathBuf,
    path_to_commit: PathBuf,
) -> Result<DashMap<u64, Vec<(DocumentId, Vec<usize>)>>>
where
    UncommittedIter: Iterator<
        Item = (
            Vec<u8>,
            (
                TotalDocumentsWithTermInField,
                HashMap<DocumentId, Positions>,
            ),
        ),
    >,
{
    let delta_committed_storage: DashMap<u64, Vec<(DocumentId, Vec<usize>)>> = Default::default();

    let committed_file =
        std::fs::File::open(committed_path).context("Cannot open file after writing to it")?;
    let committed_mmap = unsafe { Mmap::map(&committed_file)? };
    let committed_map = Map::new(committed_mmap).context("Cannot create fst map from mmap")?;
    let stream = FTSIter {
        stream: Some(committed_map.stream()),
    };

    let merge_iterator = MergedIterator::new(
        uncommitted_iter,
        stream,
        |_, (_, positions_per_document_id)| {
            let new_posting_list_id =
                posting_id_generator.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            delta_committed_storage.insert(
                new_posting_list_id,
                positions_per_document_id
                    .into_iter()
                    .map(|(doc_id, positions)| (doc_id, positions.0))
                    .collect(),
            );

            new_posting_list_id
        },
        |_, (_, positions_per_document_id), committed_posting_id| {
            let mut committed_positions_per_doc = delta_committed_storage
                .entry(committed_posting_id)
                .or_default();

            committed_positions_per_doc.extend(
                positions_per_document_id
                    .into_iter()
                    .map(|(doc_id, positions)| (doc_id, positions.0)),
            );

            committed_posting_id
        },
    );

    let mut f = BufferedFile::create(path_to_commit).context("Cannot create file")?;
    let mut build = MapBuilder::new(&mut f)?;

    for (key, value) in merge_iterator {
        build
            .insert(key, value)
            .context("Cannot insert value to FST map")?;
    }

    build.finish().context("Cannot finish build of FST map")?;

    f.close().context("Cannot close buffered file")?;

    Ok(delta_committed_storage)
}
