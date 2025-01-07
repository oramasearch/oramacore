use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{atomic::AtomicU64, Arc};

use std::iter::Peekable;

use anyhow::{Context, Result};
use dashmap::DashMap;
use fst::{Map, MapBuilder, Streamer};
use memmap::Mmap;

use crate::document_storage::DocumentId;

use super::document_lengths::DocumentLengthsPerDocument;
use super::posting_storage::PostingIdStorage;
use super::uncommitted::{Positions, TotalDocumentsWithTermInField};
use super::{CommittedStringFieldIndex, UncommittedStringFieldIndex};

pub struct MergedIterator<
    K,
    V1,
    V2,
    I1: Iterator<Item = (K, V1)>,
    I2: Iterator<Item = (K, V2)>,
    Transformer: FnMut(&K, V1) -> V2,
    Merger: FnMut(&K, V1, V2) -> V2,
> {
    iter1: Peekable<I1>,
    iter2: Peekable<I2>,
    transformer: Transformer,
    merger: Merger,
}

impl<
        K: Ord + Eq,
        V1,
        V2,
        I1: Iterator<Item = (K, V1)>,
        I2: Iterator<Item = (K, V2)>,
        Transformer: FnMut(&K, V1) -> V2,
        Merger: FnMut(&K, V1, V2) -> V2,
    > MergedIterator<K, V1, V2, I1, I2, Transformer, Merger>
{
    pub fn new(iter1: I1, iter2: I2, transformer: Transformer, merger: Merger) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            transformer,
            merger,
        }
    }
}

impl<
        K: Ord + Eq,
        V1,
        V2,
        I1: Iterator<Item = (K, V1)>,
        I2: Iterator<Item = (K, V2)>,
        Transformer: FnMut(&K, V1) -> V2,
        Merger: FnMut(&K, V1, V2) -> V2,
    > Iterator for MergedIterator<K, V1, V2, I1, I2, Transformer, Merger>
{
    type Item = (K, V2);

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.iter1.peek();
        let second = self.iter2.peek();

        match (first, second) {
            (Some((k1, _)), Some((k2, _))) => {
                let cmp = k1.cmp(k2);
                match cmp {
                    std::cmp::Ordering::Less => {
                        let v = self.iter1.next();
                        if let Some((k, v)) = v {
                            let v = (self.transformer)(&k, v);
                            Some((k, v))
                        } else {
                            None
                        }
                    }
                    std::cmp::Ordering::Greater => self.iter2.next(),
                    std::cmp::Ordering::Equal => {
                        let (k1, v1) = self.iter1.next().unwrap();
                        let (_, v2) = self.iter2.next().unwrap();
                        let v = (self.merger)(&k1, v1, v2);
                        Some((k1, v))
                    }
                }
            }
            (Some(_), None) => {
                let v = self.iter1.next();
                if let Some((k, v)) = v {
                    let v = (self.transformer)(&k, v);
                    Some((k, v))
                } else {
                    None
                }
            }
            (None, Some(_)) => self.iter2.next(),
            (None, None) => None,
        }
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

pub fn merge(
    posting_id_generator: Arc<AtomicU64>,
    uncommitted: &UncommittedStringFieldIndex,
    committed: &CommittedStringFieldIndex,
    document_length_new_path: PathBuf,
    fst_new_path: PathBuf,
    posting_new_path: PathBuf,
    global_info_new_path: PathBuf,
) -> Result<()> {
    let data_to_commit = uncommitted.take()?;

    committed
        .document_lengths_per_document
        .merge(
            data_to_commit.get_document_lengths(),
            document_length_new_path,
        )
        .context("Cannot merge document lengths")?;

    let uncommitted_iter = data_to_commit.iter();
    let storage_updates = merge_iter(
        posting_id_generator,
        uncommitted_iter,
        committed.fst_map_path.clone(),
        fst_new_path,
    )
    .context("Cannot merge iterators")?;

    committed
        .storage
        .apply_delta(storage_updates, posting_new_path)?;

    let mut global_info_file =
        File::create(global_info_new_path).context("Cannot create file for global info")?;
    let global_info = data_to_commit.global_info() + committed.get_global_info();
    serde_json::to_writer(&mut global_info_file, &global_info)
        .context("Cannot write global info to file")?;
    global_info_file.flush().context("Cannot flush file")?;
    global_info_file
        .sync_data()
        .context("Cannot sync data to disk")?;

    data_to_commit.done();

    Ok(())
}

pub fn create(
    posting_id_generator: Arc<AtomicU64>,
    uncommitted: &UncommittedStringFieldIndex,
    document_length_new_path: PathBuf,
    fst_new_path: PathBuf,
    posting_new_path: PathBuf,
    global_info_new_path: PathBuf,
) -> Result<()> {
    let data_to_commit = uncommitted.take().context("Cannot take from uncommitted")?;

    DocumentLengthsPerDocument::create(
        data_to_commit.get_document_lengths(),
        document_length_new_path,
    )
    .context("Cannot create file for document lengths")?;

    let uncommitted_iter = data_to_commit.iter();

    let mut delta_committed_storage: HashMap<u64, Vec<(DocumentId, Vec<usize>)>> =
        Default::default();

    let mut file = std::fs::File::create(fst_new_path.clone())
        .with_context(|| format!("Cannot create file at {:?}", fst_new_path))?;
    let mut wtr = std::io::BufWriter::new(&mut file);
    let mut build = MapBuilder::new(&mut wtr)?;

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

    wtr.flush().context("Cannot flush FST map")?;
    drop(wtr);
    file.sync_data().context("Cannot sync data to disk")?;
    file.flush().context("Cannot flush file")?;

    PostingIdStorage::create(delta_committed_storage, posting_new_path)
        .context("Cannot create posting id storage")?;

    let mut global_info_file =
        File::create(global_info_new_path).context("Cannot create file for global info")?;
    let global_info = data_to_commit.global_info();
    serde_json::to_writer(&mut global_info_file, &global_info)
        .context("Cannot write global info to file")?;
    global_info_file.flush().context("Cannot flush file")?;
    global_info_file
        .sync_data()
        .context("Cannot sync data to disk")?;

    data_to_commit.done();

    Ok(())
}

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

    let mut file = std::fs::File::create(path_to_commit.clone())
        .with_context(|| format!("Cannot create file at {:?}", path_to_commit))?;
    let mut wtr = std::io::BufWriter::new(&mut file);
    let mut build = MapBuilder::new(&mut wtr)?;

    for (key, value) in merge_iterator {
        build
            .insert(key, value)
            .context("Cannot insert value to FST map")?;
    }

    build.finish().context("Cannot finish build of FST map")?;

    wtr.flush().context("Cannot flush FST map")?;
    drop(wtr);
    file.sync_data().context("Cannot sync data to disk")?;
    file.flush().context("Cannot flush file")?;

    Ok(delta_committed_storage)
}
