use core::f32;
use std::{collections::HashSet, io::{BufReader, BufWriter, Write}, path::PathBuf, ptr::{self, null}, sync::atomic::{AtomicPtr, AtomicUsize}};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

use super::{serializable_number::SerializableNumber, Number, NumberFilter};

/// struct to track the usage of a page.
#[derive(Debug)]
struct PageUsage<const N: usize> {
    buff: [AtomicPtr<(u64, usize)>; N],
    head_index: AtomicUsize,
}
impl<const N: usize> PageUsage<N> {
    fn new() -> Self {
        Self {
            // Fill the array with 0.
            // This mean the epoch is 0, so 1Gen 1970.
            buff: core::array::from_fn(|i| AtomicPtr::new(ptr::null_mut())),

            head_index: AtomicUsize::new(0),
        }
    }

    fn increment(&self, epoch: u64) {
        // We count number of access each 64 epoch (seconds).
        let idx = epoch >> 6; 

        let head = self.head_index.load(std::sync::atomic::Ordering::Relaxed);
        let head = head % N;
        let mut bucket = self.buff[head].load(std::sync::atomic::Ordering::Relaxed);

        if bucket.is_null() {
            loop {
                match self.buff[head].compare_exchange_weak(bucket, &mut (
                    idx,
                    1,
                ), std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(x) => bucket = x,
                }
            }
        } else {
            let (mut bucket_epoch, mut bucket_count) = unsafe { &mut *bucket };
            if bucket_epoch == idx {
                bucket_count += 1;
            } else {
                loop {
                    match self.buff[head].compare_exchange_weak(bucket, &mut (
                        idx,
                        1,
                    ), std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(x) => bucket = x,
                    }
                }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    key: SerializableNumber,
    // Vec is not the best data structure here.
    // Should we use a smallvec?
    // TODO: think about this.
    values: Vec<(DocumentId, FieldId)>,
}

impl PartialEq for Item {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Item {}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// A chunk can be loaded in memory or not.
/// If loaded, it contains a list of items.
/// If not loaded, it contains a pointer to the file on disk (TODO).
#[derive(Debug)]
enum PagePointer {
    InMemory(Vec<Item>),
    OnFile(PathBuf),
}

/// This is the index of the chunk used in `chunks` array.
#[derive(Debug, Clone, Copy)]
struct ChunkId(usize);

#[derive(Debug)]
struct Page {
    id: ChunkId,
    pointer: PagePointer,
    min: Number,
    max: Number,
    usage: PageUsage<10>,
}

fn get_filter_fn<'filter>(filter: &'filter NumberFilter) -> Box<dyn Fn(&Item) -> bool + 'filter> {
    match filter {
        NumberFilter::Equal(value) => Box::new(move |x| &x.key.0 == value),
        _ => unimplemented!(),
    }
}

impl Page {
    fn filter(&self, m_field_id: FieldId, filter: &NumberFilter, matching_docs: &mut HashSet<DocumentId>, epoch: u64) -> Result<()> {
        self.usage.increment(epoch);
        match &self.pointer {
            PagePointer::InMemory(items) => {
                Self::filter_on_items(items, m_field_id, filter, matching_docs);
            }
            PagePointer::OnFile(p) => {
                let f = std::fs::File::open(p)?;
                let buf = BufReader::new(f);
                let items: Vec<Item> = bincode::deserialize_from(buf)?;

                Self::filter_on_items(&items, m_field_id, filter, matching_docs);
            },
        };

        Ok(())
    }

    fn filter_on_items(items: &[Item], m_field_id: FieldId, filter: &NumberFilter, matching_docs: &mut HashSet<DocumentId>) {
        let filter_fn = get_filter_fn(filter);

        matching_docs.extend(
            items.iter()
                .filter(|item| filter_fn(item))
                .flat_map(|item| item.values.iter()
                    .filter(|(_, field_id)| *field_id == m_field_id)
                    .map(|(doc_id, _)| *doc_id)
                )
        );
    }
}

#[derive(Debug)]
pub struct LinearNumberIndex {
    /// List of chunks.
    pages: Vec<Page>,
    /// Map of the bounds of each chunk.
    /// Lower bound is inclusive, upper bound is exclusive.
    bounds: Vec<((Number, Number), ChunkId)>,

    max_size_per_chunk: usize,
}

impl LinearNumberIndex {
    fn new(max_size_per_chunk: usize) -> Self {
        Self {
            pages: Vec::new(),
            bounds: Vec::new(),
            max_size_per_chunk,
        }
    }

    /// `data`` should be already sorter
    pub fn from_iter<I>(iter: I, max_size_per_chunk: usize) -> Self
    where
        I: IntoIterator<Item = (Number, (DocumentId, FieldId))>,
    {
        let mut index = Self::new(max_size_per_chunk);

        let mut current_chunk_size = 0_usize;
        let mut current_chunk = Page {
            id: ChunkId(0),
            pointer: PagePointer::InMemory(Vec::new()),
            min: Number::from(f32::NEG_INFINITY),
            max: Number::from(f32::INFINITY),
            usage: PageUsage::new(),
        };

        let mut iter = iter.into_iter();

        let mut current_row = iter.next();

        while let Some((key, doc_id_field_id_pair)) = current_row {
            let mut doc_id_field_id_pairs: HashSet<(DocumentId, FieldId)> = Default::default();
            doc_id_field_id_pairs.insert(doc_id_field_id_pair);

            let mut first_key_different = None;
            for (inner_key, doc_id_field_id_pair) in iter.by_ref() {
                if inner_key != key {
                    first_key_different = Some((inner_key, doc_id_field_id_pair));
                    break;
                }
                doc_id_field_id_pairs.insert(doc_id_field_id_pair);
            }

            // This is not exaclty the size of the page.
            // We forgot some headers as keys and other stuff
            // but it's good enough for now: we don't need to be precise here.
            // the last `* 2` is a rough estimation of bincode serialization.
            let page_size = doc_id_field_id_pairs.len() * size_of::<(DocumentId, FieldId)>() * 2;

            // This is a check. This could fail if many documents have the same key.
            // In this case, the distribution is not random, but it concentrates on a few values.
            // We should handle this case better.
            // Currently the limit is 10k elements inside an Item.
            // See the test `test_indexes_number_linear_how_many_docs_live_in_64k` in this file.
            // TODO: handle the case where the page is too big.
            assert!(page_size <= max_size_per_chunk);

            if (current_chunk_size + page_size) >= max_size_per_chunk {
                current_chunk.max = key;
                index.bounds.push(((current_chunk.min, current_chunk.max), current_chunk.id));
                index.pages.push(current_chunk);

                current_chunk = Page {
                    id: ChunkId(index.pages.len()),
                    pointer: PagePointer::InMemory(vec![]),
                    min: key,
                    max: Number::from(f32::INFINITY),
                    usage: PageUsage::new(),
                };
                
                current_chunk_size = 0;
            }

            current_chunk_size += page_size;
            match current_chunk.pointer {
                PagePointer::InMemory(ref mut items) => {
                    items.push(Item {
                        key: SerializableNumber(key),
                        values: doc_id_field_id_pairs.into_iter().collect(),
                    })
                }
                PagePointer::OnFile(_) => unreachable!("Chunk should be loaded"),
            };

            current_row = first_key_different;
        }

        // Fix the max of the last chunk
        current_chunk.max = Number::from(f32::INFINITY);

        // Fix the min of the first chunk
        let first = if let Some(first) = index.pages.first_mut() {
            first
        } else {
            &mut current_chunk
        };
        first.min = Number::from(f32::NEG_INFINITY);

        // Add the last chunk
        index.bounds.push(((current_chunk.min, current_chunk.max), current_chunk.id));
        index.pages.push(current_chunk);

        // Checks we cover all the `Number` space
        assert_eq!(index.pages.len(), index.bounds.len());
        assert_eq!(index.pages[0].min, Number::from(f32::NEG_INFINITY));
        assert_eq!(index.pages.last().unwrap().max, Number::from(f32::INFINITY));

        index
    }

    pub fn filter(&self, field_id: FieldId, filter: &NumberFilter, epoch: u64) -> Result<HashSet<DocumentId>> {
        let pages = match filter {
            NumberFilter::Equal(value) => {
                let page = match self.find_page(value) {
                    Ok(page) => page,
                    Err(_) => return Ok(HashSet::new()),
                };
                vec![page]
            },
            _ => unimplemented!()
        };

        let mut matching_docs = HashSet::new();

        for page in pages {
            // If the `filter` fails, should we ignore it?
            // TODO: think better about this. 
            page.filter(field_id, filter, &mut matching_docs, epoch)?;
        }

        Ok(matching_docs)
    }

    fn find_page(&self, value: &Number) -> Result<&Page> {
        if self.pages.is_empty() {
            return Err(anyhow::anyhow!("No pages in the index"));
        }

        let pos = self.bounds.binary_search_by_key(value, |(bounds, _)| bounds.0);

        let page = pos.map(|pos| &self.pages[pos])
            // If the value i'm looking for is contained in a boud, the `binary_search_by_key` returns a error.
            // That error is the index where the value should be inserted to keep the array sorted.
            // Because our pages are:
            // - sorted
            // - contiguous
            // the page I'm looking for is the one before that index.
            .unwrap_or_else(|i| {
                if i == 0 {
                    error!(r#"binary_search on LinearNumberIndex identify a number less then NEG_INFINITY (the first lower bound).
And this should not happen. Return the first page."#);
                    return self.pages.first().expect("The check on the index is empty should be done before this line");
                }
                if i > self.pages.len() {
                    error!(r#"binary_search on LinearNumberIndex identify a number greater then INFINITY (the last upper bound).
And this should not happen. Return the last page."#);
                    return self.pages.last().expect("The check on the index is empty should be done before this line");
                }
                &self.pages[i - 1]
            });

        Ok(page)
    }

    pub fn save_on_fs_and_unload<P>(&mut self, p: P) -> Result<()>
    where 
        P: Into<PathBuf>
    {
        let p = p.into();

        for page in &mut self.pages {
            if let PagePointer::InMemory(items) = &mut page.pointer {

                let page_file = p
                    .join(format!("page_{}.bin", page.id.0));

                println!("Save page {} on file {:?}", page.id.0, page_file);

                let mut file = std::fs::File::create(page_file.clone())?;
                let mut buf_writer = BufWriter::new(&mut file);
                bincode::serialize_into(&mut buf_writer, items)?;
                buf_writer.flush()?;
                drop(buf_writer);
                file.sync_data()?;

                page.pointer = PagePointer::OnFile(page_file);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_indexes_number_linear_equal() -> Result<()> {
        let iter = (0..2).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        let index_1 = LinearNumberIndex::from_iter(iter, 2048);

        let iter = (0..1_000).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        let index_2 = LinearNumberIndex::from_iter(iter, 2048);

        let tests = [
            (NumberFilter::Equal(Number::from(1)), HashSet::from_iter(vec![DocumentId(1)])),
            (NumberFilter::Equal(Number::from(1.0)), HashSet::from_iter(vec![DocumentId(1)])),
            (NumberFilter::Equal(Number::from(0)), HashSet::from_iter(vec![DocumentId(0)])),
            (NumberFilter::Equal(Number::from(0.0)), HashSet::from_iter(vec![DocumentId(0)])),
            (NumberFilter::Equal(Number::from(-1)), HashSet::from_iter(vec![])),
            (NumberFilter::Equal(Number::from(10_000_000)), HashSet::from_iter(vec![])),
        ];

        for (filter, expected) in tests {
            let matching_docs = index_1.filter(FieldId(0), &filter, 0)?;
            assert_eq!(matching_docs, expected);

            let matching_docs = index_2.filter(FieldId(0), &filter, 0)?;
            assert_eq!(matching_docs, expected);
        }

        Ok(())
    }


    #[test]
    fn test_indexes_number_linear_dump_on_fs() -> Result<()> {
        let iter = (0..2).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        let mut index_1 = LinearNumberIndex::from_iter(iter, 2048);

        let iter = (0..1_000).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        let mut index_2 = LinearNumberIndex::from_iter(iter, 2048);

        let tmp_dir = TempDir::new("example")?;
        let dump_path = tmp_dir.path().join("index_1");
        std::fs::remove_dir_all(dump_path.clone()).ok();
        std::fs::create_dir_all(dump_path.clone())?;
        index_1.save_on_fs_and_unload(dump_path.clone())?;

        let tmp_dir = TempDir::new("example")?;
        let dump_path = tmp_dir.path().join("index_2");
        std::fs::remove_dir_all(dump_path.clone()).ok();
        std::fs::create_dir_all(dump_path.clone())?;
        index_2.save_on_fs_and_unload(dump_path.clone())?;

        let tests = [
            (NumberFilter::Equal(Number::from(1)), HashSet::from_iter(vec![DocumentId(1)])),
            (NumberFilter::Equal(Number::from(1.0)), HashSet::from_iter(vec![DocumentId(1)])),
            (NumberFilter::Equal(Number::from(0)), HashSet::from_iter(vec![DocumentId(0)])),
            (NumberFilter::Equal(Number::from(0.0)), HashSet::from_iter(vec![DocumentId(0)])),
            (NumberFilter::Equal(Number::from(-1)), HashSet::from_iter(vec![])),
            (NumberFilter::Equal(Number::from(10_000_000)), HashSet::from_iter(vec![])),
        ];

        for (filter, expected) in tests {
            let matching_docs = index_1.filter(FieldId(0), &filter, 0)?;
            assert_eq!(matching_docs, expected);
        }

        Ok(())
    }

    #[test]
    fn test_indexes_number_linear_how_many_docs_live_in_64k() {
        const LIMIT_64K: u64 = 64 * 1024;

        // Min estimation of the limit
        let l = 10_500;
        let item = Item {
            key: SerializableNumber(Number::from(0)),
            values: (0..l).map(|i| (DocumentId(i as u32), FieldId(0))).collect(),
        };
        let size = bincode::serialized_size(&item).unwrap();
        assert!(size < LIMIT_64K);

        // Max estimation of the limit
        let l = 11_750;
        let item = Item {
            key: SerializableNumber(Number::from(0)),
            values: (0..l).map(|i| (DocumentId(i as u32), FieldId(0))).collect(),
        };
        let size = bincode::serialized_size(&item).unwrap();
        assert!(size > LIMIT_64K);

    }
}
