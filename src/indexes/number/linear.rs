use core::f32;
use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::Debug,
    fs,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::{collection_manager::dto::FieldId, document_storage::DocumentId};

use super::{serializable_number::SerializableNumber, stats::PageUsage, Number, NumberFilter};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChunkId(usize);

struct Page {
    id: ChunkId,
    pointer: PagePointer,
    min: Number,
    max: Number,
    usage: PageUsage,
}

impl Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.id)
            .field("pointer", &self.pointer)
            .field("min", &self.min)
            .field("max", &self.max)
            .field("usage", &"...")
            .finish()
    }
}

fn get_filter_fn<'filter>(filter: &'filter NumberFilter) -> Box<dyn Fn(&Item) -> bool + 'filter> {
    match filter {
        NumberFilter::Equal(value) => Box::new(move |x| &x.key.0 == value),
        NumberFilter::Between((min, max)) => Box::new(move |x| &x.key.0 >= min && &x.key.0 <= max),
        NumberFilter::GreaterThan(min) => Box::new(move |x| &x.key.0 > min),
        NumberFilter::GreaterThanOrEqual(min) => Box::new(move |x| &x.key.0 >= min),
        NumberFilter::LessThan(max) => Box::new(move |x| &x.key.0 < max),
        NumberFilter::LessThanOrEqual(max) => Box::new(move |x| &x.key.0 <= max),
    }
}

impl Page {
    fn filter(
        &self,
        m_field_id: FieldId,
        filter: &NumberFilter,
        matching_docs: &mut HashSet<DocumentId>,
        epoch: u64,
    ) -> Result<()> {
        self.usage.increment(epoch);

        match &self.pointer {
            PagePointer::InMemory(items) => {
                Self::filter_on_items(items, m_field_id, filter, matching_docs);
            }
            PagePointer::OnFile(p) => {
                let f = std::fs::File::open(p).with_context(|| format!("Cannot open {p:?}"))?;
                let buf = BufReader::new(f);
                let items: Vec<Item> = bincode::deserialize_from(buf)
                    .with_context(|| format!("Cannot deserialize items from {p:?}"))?;

                Self::filter_on_items(&items, m_field_id, filter, matching_docs);
            }
        };

        Ok(())
    }

    fn filter_on_items(
        items: &[Item],
        m_field_id: FieldId,
        filter: &NumberFilter,
        matching_docs: &mut HashSet<DocumentId>,
    ) {
        let filter_fn = get_filter_fn(filter);

        matching_docs.extend(
            items
                .iter()
                .filter(|item| filter_fn(item))
                .flat_map(|item| {
                    item.values
                        .iter()
                        .filter(|(_, field_id)| *field_id == m_field_id)
                        .map(|(doc_id, _)| *doc_id)
                }),
        );
    }

    fn move_to_fs(&mut self, page_file: PathBuf) -> Result<()> {
        let items = match &self.pointer {
            PagePointer::InMemory(items) => items,
            PagePointer::OnFile(_) => return Ok(()),
        };

        let mut file = std::fs::File::create(page_file.clone())?;
        let mut buf_writer = BufWriter::new(&mut file);
        bincode::serialize_into(&mut buf_writer, items).unwrap();
        buf_writer.flush().unwrap();
        drop(buf_writer);
        file.sync_data().unwrap();

        self.pointer = PagePointer::OnFile(page_file);

        Ok(())
    }
}

fn get_page_file(page: &Page, base_path: &Path) -> PathBuf {
    base_path.join(format!("page_{}.bin", page.id.0))
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

pub struct FromIterConfig {
    pub max_size_per_chunk: usize,
    pub base_path: PathBuf,
}

impl LinearNumberIndex {
    fn new(max_size_per_chunk: usize) -> Self {
        Self {
            pages: Vec::new(),
            bounds: Vec::new(),
            max_size_per_chunk,
        }
    }

    #[allow(dead_code)]
    pub fn bounds(&self) -> &[((Number, Number), ChunkId)] {
        &self.bounds
    }

    pub fn from_fs(base_path: PathBuf, default_max_size_per_chunk: usize) -> Result<Self> {
        let bound_file = base_path.join("bounds.bin");

        let exists = fs::exists(&bound_file).unwrap_or(false);
        if !exists {
            return Self::from_iter(
                std::iter::empty(),
                FromIterConfig {
                    max_size_per_chunk: default_max_size_per_chunk,
                    base_path,
                },
            );
        }

        let (max_size_per_chunk, bounds) = {
            let f = std::fs::File::open(bound_file)
                .with_context(|| anyhow::anyhow!("Cannot open file"))?;
            let mut buf = BufReader::new(f);
            let max_size_per_chunk: usize = bincode::deserialize_from(&mut buf)
                .with_context(|| anyhow::anyhow!("Cannot deserialize `max_size_per_chunk`"))?;
            let bounds: Vec<((Number, Number), ChunkId)> = bincode::deserialize_from(&mut buf)
                .with_context(|| anyhow::anyhow!("Cannot deserialize `bounds`"))?;

            (max_size_per_chunk, bounds)
        };

        let mut pages = Vec::with_capacity(bounds.len());

        for ((min, max), id) in &bounds {
            let page_file = base_path.join(format!("page_{}.bin", id.0));
            let pointer = PagePointer::OnFile(page_file);
            let page = Page {
                id: *id,
                pointer,
                min: *min,
                max: *max,
                usage: PageUsage::new(),
            };

            pages.push(page);
        }

        Ok(Self {
            pages,
            bounds,
            max_size_per_chunk,
        })
    }

    /// `data`` should be already sorter
    pub fn from_iter<I>(iter: I, config: FromIterConfig) -> Result<Self>
    where
        I: IntoIterator<Item = (Number, (DocumentId, FieldId))>,
    {
        let FromIterConfig {
            max_size_per_chunk,
            base_path,
        } = config;

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
                index
                    .bounds
                    .push(((current_chunk.min, current_chunk.max), current_chunk.id));

                let page_file = get_page_file(&current_chunk, &base_path);
                current_chunk
                    .move_to_fs(page_file)
                    .with_context(|| anyhow::anyhow!("Cannot move page to FS"))?;
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
                PagePointer::InMemory(ref mut items) => items.push(Item {
                    key: SerializableNumber(key),
                    values: doc_id_field_id_pairs.into_iter().collect(),
                }),
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
        index
            .bounds
            .push(((current_chunk.min, current_chunk.max), current_chunk.id));

        let page_file = get_page_file(&current_chunk, &base_path);
        current_chunk
            .move_to_fs(page_file)
            .with_context(|| anyhow::anyhow!("Cannot move page to FS"))?;

        index.pages.push(current_chunk);

        let bound_file = base_path.join("bounds.bin");
        let mut file = std::fs::File::create(bound_file.clone())
            .with_context(|| anyhow::anyhow!("Cannot create bounds.bin file"))?;
        let mut buf_writer = BufWriter::new(&mut file);
        bincode::serialize_into(&mut buf_writer, &index.max_size_per_chunk)
            .context("Cannot serialize `max_size_per_chunk`")?;
        bincode::serialize_into(&mut buf_writer, &index.bounds)
            .context("Cannot serialize `bounds`")?;
        buf_writer.flush().unwrap();
        drop(buf_writer);
        file.sync_data().unwrap();

        // Checks we cover all the `Number` space
        debug_assert_eq!(index.pages.len(), index.bounds.len());
        debug_assert_eq!(index.pages[0].min, Number::from(f32::NEG_INFINITY));
        debug_assert_eq!(index.pages.last().unwrap().max, Number::from(f32::INFINITY));

        Ok(index)
    }

    pub fn filter(
        &self,
        field_id: FieldId,
        filter: &NumberFilter,
        epoch: u64,
    ) -> Result<HashSet<DocumentId>> {
        let pages = match filter {
            NumberFilter::Equal(value) => {
                let page = match self.find_page(value) {
                    Ok(page) => page,
                    Err(_) => return Ok(HashSet::new()),
                };
                vec![page]
            }
            NumberFilter::Between((min, max)) => {
                let min_page = match self.find_page(min) {
                    Ok(page) => page,
                    Err(_) => return Ok(HashSet::new()),
                };
                let max_page = match self.find_page(max) {
                    Ok(page) => page,
                    Err(_) => return Ok(HashSet::new()),
                };

                let min_page_id = min_page.id;
                let max_page_id = max_page.id;

                // let min_pos = self.pages.iter().position(|p| p.id == min_page_id).unwrap();
                // let max_pos = self.pages.iter().position(|p| p.id == max_page_id).unwrap();

                self.pages
                    .iter()
                    .skip_while(|p| p.id < min_page_id)
                    .take_while(|p| p.id <= max_page_id)
                    .collect()
            }
            NumberFilter::GreaterThan(min) | NumberFilter::GreaterThanOrEqual(min) => {
                let min_page = match self.find_page(min) {
                    Ok(page) => page,
                    Err(_) => return Ok(HashSet::new()),
                };

                let min_page_id = min_page.id;

                self.pages
                    .iter()
                    .skip_while(|p| p.id < min_page_id)
                    .collect()
            }
            NumberFilter::LessThan(max) | NumberFilter::LessThanOrEqual(max) => {
                let max_page = match self.find_page(max) {
                    Ok(page) => page,
                    Err(_) => return Ok(HashSet::new()),
                };

                let max_page_id = max_page.id;

                self.pages
                    .iter()
                    .take_while(|p| p.id <= max_page_id)
                    .collect()
            }
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
            // This should never fail.
            // We could put an empty page, so we can avoid this check.
            // TODO: do it.
            return Err(anyhow::anyhow!("No pages in the index"));
        }

        let pos = self
            .bounds
            .binary_search_by_key(value, |(bounds, _)| bounds.0);

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

    pub fn iter(&self) -> LinearNumberIndexIter<'_> {
        LinearNumberIndexIter {
            index: self,
            current_page: 0,
            current_item: 0,
        }
    }
}

pub struct LinearNumberIndexIter<'s> {
    index: &'s LinearNumberIndex,
    current_page: usize,
    current_item: usize,
}

impl<'a> Iterator for LinearNumberIndexIter<'a> {
    type Item = Result<(Number, Cow<'a, Vec<(DocumentId, FieldId)>>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_page >= self.index.pages.len() {
            return None;
        }

        let page = &self.index.pages[self.current_page];
        let item = &page.pointer;

        match item {
            PagePointer::InMemory(items) => {
                if self.current_item >= items.len() {
                    self.current_page += 1;
                    self.current_item = 0;
                    return self.next();
                }

                let item = &items[self.current_item];
                let values = &item.values;

                self.current_item += 1;

                Some(Ok((item.key.0, Cow::Borrowed(values))))
            }
            PagePointer::OnFile(p) => {
                let f = match std::fs::File::open(p) {
                    Ok(f) => f,
                    Err(e) => return Some(Err(e.into())),
                };
                let buf = BufReader::new(f);
                let mut items: Vec<Item> = match bincode::deserialize_from(buf) {
                    Ok(items) => items,
                    Err(e) => return Some(Err(e.into())),
                };

                if self.current_item >= items.len() {
                    self.current_page += 1;
                    self.current_item = 0;
                    return self.next();
                }

                let item = items.remove(self.current_item);

                self.current_item += 1;

                Some(Ok((item.key.0, Cow::Owned(item.values))))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_indexes_number_linear_equal() -> Result<()> {
        let base_path1 = generate_new_path();
        fs::create_dir_all(&base_path1)?;
        let iter = (0..2).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        let index_1 = LinearNumberIndex::from_iter(
            iter,
            FromIterConfig {
                max_size_per_chunk: 2048,
                base_path: base_path1,
            },
        )?;

        let base_path2 = generate_new_path();
        fs::create_dir_all(&base_path2)?;
        let iter = (0..1_000).map(|i| (Number::from(i), (DocumentId(i as u32), FieldId(0))));
        let index_2 = LinearNumberIndex::from_iter(
            iter,
            FromIterConfig {
                max_size_per_chunk: 2048,
                base_path: base_path2,
            },
        )?;

        let tests = [
            (
                NumberFilter::Equal(Number::from(1)),
                HashSet::from_iter(vec![DocumentId(1)]),
            ),
            (
                NumberFilter::Equal(Number::from(1.0)),
                HashSet::from_iter(vec![DocumentId(1)]),
            ),
            (
                NumberFilter::Equal(Number::from(0)),
                HashSet::from_iter(vec![DocumentId(0)]),
            ),
            (
                NumberFilter::Equal(Number::from(0.0)),
                HashSet::from_iter(vec![DocumentId(0)]),
            ),
            (
                NumberFilter::Equal(Number::from(-1)),
                HashSet::from_iter(vec![]),
            ),
            (
                NumberFilter::Equal(Number::from(10_000_000)),
                HashSet::from_iter(vec![]),
            ),
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
