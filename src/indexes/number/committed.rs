use core::{f32, panic};
use std::{collections::HashSet, fmt::Debug, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::{collection_manager::sides::Offset, file_utils::BufferedFile, types::DocumentId};

use super::{n::SerializableNumber, Number, NumberFilter};

const MAX_NUMBER_PER_PAGE: usize = 1_000_000;

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    key: SerializableNumber,
    // Vec is not the best data structure here.
    // Should we use a smallvec?
    // TODO: think about this.
    values: HashSet<DocumentId>,
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
struct ChunkId(usize);

#[derive(Debug)]
struct Page {
    id: ChunkId,
    pointer: PagePointer,
    min: Number,
    max: Number,
}

fn load_items(p: &PathBuf) -> Result<Vec<Item>> {
    BufferedFile::open(p)
        .context("Cannot open number index page file")?
        .read_bincode_data()
        .context("Cannot deserialize number index page")
}

impl Page {
    fn filter(&self, filter: &NumberFilter, matching_docs: &mut HashSet<DocumentId>) -> Result<()> {
        // self.usage.increment(epoch);

        match &self.pointer {
            PagePointer::InMemory(items) => {
                Self::filter_on_items(items, filter, matching_docs);
            }
            PagePointer::OnFile(p) => {
                let items = load_items(p)?;
                Self::filter_on_items(&items, filter, matching_docs);
            }
        };

        Ok(())
    }

    fn filter_on_items(
        items: &[Item],
        filter: &NumberFilter,
        matching_docs: &mut HashSet<DocumentId>,
    ) {
        let filter_fn = get_filter_fn(filter);

        matching_docs.extend(
            items
                .iter()
                .filter(|item| filter_fn(item))
                .flat_map(|item| item.values.iter()),
        );
    }

    fn commit(&self, page_file: PathBuf) -> Result<()> {
        let items = match &self.pointer {
            PagePointer::InMemory(items) => items,
            PagePointer::OnFile(p) => {
                if p != &page_file {
                    std::fs::copy(p, &page_file)?;
                }
                return Ok(());
            }
        };

        BufferedFile::create(page_file)
            .context("Cannot create number page file")?
            .write_bincode_data(&items)
            .context("Cannot serialize number page")?;

        Ok(())
    }

    fn move_to_fs(&mut self, page_file: PathBuf) -> Result<()> {
        self.commit(page_file.clone())?;
        self.pointer = PagePointer::OnFile(page_file);
        Ok(())
    }
}

#[derive(Debug)]
pub struct CommittedNumberFieldIndex {
    offset: Offset,
    /// List of chunks.
    pages: Vec<Page>,
    /// Map of the bounds of each chunk.
    /// Lower bound is inclusive, upper bound is exclusive.
    bounds: Vec<((SerializableNumber, SerializableNumber), ChunkId)>,
}

impl CommittedNumberFieldIndex {
    // pub fn new(data_dir: PathBuf) -> Self {
    //     todo!()
    // }

    pub fn filter(&self, filter: &NumberFilter) -> Result<HashSet<DocumentId>> {
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
            page.filter(filter, &mut matching_docs)?;
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
            .binary_search_by_key(value, |(bounds, _)| bounds.0 .0);

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

    pub fn load(offset: Offset, data_dir: PathBuf) -> Result<Self> {
        let bounds: Vec<((SerializableNumber, SerializableNumber), ChunkId)> =
            BufferedFile::open(data_dir.join("bounds.json"))
                .context("Cannot open bounds file")?
                .read_json_data()
                .context("Cannot deserialize bounds file")?;

        let mut pages = Vec::with_capacity(bounds.len());
        for (bounds, id) in &bounds {
            let page_file = data_dir.join(format!("page_{}.bin", id.0));
            let (min, max) = bounds;

            pages.push(Page {
                id: *id,
                pointer: PagePointer::OnFile(page_file),
                min: min.0,
                max: max.0,
            });
        }

        Ok(CommittedNumberFieldIndex {
            offset,
            bounds,
            pages,
        })
    }

    pub fn commit(&self, data_dir: PathBuf) -> Result<()> {
        let bounds_file = data_dir.join("bounds.json");
        BufferedFile::create(bounds_file.clone())
            .context("Cannot create bounds file")?
            .write_json_data(&self.bounds)
            .context("Cannot serialize bounds")?;

        Ok(())
    }
}

impl CommittedNumberFieldIndex {
    pub fn from_iter<T: IntoIterator<Item = (Number, HashSet<DocumentId>)>>(
        offset: Offset,
        iter: T,
        base_dir: PathBuf,
    ) -> Result<Self> {
        std::fs::create_dir_all(&base_dir)
            .context("Cannot create the base directory for the committed number index")?;

        let mut committed_number_field_index = CommittedNumberFieldIndex {
            offset,
            pages: Vec::new(),
            bounds: Vec::new(),
        };

        let mut page_id = 0;
        let mut current_page = Page {
            id: ChunkId(page_id),
            pointer: PagePointer::InMemory(Vec::new()),
            min: Number::F32(f32::NEG_INFINITY),
            max: Number::F32(f32::INFINITY),
        };

        let mut prev = Number::F32(f32::NEG_INFINITY);
        let mut current_page_count = 0;
        for (value, doc_ids) in iter {
            assert!(value > prev);
            prev = value;
            current_page_count += doc_ids.len();

            if current_page_count > MAX_NUMBER_PER_PAGE {
                current_page.max = value;
                committed_number_field_index.bounds.push((
                    (
                        SerializableNumber(current_page.min),
                        SerializableNumber(current_page.max),
                    ),
                    current_page.id,
                ));
                current_page
                    .move_to_fs(base_dir.join(format!("page_{}.bin", page_id)))
                    .context("Cannot move the page to fs")?;
                committed_number_field_index.pages.push(current_page);

                page_id += 1;
                current_page_count = 0;
                current_page = Page {
                    id: ChunkId(page_id),
                    pointer: PagePointer::InMemory(Vec::new()),
                    min: value,
                    max: Number::F32(f32::INFINITY),
                };
            }

            current_page.max = value;
            match current_page.pointer {
                PagePointer::InMemory(ref mut items) => {
                    items.push(Item {
                        key: SerializableNumber(value),
                        values: doc_ids,
                    });
                }
                PagePointer::OnFile(_) => {
                    panic!("This should not happen");
                }
            };
        }

        current_page.max = Number::F32(f32::INFINITY);

        committed_number_field_index.bounds.push((
            (
                SerializableNumber(current_page.min),
                SerializableNumber(current_page.max),
            ),
            current_page.id,
        ));

        current_page
            .move_to_fs(base_dir.join(format!("page_{}.bin", page_id)))
            .context("Cannot move the page to fs")?;
        committed_number_field_index.pages.push(current_page);

        Ok(committed_number_field_index)
    }
}

impl CommittedNumberFieldIndex {
    pub fn iter(&self) -> impl Iterator<Item = (Number, HashSet<DocumentId>)> + '_ {
        self.pages.iter().flat_map(|page| {
            // We `collect` the items. This is not the best approach.
            // We should avoid it
            // TODO: think about this.

            match &page.pointer {
                PagePointer::InMemory(items) => items
                    .iter()
                    .map(|item| (item.key.0, item.values.clone()))
                    .collect::<Vec<_>>(),
                PagePointer::OnFile(p) => {
                    // `load_items` can fail. We should propagate the error.
                    // TODO: think about this.
                    let items = load_items(p).expect("Cannot load items");
                    items
                        .into_iter()
                        .map(|item| (item.key.0, item.values))
                        .collect::<Vec<_>>()
                }
            }
        })
    }

    pub fn current_offset(&self) -> Offset {
        self.offset
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

#[cfg(test)]
mod tests {
    use crate::test_utils::generate_new_path;

    use super::*;

    #[test]
    fn test_indexes_number_from_iter() -> Result<()> {
        let data: Vec<(_, HashSet<_>)> = vec![
            (Number::I32(0), HashSet::from_iter([DocumentId(0)])),
            (Number::I32(1), HashSet::from_iter([DocumentId(1)])),
            (Number::I32(2), HashSet::from_iter([DocumentId(2)])),
            (Number::I32(3), HashSet::from_iter([DocumentId(3)])),
            (Number::I32(4), HashSet::from_iter([DocumentId(4)])),
            (Number::I32(5), HashSet::from_iter([DocumentId(5)])),
            (Number::I32(6), HashSet::from_iter([DocumentId(6)])),
            (Number::I32(7), HashSet::from_iter([DocumentId(7)])),
            (Number::I32(8), HashSet::from_iter([DocumentId(8)])),
            (Number::I32(9), HashSet::from_iter([DocumentId(9)])),
        ];

        let committed_number_field_index =
            CommittedNumberFieldIndex::from_iter(Offset(9), data, generate_new_path())?;

        let output = committed_number_field_index
            .filter(&crate::indexes::number::NumberFilter::Equal(Number::I32(0)))?;
        assert_eq!(output, HashSet::from_iter([DocumentId(0)]));

        Ok(())
    }
}
