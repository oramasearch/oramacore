use core::{f32, panic};
use std::{collections::HashSet, io::Write, path::PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::error;

use crate::{file_utils::BufferedFile, types::DocumentId};

use super::{Number, NumberFilter};

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

impl Page {
    fn filter(&self, filter: &NumberFilter, matching_docs: &mut HashSet<DocumentId>) -> Result<()> {
        // self.usage.increment(epoch);

        match &self.pointer {
            PagePointer::InMemory(items) => {
                Self::filter_on_items(items, filter, matching_docs);
            }
            PagePointer::OnFile(p) => {
                let items: Vec<Item> = BufferedFile::open(p)
                    .context("Cannot open number index page file")?
                    .read_bincode_data()
                    .context("Cannot deserialize number index page")?;

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

    fn move_to_fs(&mut self, page_file: PathBuf) -> Result<()> {
        let items = match &self.pointer {
            PagePointer::InMemory(items) => items,
            PagePointer::OnFile(p) => {
                if p != &page_file {
                    std::fs::copy(p, &page_file)?;
                }
                return Ok(());
            }
        };

        BufferedFile::create(page_file.clone())
            .context("Cannot create number page file")?
            .write_bincode_data(&items)
            .context("Cannot serialize number page")?;

        self.pointer = PagePointer::OnFile(page_file);

        Ok(())
    }
}

#[derive(Debug)]
pub struct CommittedNumberFieldIndex {
    /// List of chunks.
    pages: Vec<Page>,
    /// Map of the bounds of each chunk.
    /// Lower bound is inclusive, upper bound is exclusive.
    bounds: Vec<((Number, Number), ChunkId)>,
}

impl CommittedNumberFieldIndex {
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
}

impl CommittedNumberFieldIndex {
    pub fn from_iter<T: IntoIterator<Item = (Number, HashSet<DocumentId>)>>(
        iter: T,
        base_dir: PathBuf,
    ) -> Result<Self> {
        let mut committed_number_field_index = CommittedNumberFieldIndex {
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
        let mut current_page_count = 0;
        for (value, doc_ids) in iter {
            current_page_count += doc_ids.len();

            // 1000 is a magic number.
            if current_page_count > 1000 {
                current_page.max = value;
                committed_number_field_index
                    .bounds
                    .push(((current_page.min, current_page.max), current_page.id));
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

        committed_number_field_index
            .bounds
            .push(((current_page.min, current_page.max), current_page.id));

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
            let items = match &page.pointer {
                PagePointer::InMemory(items) => items,
                PagePointer::OnFile(_) => panic!("This should not happen"),
            };

            items.iter().map(|item| (item.key.0, item.values.clone()))
        })
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SerializableNumber(pub Number);

impl Serialize for SerializableNumber {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeTuple;

        match &self.0 {
            Number::F32(v) => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(&1_u8)?;
                tuple.serialize_element(v)?;
                tuple.end()
            }
            Number::I32(v) => {
                let mut tuple = serializer.serialize_tuple(2)?;
                tuple.serialize_element(&2_u8)?;
                tuple.serialize_element(v)?;
                tuple.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SerializableNumber {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{Error, Visitor};

        struct SerializableNumberVisitor;

        impl<'de> Visitor<'de> for SerializableNumberVisitor {
            type Value = SerializableNumber;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(
                    formatter,
                    "a tuple of size 2 consisting of a u64 discriminant and a value"
                )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let discriminant: u8 = seq
                    .next_element()?
                    .ok_or_else(|| A::Error::invalid_length(0, &self))?;
                match discriminant {
                    1_u8 => {
                        let x = seq
                            .next_element()?
                            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                        Ok(SerializableNumber(Number::F32(x)))
                    }
                    2 => {
                        let y = seq
                            .next_element()?
                            .ok_or_else(|| A::Error::invalid_length(1, &self))?;
                        Ok(SerializableNumber(Number::I32(y)))
                    }
                    d => Err(A::Error::invalid_value(
                        serde::de::Unexpected::Unsigned(d.into()),
                        &"1, 2",
                    )),
                }
            }
        }

        deserializer.deserialize_tuple(2, SerializableNumberVisitor)
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

pub mod merge {
    use std::iter::Peekable;

    pub struct MergedIterator<
        K,
        V,
        I1: Iterator<Item = (K, V)>,
        I2: Iterator<Item = (K, V)>,
        Merger: Fn(&K, V, V) -> V,
    > {
        iter1: Peekable<I1>,
        iter2: Peekable<I2>,
        merger: Merger,
    }

    impl<
            K: Ord + Eq,
            V,
            I1: Iterator<Item = (K, V)>,
            I2: Iterator<Item = (K, V)>,
            Merger: Fn(&K, V, V) -> V,
        > MergedIterator<K, V, I1, I2, Merger>
    {
        pub fn new(iter1: I1, iter2: I2, merger: Merger) -> Self {
            Self {
                iter1: iter1.peekable(),
                iter2: iter2.peekable(),
                merger,
            }
        }
    }

    impl<
            K: Ord + Eq,
            V,
            I1: Iterator<Item = (K, V)>,
            I2: Iterator<Item = (K, V)>,
            Merger: Fn(&K, V, V) -> V,
        > Iterator for MergedIterator<K, V, I1, I2, Merger>
    {
        type Item = (K, V);

        fn next(&mut self) -> Option<Self::Item> {
            let first = self.iter1.peek();
            let second = self.iter2.peek();

            match (first, second) {
                (Some((k1, _)), Some((k2, _))) => {
                    let cmp = k1.cmp(k2);
                    match cmp {
                        std::cmp::Ordering::Less => self.iter1.next(),
                        std::cmp::Ordering::Greater => self.iter2.next(),
                        std::cmp::Ordering::Equal => {
                            let (k1, v1) = self.iter1.next().unwrap();
                            let (_, v2) = self.iter2.next().unwrap();
                            let v = (self.merger)(&k1, v1, v2);
                            Some((k1, v))
                        }
                    }
                }
                (Some(_), None) => self.iter1.next(),
                (None, Some(_)) => self.iter2.next(),
                (None, None) => None,
            }
        }
    }

    /// This function should merge committed and uncommitted data.
    /// The idea is to have 2 iterators, one for each data source.
    /// The iterators should be already sorted.
    /// The merged iterator should be sorted too.
    /// The merger function is called to resolve conflicts on the same key.
    pub fn merge<
        K: Ord + Eq,
        V,
        I1: Iterator<Item = (K, V)>,
        I2: Iterator<Item = (K, V)>,
        Merger: Fn(&K, V, V) -> V,
    >(
        iter1: I1,
        iter2: I2,
        merger: Merger,
    ) -> MergedIterator<K, V, I1, I2, Merger> {
        MergedIterator {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            merger,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use anyhow::Result;

    use crate::{indexes::number::Number, test_utils::generate_new_path, types::DocumentId};

    use super::{merge::*, CommittedNumberFieldIndex};

    #[test]
    fn test_indexes_number_merge() {
        let iter1 = vec![(1, vec![1]), (2, vec![2]), (3, vec![3])].into_iter();
        let iter2 = vec![(1, vec![1]), (2, vec![2]), (3, vec![3])].into_iter();

        let merger = |_: &i32, mut v1: Vec<i32>, v2: Vec<i32>| {
            v1.extend(v2);
            v1
        };

        let merged: Vec<_> = merge(iter1, iter2, &merger).collect();

        assert_eq!(
            vec![(1, vec![1, 1]), (2, vec![2, 2]), (3, vec![3, 3])],
            merged
        );
    }

    #[test]
    fn test_indexes_number_merge_2() {
        let iter1 = vec![(1, vec![1]), (3, vec![2]), (5, vec![3])].into_iter();
        let iter2 = vec![(2, vec![1]), (4, vec![2]), (6, vec![3])].into_iter();

        let merger = |_: &i32, _: Vec<i32>, _: Vec<i32>| {
            panic!("This should not be called");
        };

        let merged: Vec<_> = merge(iter1, iter2, &merger).collect();

        assert_eq!(
            vec![
                (1, vec![1]),
                (2, vec![1]),
                (3, vec![2]),
                (4, vec![2]),
                (5, vec![3]),
                (6, vec![3])
            ],
            merged
        );
    }

    #[test]
    fn test_indexes_number_merge_3() {
        let iter1 = vec![(2, vec![1]), (4, vec![2]), (6, vec![3])].into_iter();
        let iter2 = vec![(1, vec![1]), (3, vec![2]), (5, vec![3])].into_iter();

        let merger = |_: &i32, _: Vec<i32>, _: Vec<i32>| {
            panic!("This should not be called");
        };

        let merged: Vec<_> = merge(iter1, iter2, &merger).collect();

        assert_eq!(
            vec![
                (1, vec![1]),
                (2, vec![1]),
                (3, vec![2]),
                (4, vec![2]),
                (5, vec![3]),
                (6, vec![3])
            ],
            merged
        );
    }

    #[test]
    fn test_indexes_number_merge_4() {
        let iter1 = vec![(1, vec![1]), (2, vec![2]), (3, vec![3])].into_iter();
        let iter2 = vec![(4, vec![1]), (5, vec![2]), (6, vec![3])].into_iter();

        let merger = |_: &i32, _: Vec<i32>, _: Vec<i32>| {
            panic!("This should not be called");
        };

        let merged: Vec<_> = merge(iter1, iter2, &merger).collect();

        assert_eq!(
            vec![
                (1, vec![1]),
                (2, vec![2]),
                (3, vec![3]),
                (4, vec![1]),
                (5, vec![2]),
                (6, vec![3])
            ],
            merged
        );
    }

    #[test]
    fn test_indexes_number_merge_5() {
        let iter1 = vec![(4, vec![1]), (5, vec![2]), (6, vec![3])].into_iter();
        let iter2 = vec![(1, vec![1]), (2, vec![2]), (3, vec![3])].into_iter();

        let merger = |_: &i32, _: Vec<i32>, _: Vec<i32>| {
            panic!("This should not be called");
        };

        let merged: Vec<_> = merge(iter1, iter2, &merger).collect();

        assert_eq!(
            vec![
                (1, vec![1]),
                (2, vec![2]),
                (3, vec![3]),
                (4, vec![1]),
                (5, vec![2]),
                (6, vec![3])
            ],
            merged
        );
    }

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
            CommittedNumberFieldIndex::from_iter(data, generate_new_path())?;

        let output = committed_number_field_index
            .filter(&crate::indexes::number::NumberFilter::Equal(Number::I32(0)))?;
        assert_eq!(output, HashSet::from_iter([DocumentId(0)]));

        Ok(())
    }
}
