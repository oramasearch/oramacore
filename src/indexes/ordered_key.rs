use std::{collections::HashSet, fmt::Debug, hash::Hash, path::PathBuf};

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tracing::error;

use crate::file_utils::BufferedFile;

const BOUND_FILE_NAME: &str = "bounds.bin";

#[derive(Debug, Serialize, Deserialize)]
pub struct Item<Key, Value: Eq + Hash> {
    pub key: Key,
    // Vec is not the best data structure here.
    // Should we use a smallvec?
    // TODO: think about this.
    pub values: HashSet<Value>,
}

impl<Key: Clone, Value: Eq + Hash + Clone> Clone for Item<Key, Value> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            values: self.values.clone(),
        }
    }
}

impl<Key: PartialEq, Value: Eq + Hash> PartialEq for Item<Key, Value> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl<Key: Eq, Value: Eq + Hash> Eq for Item<Key, Value> {}

impl<Key: Ord, Value: Eq + Hash> PartialOrd for Item<Key, Value> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<Key: Ord, Value: Eq + Hash> Ord for Item<Key, Value> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

#[derive(Debug)]
enum PagePointer<Key, Value: Eq + Hash> {
    InMemory {
        path: PathBuf,
        items: Vec<Item<Key, Value>>,
    },
    OnFile {
        path: PathBuf,
    },
}

impl<Key, Value: Eq + Hash> Serialize for PagePointer<Key, Value> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let path = match self {
            PagePointer::InMemory { path, .. } => path,
            PagePointer::OnFile { path } => path,
        };

        path.serialize(serializer)
    }
}

impl<'de, Key, Value: Eq + Hash> Deserialize<'de> for PagePointer<Key, Value> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let path = PathBuf::deserialize(deserializer)?;
        Ok(PagePointer::OnFile { path })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct ChunkId(usize);

#[derive(Debug, Serialize, Deserialize)]
pub struct Page<Key, Value: Eq + Hash> {
    id: ChunkId,
    pointer: PagePointer<Key, Value>,
    min: Key,
    max: Key,
}

impl<
        Key: DeserializeOwned + Serialize + Clone + Debug,
        Value: Eq + Hash + DeserializeOwned + Serialize + Clone + Debug,
    > Page<Key, Value>
{
    pub fn get_page_items(&self) -> Result<Vec<Item<Key, Value>>> {
        match &self.pointer {
            PagePointer::InMemory { items, .. } => Ok(items.to_vec()),
            PagePointer::OnFile { path } => {
                let items = self.load_items(path)?;
                Ok(items)
            }
        }
    }

    pub fn count(&self) -> Result<usize> {
        match &self.pointer {
            PagePointer::InMemory { items, .. } => Ok(items.len()),
            PagePointer::OnFile { path } => {
                let items = self.load_items(path)?;
                Ok(items.len())
            }
        }
    }

    pub fn min_item_key(&self) -> Result<Key> {
        match &self.pointer {
            PagePointer::InMemory { items, .. } => Ok(items[0].key.clone()),
            PagePointer::OnFile { path } => {
                let items = self.load_items(path)?;
                Ok(items[0].key.clone())
            }
        }
    }

    pub fn max_item_key(&self) -> Result<Key> {
        match &self.pointer {
            PagePointer::InMemory { items, .. } => Ok(items.last().unwrap().key.clone()),
            PagePointer::OnFile { path } => {
                let items = self.load_items(path)?;
                Ok(items.last().unwrap().key.clone())
            }
        }
    }

    fn save_on_disk(&self) -> Result<()> {
        match &self.pointer {
            PagePointer::InMemory { items, path } => {
                BufferedFile::create_or_overwrite(path.clone())
                    .context("Cannot create page file")?
                    .write_bincode_data(items)
                    .context("Cannot serialize page")?;
            }
            PagePointer::OnFile { .. } => {}
        };
        Ok(())
    }

    fn load_items(&self, p: &PathBuf) -> Result<Vec<Item<Key, Value>>> {
        BufferedFile::open(p)
            .context("Cannot open page file")?
            .read_bincode_data()
            .context("Cannot deserialize page")
    }
}

pub trait BoundedValue {
    fn max_value() -> Self;
    fn min_value() -> Self;
}

pub struct OrderedKeyIndex<Key, Value: Eq + Hash> {
    pages: Vec<Page<Key, Value>>,
    bounds: Vec<(Key, Key)>,
}

impl<Key: Debug, Value: Eq + Hash + Debug> Debug for OrderedKeyIndex<Key, Value> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedIndex")
            .field("pages", &self.pages)
            .field("bounds", &self.bounds)
            .finish()
    }
}

const MAX_NUMBER_PER_PAGE: usize = 1000;
impl<
        Key: Ord + DeserializeOwned + Clone + Serialize + BoundedValue + Debug,
        Value: Eq + Hash + Clone + DeserializeOwned + Serialize + Debug,
    > OrderedKeyIndex<Key, Value>
{
    pub fn from_iter<I>(iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (Key, HashSet<Value>)>,
    {
        std::fs::create_dir_all(&data_dir)
            .context("Cannot create the base directory for the committed index")?;

        let mut items: Vec<Item<Key, Value>> = Vec::new();

        let mut bounds = Vec::new();
        let mut pages = Vec::new();

        let mut prev = Key::min_value();
        let mut current_page_count = 0;
        for (value, doc_ids) in iter {
            assert!(value > prev);
            prev = value.clone();
            current_page_count += doc_ids.len();

            if current_page_count > MAX_NUMBER_PER_PAGE {
                let min = items.first().expect("The item is empty").key.clone();
                let max = value.clone();

                let page_id = pages.len();
                let page_file = data_dir.join(format!("page_{}.bin", page_id));
                let current_page = Page {
                    id: ChunkId(page_id),
                    pointer: PagePointer::<Key, Value>::InMemory {
                        path: page_file.clone(),
                        items,
                    },
                    min: min.clone(),
                    max: max.clone(),
                };
                current_page.save_on_disk()?;

                bounds.push((min.clone(), max.clone()));
                pages.push(Page {
                    id: ChunkId(pages.len()),
                    pointer: PagePointer::<Key, Value>::OnFile { path: page_file },
                    min: current_page.min,
                    max: current_page.max,
                });

                items = Vec::new();
                current_page_count = 0;
            }

            items.push(Item {
                key: value,
                values: doc_ids,
            });
        }

        let page_id = pages.len();
        let max = Key::max_value();
        let min = items.first().expect("The item is empty").key.clone();
        let page_file = data_dir.join(format!("page_{}.bin", page_id));
        let current_page = Page {
            id: ChunkId(page_id),
            pointer: PagePointer::<Key, Value>::InMemory {
                path: page_file.clone(),
                items,
            },
            min: min.clone(),
            max: max.clone(),
        };
        current_page.save_on_disk()?;

        bounds.push((min.clone(), max.clone()));
        pages.push(Page {
            id: ChunkId(pages.len()),
            pointer: PagePointer::<Key, Value>::OnFile { path: page_file },
            min: current_page.min,
            max: current_page.max,
        });

        pages[0].min = Key::min_value();
        bounds[0].0 = Key::min_value();

        let bounds_file = data_dir.join(BOUND_FILE_NAME);
        BufferedFile::create_or_overwrite(bounds_file.clone())
            .context("Cannot create bounds file")?
            .write_bincode_data(&bounds)
            .context("Cannot serialize bounds")?;

        let s = Self { pages, bounds };

        Ok(s)
    }

    pub fn load(data_dir: PathBuf) -> Result<Self> {
        let bounds_file = data_dir.join(BOUND_FILE_NAME);
        let bounds: Vec<(Key, Key)> = BufferedFile::open(bounds_file.clone())
            .context("Cannot create bounds file")?
            .read_bincode_data()
            .context("Cannot serialize bounds")?;
        let pages = bounds
            .iter()
            .enumerate()
            .map(|(index, (min, max))| {
                let page_file = data_dir.join(format!("page_{}.bin", index));
                Page {
                    id: ChunkId(index),
                    pointer: PagePointer::<Key, Value>::OnFile { path: page_file },
                    min: min.clone(),
                    max: max.clone(),
                }
            })
            .collect();

        Ok(Self { pages, bounds })
    }

    pub fn count(&self, key: Key) -> Result<usize> {
        let page_index = self.find_page_index(&key)?;
        let page = self
            .pages
            .get(page_index)
            .context("The page is not present")?;
        let items = page.get_page_items()?;
        let count = items
            .into_iter()
            .filter(|item| item.key == key)
            .map(|item| item.values.len())
            .sum();
        Ok(count)
    }

    pub fn min_max(&self) -> Result<(Key, Key)> {
        let min = self.pages.first().expect("The page is empty");
        let min = min.min_item_key()?;
        let max = self.pages.last().expect("The page is empty");
        let max = max.max_item_key()?;
        Ok((min, max))
    }

    pub fn get_items(
        &self,
        min: Key,
        max: Key,
    ) -> Result<impl Iterator<Item = Item<Key, Value>> + '_> {
        let min_page_index = self.find_page_index(&min)?;
        let max_page_index = self.find_page_index(&max)?;

        let min = min.clone();
        let max = max.clone();

        Ok((min_page_index..=max_page_index)
            .filter_map(|i| self.pages.get(i))
            .filter_map(|page| page.get_page_items().ok())
            .flat_map(move |items| {
                let min = min.clone();
                let max = max.clone();

                items
                    .into_iter()
                    .skip_while(move |p| p.key < min)
                    .take_while(move |p| p.key <= max)
            }))
    }

    fn find_page_index(&self, value: &Key) -> Result<usize> {
        if self.pages.is_empty() {
            // This should never fail.
            // We could put an empty page, so we can avoid this check.
            return Err(anyhow::anyhow!("No pages in the index"));
        }

        let pos = self
            .bounds
            .binary_search_by_key(value, |(bounds, _)| bounds.clone());

        let page_index = pos
            // If the value i'm looking for is contained in a boud, the `binary_search_by_key` returns a error.
            // That error is the index where the value should be inserted to keep the array sorted.
            // Because our pages are:
            // - sorted
            // - contiguous
            // the page I'm looking for is the one before that index.
            .unwrap_or_else(|i| {
                if i == 0 {
                    error!(r#"binary_search on PagedData identify a number less then NEG_INFINITY (the first lower bound).
And this should not happen. Return the first page."#);
                    return 0;
                }
                if i > self.pages.len() {
                    return self.bounds.len() - 1;
                }
                i - 1
            });

        Ok(page_index)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Key, HashSet<Value>)> + '_ {
        self.pages
            .iter()
            .filter_map(|page| page.get_page_items().ok())
            .flat_map(|items| items.into_iter().map(|item| (item.key, item.values)))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        collection_manager::dto::{Number, SerializableNumber},
        tests::utils::generate_new_path,
        types::DocumentId,
    };

    use super::*;

    #[test]
    fn test_indexes_pages_data() -> Result<()> {
        let data = vec![
            (
                SerializableNumber(Number::I32(-1)),
                HashSet::from([DocumentId(1), DocumentId(2), DocumentId(3)]),
            ),
            (
                SerializableNumber(Number::I32(1)),
                HashSet::from([DocumentId(4), DocumentId(5), DocumentId(6)]),
            ),
            (
                SerializableNumber(Number::I32(2)),
                HashSet::from([DocumentId(1), DocumentId(3), DocumentId(5)]),
            ),
        ];
        let data_dir = generate_new_path();
        let paged_index = OrderedKeyIndex::from_iter(data.into_iter(), data_dir.clone())?;
        test(&paged_index)?;
        let paged_index = OrderedKeyIndex::load(data_dir)?;
        test(&paged_index)?;

        fn test(paged_index: &OrderedKeyIndex<SerializableNumber, DocumentId>) -> Result<()> {
            let output = paged_index.get_items(
                SerializableNumber(Number::I32(1)),
                SerializableNumber(Number::I32(1)),
            )?;
            let output: HashSet<_> = output.flat_map(|item| item.values.into_iter()).collect();
            assert_eq!(
                output,
                HashSet::from([DocumentId(4), DocumentId(5), DocumentId(6)])
            );

            let output = paged_index.get_items(
                SerializableNumber(Number::I32(-10)),
                SerializableNumber(Number::I32(-8)),
            )?;
            let output: HashSet<_> = output.flat_map(|item| item.values.into_iter()).collect();
            assert_eq!(output, HashSet::from([]));

            let output = paged_index.get_items(
                SerializableNumber(Number::I32(-10)),
                SerializableNumber(Number::I32(10)),
            )?;
            let output: HashSet<_> = output.flat_map(|item| item.values.into_iter()).collect();
            assert_eq!(
                output,
                HashSet::from([
                    DocumentId(1),
                    DocumentId(2),
                    DocumentId(3),
                    DocumentId(4),
                    DocumentId(5),
                    DocumentId(6)
                ])
            );

            Ok(())
        }

        Ok(())
    }
}
