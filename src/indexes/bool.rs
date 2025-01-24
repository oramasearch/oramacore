use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

use crate::{
    collection_manager::{dto::FieldId, sides::Offset},
    field_id_hashmap::FieldIdHashMap,
    file_utils::{create_if_not_exists, BufferedFile},
    offset_storage::OffsetStorage,
    types::DocumentId,
};

#[derive(Debug, Default)]
struct BoolIndexPerField {
    true_docs: HashSet<DocumentId>,
    false_docs: HashSet<DocumentId>,
}

#[derive(Debug)]
enum BoolIndexState {
    Left,
    Right,
}

#[derive(Debug)]
struct InnerUncommittedNumberFieldIndex {
    left: BoolIndexPerField,
    right: BoolIndexPerField,
    state: BoolIndexState,
}

impl InnerUncommittedNumberFieldIndex {
    fn new() -> Self {
        Self {
            left: BoolIndexPerField::default(),
            right: BoolIndexPerField::default(),
            state: BoolIndexState::Left,
        }
    }

    fn add(&mut self, value: bool, doc_id: DocumentId) -> Result<()> {
        let doc_ids = match self.state {
            BoolIndexState::Left => &mut self.left,
            BoolIndexState::Right => &mut self.right,
        };

        if value {
            doc_ids.true_docs.insert(doc_id);
        } else {
            doc_ids.false_docs.insert(doc_id);
        }

        Ok(())
    }

    fn filter(&self, val: bool, doc_ids: &mut HashSet<DocumentId>) -> Result<()> {
        if val {
            doc_ids.extend(&self.left.true_docs);
            doc_ids.extend(&self.right.true_docs);
        } else {
            doc_ids.extend(&self.left.false_docs);
            doc_ids.extend(&self.right.false_docs);
        }

        Ok(())
    }
}

#[derive(Debug)]
struct UncommittedBoolFieldIndex {
    inner: RwLock<(OffsetStorage, InnerUncommittedNumberFieldIndex)>,
}

impl UncommittedBoolFieldIndex {
    async fn add(&self, offset: Offset, value: bool, doc_id: DocumentId) -> Result<()> {
        let mut inner = self.inner.write().await;
        // Ignore inserts with lower offset
        if offset <= inner.0.get_offset() {
            warn!("Skip insert number with lower offset");
            return Ok(());
        }

        inner.1.add(value, doc_id)?;
        inner.0.set_offset(offset);

        Ok(())
    }

    async fn filter(&self, val: bool, doc_ids: &mut HashSet<DocumentId>) -> Result<()> {
        let inner = self.inner.read().await;

        inner.1.filter(val, doc_ids)?;

        Ok(())
    }

    async fn get_offset(&self) -> Offset {
        let inner = self.inner.read().await;
        inner.0.get_offset()
    }
}

#[derive(Debug)]
struct CommittedBoolFieldIndex {
    true_docs: HashSet<DocumentId>,
    false_docs: HashSet<DocumentId>,
}

impl CommittedBoolFieldIndex {
    fn filter(&self, val: bool) -> Result<HashSet<DocumentId>> {
        let doc_ids = if val {
            self.true_docs.clone()
        } else {
            self.false_docs.clone()
        };

        Ok(doc_ids)
    }
}

#[derive(Debug)]
pub struct BoolIndex {
    uncommitted: DashMap<FieldId, UncommittedBoolFieldIndex>,
    committed: DashMap<FieldId, CommittedBoolFieldIndex>,
}

impl Default for BoolIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolIndex {
    pub fn new() -> Self {
        Self {
            uncommitted: Default::default(),
            committed: Default::default(),
        }
    }

    pub async fn add(
        &self,
        offset: Offset,
        doc_id: DocumentId,
        field_id: FieldId,
        value: bool,
    ) -> Result<()> {
        trace!("Adding bool value");

        let uncommitted =
            self.uncommitted
                .entry(field_id)
                .or_insert_with(|| UncommittedBoolFieldIndex {
                    inner: RwLock::new((
                        OffsetStorage::new(),
                        InnerUncommittedNumberFieldIndex::new(),
                    )),
                });
        uncommitted.add(offset, value, doc_id).await?;

        trace!("Bool value added");

        Ok(())
    }

    pub async fn filter(&self, field_id: FieldId, val: bool) -> Result<HashSet<DocumentId>> {
        let mut doc_ids = if let Some(committed) = self.committed.get(&field_id) {
            committed.filter(val)?
        } else {
            HashSet::new()
        };

        if let Some(uncommitted) = self.uncommitted.get(&field_id) {
            uncommitted.filter(val, &mut doc_ids).await?;
        }

        Ok(doc_ids)
    }

    pub fn load(&mut self, data_dir: PathBuf) -> Result<()> {
        let info_path = data_dir.join("info.json");
        let info: BoolIndexInfo = BufferedFile::open(info_path.clone())
            .context("Cannot open bool info.json file")?
            .read_json_data()
            .context("Cannot deserialize info.json file")?;
        let BoolIndexInfo::V1(info) = info;

        for (field_id, offset) in info.field_infos.into_inner() {
            let field_index_dump_path = data_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}.json", offset.0));

            let (true_docs, false_docs): (HashSet<DocumentId>, HashSet<DocumentId>) =
                BufferedFile::open(field_index_dump_path)
                    .context(format!("Cannot open bool field {:?} dump file", field_id))?
                    .read_json_data()
                    .context(format!("Cannot deserialize bool field {:?}", field_id))?;

            self.committed.insert(
                field_id,
                CommittedBoolFieldIndex {
                    true_docs,
                    false_docs,
                },
            );
        }

        Ok(())
    }

    pub async fn commit(&self, data_dir: PathBuf) -> Result<()> {
        info!("Committing bool index");

        create_if_not_exists(&data_dir).context("Cannot create data directory for bool index")?;

        let info_path = data_dir.join("info.json");
        let field_infos = BufferedFile::open(info_path.clone())
            .and_then(|file| file.read_json_data::<BoolIndexInfo>())
            .map(|info| match info {
                BoolIndexInfo::V1(info) => info,
            });
        let mut field_infos = match field_infos {
            Ok(info) => info.field_infos,
            Err(_) => {
                debug!("Cannot open bool info.json file, creating a new one");
                FieldIdHashMap::new()
            }
        };

        info!(
            "Committing string fields {:?}",
            self.uncommitted
                .iter()
                .map(|e| *e.key())
                .collect::<Vec<_>>()
        );

        for e in self.uncommitted.iter() {
            let (field_id, uncommitted) = e.pair();

            info!("Committing field {:?}", field_id);

            let mut uncommitted_data = uncommitted.inner.write().await;
            let (true_docs, false_docs, state) = match uncommitted_data.1.state {
                BoolIndexState::Left => (
                    uncommitted_data.1.left.true_docs.clone(),
                    uncommitted_data.1.left.false_docs.clone(),
                    BoolIndexState::Left,
                ),
                BoolIndexState::Right => (
                    uncommitted_data.1.right.true_docs.clone(),
                    uncommitted_data.1.right.false_docs.clone(),
                    BoolIndexState::Right,
                ),
            };
            uncommitted_data.1.state = match uncommitted_data.1.state {
                BoolIndexState::Left => BoolIndexState::Right,
                BoolIndexState::Right => BoolIndexState::Left,
            };
            drop(uncommitted_data);

            if true_docs.is_empty() && false_docs.is_empty() {
                info!("Everything is already committed. Skip dumping");
                continue;
            }

            let offset = uncommitted.get_offset().await;
            field_infos.insert(*field_id, offset);

            let old_committed = self.committed.get(field_id);

            info!(
                ?field_id,
                committed = old_committed.is_some(),
                uncommitted_count = true_docs.len() + false_docs.len(),
                "Committing string index"
            );

            let new_committed = if let Some(old_committed) = old_committed {
                // This clone duplicates data, so we (potentially) use a lot of memory
                // We should use a more efficient way to do this
                // TODO: Use a more efficient way to do this

                let mut old_false_docs = old_committed.false_docs.clone();
                old_false_docs.extend(false_docs);
                let mut old_true_docs = old_committed.true_docs.clone();
                old_true_docs.extend(true_docs);

                CommittedBoolFieldIndex {
                    true_docs: old_true_docs,
                    false_docs: old_false_docs,
                }
            } else {
                CommittedBoolFieldIndex {
                    true_docs,
                    false_docs,
                }
            };

            let field_index_dump_path = data_dir
                .join(format!("field-{}", field_id.0))
                .join(format!("offset-{}.json", offset.0));
            create_if_not_exists(
                field_index_dump_path
                    .parent()
                    .expect("Dump has always a parent folder"),
            )?;

            BufferedFile::create_or_overwrite(field_index_dump_path)
                .context(format!("Cannot create bool field {:?} dump file", field_id))?
                .write_json_data(&(&new_committed.true_docs, &new_committed.false_docs))
                .context(format!("Cannot serialize bool field {:?}", field_id))?;

            self.committed.insert(*field_id, new_committed);

            let mut uncommitted_data = uncommitted.inner.write().await;
            match state {
                BoolIndexState::Left => {
                    // `clear` keeps the capacity
                    // if 1_000_000 documents are added, the memory should 8MB
                    // So the capacity don't keep too much memory
                    uncommitted_data.1.left.true_docs.clear();
                    uncommitted_data.1.left.false_docs.clear();
                }
                BoolIndexState::Right => {
                    // `clear` keeps the capacity
                    // if 1_000_000 documents are added, the memory should 8MB
                    // So the capacity don't keep too much memory
                    uncommitted_data.1.right.true_docs.clear();
                    uncommitted_data.1.right.false_docs.clear();
                }
            };
            drop(uncommitted_data);

            info!("Field committed");
        }

        trace!("Dumping string index info");
        BufferedFile::create_or_overwrite(info_path)
            .context("Cannot create bool info.json file")?
            .write_json_data(&BoolIndexInfo::V1(BoolIndexInfoV1 { field_infos }))
            .context("Cannot deserialize info.json file")?;
        trace!("String index info dumped");

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
enum BoolIndexInfo {
    #[serde(rename = "1")]
    V1(BoolIndexInfoV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct BoolIndexInfoV1 {
    field_infos: FieldIdHashMap<Offset>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bool_index_filter() -> Result<()> {
        let index = BoolIndex::new();

        index
            .add(Offset(1), DocumentId(0), FieldId(0), true)
            .await?;
        index
            .add(Offset(2), DocumentId(1), FieldId(0), false)
            .await?;
        index
            .add(Offset(3), DocumentId(2), FieldId(0), true)
            .await?;
        index
            .add(Offset(4), DocumentId(3), FieldId(0), false)
            .await?;
        index
            .add(Offset(5), DocumentId(4), FieldId(0), true)
            .await?;
        index
            .add(Offset(6), DocumentId(5), FieldId(0), false)
            .await?;

        let true_docs = index.filter(FieldId(0), true).await.unwrap();
        assert_eq!(
            true_docs,
            HashSet::from([DocumentId(0), DocumentId(2), DocumentId(4)])
        );

        let false_docs = index.filter(FieldId(0), false).await.unwrap();
        assert_eq!(
            false_docs,
            HashSet::from([DocumentId(1), DocumentId(3), DocumentId(5)])
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_bool_index_filter_unknown_field() -> Result<()> {
        let index = BoolIndex::new();

        index
            .add(Offset(1), DocumentId(0), FieldId(0), true)
            .await?;
        index
            .add(Offset(2), DocumentId(1), FieldId(0), false)
            .await?;
        index
            .add(Offset(3), DocumentId(2), FieldId(0), true)
            .await?;
        index
            .add(Offset(4), DocumentId(3), FieldId(0), false)
            .await?;
        index
            .add(Offset(5), DocumentId(4), FieldId(0), true)
            .await?;
        index
            .add(Offset(6), DocumentId(5), FieldId(0), false)
            .await?;

        let true_docs = index.filter(FieldId(1), true).await.unwrap();
        assert_eq!(true_docs, HashSet::from([]));

        let false_docs = index.filter(FieldId(1), false).await.unwrap();
        assert_eq!(false_docs, HashSet::from([]));

        Ok(())
    }
}
