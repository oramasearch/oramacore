use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    path::PathBuf,
};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    indexes::hnsw2::HNSW2Index,
    types::DocumentId,
};

pub struct VectorField {
    inner: HNSW2Index,
    data_dir: PathBuf,
}

impl Debug for VectorField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = serde_json::to_string(&self.inner).unwrap();
        f.debug_struct("VectorField")
            .field("inner", &inner)
            .field("data_dir", &self.data_dir)
            .finish()
    }
}

impl VectorField {
    pub fn from_iter<I>(iter: I, dim: usize, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {
        let mut inner = HNSW2Index::new(dim);
        for (doc_id, vectors) in iter {
            for vector in vectors {
                inner.add(&vector, doc_id).context("Cannot add vector")?;
            }
        }
        inner.build().context("Cannot build hnsw index")?;

        create_if_not_exists(&data_dir).context("Cannot create data directory")?;
        BufferedFile::create(data_dir.join("index.hnsw"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&inner)
            .context("Cannot write hnsw file")?;

        Ok(Self { inner, data_dir })
    }

    pub fn from_dump_and_iter(
        data_dir: PathBuf,
        iter: impl ExactSizeIterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        new_data_dir: PathBuf,
    ) -> Result<Self> {
        let dump_file_path = data_dir.join("index.hnsw");

        let inner: HNSW2Index = BufferedFile::open(dump_file_path)
            .context("Cannot open hnsw file")?
            .read_bincode_data()
            .context("Cannot read hnsw file")?;
        let dim = inner.dim();

        let iter = iter
            .flat_map(|(doc_id, vectors)| vectors.into_iter().map(move |vector| (doc_id, vector)))
            .chain(inner.into_data())
            .filter(|(doc_id, _)| !uncommitted_document_deletions.contains(doc_id));

        let mut new_inner = HNSW2Index::new(dim);

        for (doc_id, vector) in iter {
            new_inner
                .add(&vector, doc_id)
                .context("Cannot add vector")?;
        }
        new_inner.build().context("Cannot build hnsw index")?;

        create_if_not_exists(&new_data_dir).context("Cannot create data directory")?;
        BufferedFile::create(new_data_dir.join("index.hnsw"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&new_inner)
            .context("Cannot write hnsw file")?;

        Ok(Self {
            inner: new_inner,
            data_dir: new_data_dir,
        })
    }

    pub fn load(info: VectorFieldInfo) -> Result<Self> {
        let dump_file_path = info.data_dir.join("index.hnsw");

        let inner: HNSW2Index = BufferedFile::open(dump_file_path)
            .map_err(|e| anyhow!("Cannot open hnsw file: {}", e))?
            .read_bincode_data()
            .map_err(|e| anyhow!("Cannot read hnsw file: {}", e))?;

        Ok(Self {
            inner,
            data_dir: info.data_dir,
        })
    }

    pub fn get_field_info(&self) -> VectorFieldInfo {
        VectorFieldInfo {
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn search(
        &self,
        target: &[f32],
        similarity: f32,
        limit: usize,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        output: &mut HashMap<DocumentId, f32>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<()> {
        // We filtered matches by:
        // - `filtered_doc_ids`: removed by `search` method
        // - `uncommitted_deleted_documents`: removed by `document deletion` method
        // - `similarity` threshold: removed by `search` method
        // If there're not uncomitted deletions or the user doesn't filter, the limit is ok:
        // HNSW returns the most probable matches first, so we can stop when we reach the limit.
        // Otherwise, we should continue the search till reach the limit.
        // Anyway, the implementation below returns a Vec, so we should redo the search till reach the limit.
        // For now, we just double the limit.
        // TODO: implement a better way to handle this.
        let limit = if filtered_doc_ids.is_none() && uncommitted_deleted_documents.is_empty() {
            limit
        } else {
            limit * 2
        };
        let data = self.inner.search(target.to_vec(), limit * 2);

        for (doc_id, score) in data {
            if filtered_doc_ids.is_some_and(|ids| !ids.contains(&doc_id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(&doc_id) {
                continue;
            }

            if score >= similarity {
                let v = output.entry(doc_id).or_insert(0.0);
                *v += score;
            }
        }

        Ok(())
    }

    pub fn get_stats(&self) -> Result<VectorCommittedFieldStats> {
        Ok(VectorCommittedFieldStats {
            len: self.inner.len(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorFieldInfo {
    pub data_dir: PathBuf,
}

#[derive(Serialize)]
pub struct VectorCommittedFieldStats {
    pub len: usize,
}

#[cfg(test)]
mod tests {
    use crate::tests::utils::generate_new_path;

    use super::*;

    #[test]
    fn test_vector_index() {
        let data = [
            (DocumentId(0), vec![vec![1.0, 1.0, 1.0]]),
            (DocumentId(1), vec![vec![2.0, 2.0, 2.0]]),
            (DocumentId(2), vec![vec![-1.0, -1.0, -1.0]]),
            (DocumentId(3), vec![vec![0.0, 0.0, 0.0]]),
        ];

        let index = VectorField::from_iter(data.into_iter(), 3, generate_new_path()).unwrap();

        let mut output = HashMap::new();
        index
            .search(&[0.1, 0.1, 0.1], 0.6, 5, None, &mut output, &HashSet::new())
            .unwrap();
        println!("output: {:?}", output);
        assert_eq!(
            HashSet::from([DocumentId(1), DocumentId(0),]),
            output.keys().cloned().collect()
        );

        let info = index.get_field_info();

        let index = VectorField::load(info).unwrap();

        let mut output = HashMap::new();
        index
            .search(&[0.1, 0.1, 0.1], 0.6, 5, None, &mut output, &HashSet::new())
            .unwrap();
        assert_eq!(
            HashSet::from([DocumentId(1), DocumentId(0),]),
            output.keys().cloned().collect()
        );
    }
}
