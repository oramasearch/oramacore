use std::{
    collections::{HashMap, HashSet}, fmt::Debug, path::PathBuf
};

use anyhow::{anyhow, Context, Result};

use serde::{Deserialize, Serialize};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile}, indexes::hnsw::{Builder, HnswMap, Point, Search}, types::DocumentId
};

#[derive(Clone, Serialize, Deserialize, Debug)]
struct VectorPoint {
    magnitude: f32,
    point: Vec<f32>,
}

impl VectorPoint {
    fn new(v: Vec<f32>) -> Self {
        let magnitude = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude = magnitude.max(0.0001);
        debug_assert!(magnitude > 0.0, "Magnitude is zero");

        Self {
            magnitude,
            point: v,
        }
    }
}
impl Point for VectorPoint {
    fn distance(&self, other: &Self) -> f32 {
        let my_mag = self.magnitude;
        let other_mag = other.magnitude;
        let dot_product = self
            .point
            .iter()
            .zip(other.point.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let similarity = dot_product / (my_mag * other_mag);

        // cosine similarity is between 0 and 1.
        // 1 = equal
        // 0 = orthogonal
        // Anyway the definition of distance (https://en.wikipedia.org/wiki/Distance#Mathematical_formalization)
        // consider 0 = equal.
        // So we need to invert the similarity to have a distance.
        1.0 - similarity
    }
}

pub struct VectorField {
    inner: HnswMap<VectorPoint, DocumentId>,
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
    pub fn from_iter<I>(iter: I, _dimension: usize, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
    {

        println!("Flatting iter.... ------ ");
        let iter = iter.flat_map(|(doc_id, vectors)| {
            vectors.into_iter().map(move |vector| (VectorPoint::new(vector), doc_id))
        });
        println!("Unzipping.. ------ ");
        let (points, values): (Vec<_>, Vec<_>) = iter.unzip();
        println!("Unzipped.. ------ ");
        println!("Building..... ------ ");
        let inner: HnswMap<VectorPoint, DocumentId> = Builder::default().build(points, values);
        println!("Built ------ ");

        create_if_not_exists(&data_dir)
            .context("Cannot create data directory")?;
        println!("Committing.... ------ ");
        BufferedFile::create(data_dir.join("index.hnsw"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&inner)
            .context("Cannot write hnsw file")?;
        println!("Committed.... ------ ");

        Ok(Self {
            inner,
            data_dir,
        })
    }

    pub fn from_dump_and_iter(
        data_dir: PathBuf,
        iter: impl Iterator<Item = (DocumentId, Vec<Vec<f32>>)>,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        new_data_dir: PathBuf,
    ) -> Result<Self> {
        let dump_file_path = data_dir.join("index.hnsw");

        let inner: HnswMap<VectorPoint, DocumentId> = BufferedFile::open(dump_file_path)
            .context("Cannot open hnsw file")?
            .read_bincode_data()
            .context("Cannot read hnsw file")?;

        let iter = iter
            .flat_map(|(doc_id, vectors)| {
                vectors.into_iter().map(move |vector| (VectorPoint::new(vector), doc_id))
            })
            .chain(inner.into_iter())
            .filter(|(_, doc_id)| !uncommitted_document_deletions.contains(doc_id));

        let (points, values): (Vec<_>, Vec<_>) = iter.unzip();
        let inner: HnswMap<VectorPoint, DocumentId> = Builder::default().build(points, values);

        println!("1 Committing.... ------ ");
        create_if_not_exists(&new_data_dir)
            .context("Cannot create data directory")?;
        BufferedFile::create(new_data_dir.join("index.hnsw"))
            .context("Cannot create hnsw file")?
            .write_bincode_data(&inner)
            .context("Cannot write hnsw file")?;
        println!("1 Committed ------ ");

        Ok(Self {
            inner,
            data_dir: new_data_dir,
        })
    }

    pub fn load(info: VectorFieldInfo) -> Result<Self> {
        let dump_file_path = info.data_dir.join("index.hnsw");

        let inner: HnswMap<VectorPoint, DocumentId> = BufferedFile::open(dump_file_path)
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
        let mut search = Search::default();
        let cambridge_blue = VectorPoint::new(target.to_vec());
        let iter_results = self.inner.search(&cambridge_blue, &mut search);

        let mut skipped = 0;
        let mut found = 0;
        for r in iter_results {
            let distance = r.distance;
            // See comment above
            let score = 1.0 - distance;
            let doc_id = *r.value;

            if filtered_doc_ids.map_or(false, |ids| !ids.contains(&doc_id)) {
                continue;
            }
            if uncommitted_deleted_documents.contains(&doc_id) {
                continue;
            }

            // HNSW returns the results but it does not guarantee the order of the results.
            // We count the number of the skipped documents due to the similarity.
            // If we filtered 100 documents due to the similarity, we stop the search.
            // This is a heuristic to stop the loop hopping the HNSW graph returns the relevant documents before.
            // Should we put this `100` as a parameter?
            // TODO: think about it
            if skipped > 100 {
                break;
            }

            if score >= similarity {
                let v = output.entry(doc_id).or_insert(0.0);
                *v += score;

                found += 1;

                // Reach the number of documents we want to return
                if found > limit {
                    break;
                }
            } else {
                skipped += 1;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorFieldInfo {
    pub data_dir: PathBuf,
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
        index.search(
            &[0.1, 0.1, 0.1],
            0.6,
            5,
            None,
            &mut output,
            &HashSet::new(),
        ).unwrap();
        assert_eq!(HashSet::from([
            DocumentId(1),
            DocumentId(0),
        ]), output.keys().cloned().collect());

        let info = index.get_field_info();

        let index = VectorField::load(info).unwrap();

        let mut output = HashMap::new();
        index.search(
            &[0.1, 0.1, 0.1],
            0.6,
            5,
            None,
            &mut output,
            &HashSet::new(),
        ).unwrap();
        assert_eq!(HashSet::from([
            DocumentId(1),
            DocumentId(0),
        ]), output.keys().cloned().collect());
    }
}
