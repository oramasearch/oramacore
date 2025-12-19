use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use oramacore_lib::bkd;
use oramacore_lib::bkd::{haversine_distance, BKDTree, Coord};
use serde::{Deserialize, Serialize};

use crate::collection_manager::sides::read::index::uncommitted_field::UncommittedGeoPointFilterField;
use crate::collection_manager::sides::read::index::{
    filter::Filterable,
    merge::{CommittedField, CommittedFieldMetadata, Field},
};
use crate::collection_manager::sides::read::OffloadFieldConfig;
use crate::types::{DocumentId, GeoSearchFilter};
use oramacore_lib::fs::BufferedFile;

#[derive(Debug)]
pub struct CommittedGeoPointField {
    field_path: Box<[String]>,
    tree: BKDTree<f32, DocumentId>,
    data_dir: PathBuf,
}

impl CommittedGeoPointField {
    fn update<'s, 'iter, 'to_delete>(
        &'s mut self,
        iter: impl Iterator<Item = &'iter bkd::Point<f32, DocumentId>> + 'iter,
        to_delete: &'to_delete HashSet<DocumentId>,
        data_dir: PathBuf,
    ) -> Result<()> {
        for point in iter {
            self.tree.insert(point.clone());
        }
        self.tree.delete(to_delete);

        self.data_dir = data_dir;

        self.commit()?;

        Ok(())
    }

    fn commit(&self) -> Result<()> {
        BufferedFile::create_or_overwrite(self.data_dir.join("geopoint_tree.bin"))
            .context("Cannot open geopoint file")?
            .write_bincode_data(&self.tree)
            .context("Cannot deserialize geopoint file")?;

        Ok(())
    }
}

impl CommittedField for CommittedGeoPointField {
    type FieldMetadata = GeoPointFieldInfo;
    type Uncommitted = UncommittedGeoPointFilterField;

    fn from_uncommitted(
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let mut tree = uncommitted.inner();
        tree.delete(uncommitted_document_deletions);

        let s = Self {
            field_path: uncommitted.field_path().into(),
            tree,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    fn try_load(
        metadata: Self::FieldMetadata,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let data_dir = metadata.data_dir;
        let tree: BKDTree<f32, DocumentId> = BufferedFile::open(data_dir.join("geopoint_tree.bin"))
            .context("Cannot open geopoint file")?
            .read_bincode_data()
            .context("Cannot deserialize geopoint file")?;

        let s = Self {
            data_dir,
            field_path: metadata.field_path,
            tree,
        };

        // Is it really needed?
        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    fn add_uncommitted(
        &self,
        uncommitted: &Self::Uncommitted,
        data_dir: PathBuf,
        uncommitted_document_deletions: &HashSet<DocumentId>,
        _offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        let info = self.metadata();

        let old_data_dir = info.data_dir;
        let tree: BKDTree<f32, DocumentId> =
            BufferedFile::open(old_data_dir.join("geopoint_tree.bin"))
                .context("Cannot open geopoint file")?
                .read_bincode_data()
                .context("Cannot deserialize geopoint file")?;

        let mut field = Self {
            data_dir: data_dir.clone(),
            field_path: info.field_path,
            tree,
        };

        field.update(uncommitted.iter(), uncommitted_document_deletions, data_dir)?;

        Ok(field)
    }

    fn metadata(&self) -> Self::FieldMetadata {
        GeoPointFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }
}

impl Field for CommittedGeoPointField {
    type FieldStats = CommittedGeoPointFieldStats;

    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }

    fn stats(&self) -> CommittedGeoPointFieldStats {
        CommittedGeoPointFieldStats {
            count: self.tree.len(),
        }
    }
}

impl Filterable for CommittedGeoPointField {
    type FilterParam = GeoSearchFilter;

    fn filter<'s, 'iter>(
        &'s self,
        filter_param: &Self::FilterParam,
    ) -> Result<Box<dyn Iterator<Item = DocumentId> + 'iter>>
    where
        's: 'iter,
    {
        let iter = match filter_param {
            GeoSearchFilter::Radius(filter) => Box::new(
                self.tree
                    .search_by_radius(
                        Coord::new(filter.coordinates.lat, filter.coordinates.lon),
                        filter.value.to_meter(filter.unit),
                        haversine_distance,
                        filter.inside,
                    )
                    .copied(),
            )
                as Box<dyn Iterator<Item = DocumentId>>,
            GeoSearchFilter::Polygon(filter) => {
                let iter = self
                    .tree
                    .search_by_polygon(
                        filter
                            .coordinates
                            .iter()
                            .map(|g| Coord::new(g.lat, g.lon))
                            .collect(),
                        false,
                    )
                    .copied();
                Box::new(iter) as Box<dyn Iterator<Item = DocumentId>>
            }
        };
        Ok(iter)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GeoPointFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Serialize, Debug)]
pub struct CommittedGeoPointFieldStats {
    pub count: usize,
}

impl CommittedFieldMetadata for GeoPointFieldInfo {
    fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    fn set_data_dir(&mut self, data_dir: PathBuf) {
        self.data_dir = data_dir;
    }
    fn field_path(&self) -> &Box<[String]> {
        &self.field_path
    }
}
