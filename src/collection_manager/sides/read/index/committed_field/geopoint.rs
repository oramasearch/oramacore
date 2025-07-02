use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context, Result};
use bkd::{haversine_distance, BKDTree, Coord};
use serde::{Deserialize, Serialize};

use crate::{
    file_utils::{create_if_not_exists, BufferedFile},
    types::{DocumentId, GeoSearchFilter},
};

#[derive(Debug)]
pub struct CommittedGeoPointField {
    field_path: Box<[String]>,
    tree: BKDTree<f32, DocumentId>,
    data_dir: PathBuf,
}

impl CommittedGeoPointField {
    pub fn from_raw(
        tree: BKDTree<f32, DocumentId>,
        field_path: Box<[String]>,
        data_dir: PathBuf,
    ) -> Result<Self> {
        let s = Self {
            field_path,
            tree,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    pub fn try_load(info: GeoPointFieldInfo) -> Result<Self> {
        let data_dir = info.data_dir;
        let tree: BKDTree<f32, DocumentId>  = BufferedFile::open(data_dir.join("geopoint_tree.bin"))
            .context("Cannot open geopoint file")?
            .read_bincode_data()
            .context("Cannot deserialize geopoint file")?;

        let s = Self {
            data_dir: data_dir,
            field_path: info.field_path,
            tree,
        };

        // Is it really needed?
        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    pub fn update<'s, 'iter, 'to_delete>(
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
        // Ensure the data directory exists
        create_if_not_exists(&self.data_dir).context("Failed to create data directory")?;

        BufferedFile::create_or_overwrite(self.data_dir.join("geopoint_tree.bin"))
            .context("Cannot open geopoint file")?
            .write_bincode_data(&self.tree)
            .context("Cannot deserialize geopoint file")?;

        Ok(())
    }

    pub fn get_field_info(&self) -> GeoPointFieldInfo {
        GeoPointFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.data_dir.clone(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn stats(&self) -> Result<CommittedGeoPointFieldStats> {
        Ok(CommittedGeoPointFieldStats {
            count: self.tree.len(),
        })
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_geopoint: &GeoSearchFilter,
    ) -> Box<dyn Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        match filter_geopoint {
            GeoSearchFilter::Radius(filter) => Box::new(
                self.tree
                    .search_by_radius(
                        Coord::new(filter.coordinates.lat, filter.coordinates.lon),
                        filter.value.to_meter(filter.unit),
                        haversine_distance,
                        filter.inside,
                    )
                    .copied(),
            ),
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
                Box::new(iter)
            }
        }
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
