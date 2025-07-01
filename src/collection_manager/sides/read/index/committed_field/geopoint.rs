use std::path::PathBuf;

use anyhow::{Context, Result};
use bkd::{haversine_distance, BKDTree, Coord};
use serde::{Deserialize, Serialize};

use crate::{
    collection_manager::sides::write::index::GeoPoint, file_utils::create_if_not_exists, types::{DocumentId, GeoSearchFilter, GeoSearchPolygonFilter, NumberFilter}
};

#[derive(Debug)]
pub struct CommittedGeoPointField {
    field_path: Box<[String]>,
    tree: BKDTree<f32, DocumentId>,
    data_dir: PathBuf,
}

impl CommittedGeoPointField {
    pub fn from_iter<I>(field_path: Box<[String]>, iter: I, data_dir: PathBuf) -> Result<Self>
    where
        I: Iterator<Item = (GeoPoint, DocumentId)>,
    {
        let vec: Vec<_> = iter.collect();
        let s = Self {
            field_path,
            tree: BKDTree::new(),
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
    }

    #[allow(deprecated)]
    pub fn try_load(info: GeoPointFieldInfo) -> Result<Self> {
        /*
        let data_dir = info.data_dir;
        let vec = match std::fs::File::open(data_dir.join("number_vec.bin")) {
            Ok(file) => {
                bincode::deserialize_from::<_, Vec<(GeoPoint, DocumentId)>>(file)
                    .context("Failed to deserialize number_vec.bin")?
            }
            Err(_) => {
                
                let mut vec: Vec<_> = items.map(|item| (item.key, item.values)).collect();
                // ensure the order is by key. This should not be necessary, but we do it to ensure consistency.
                vec.sort_by_key(|(key, _)| key.0);
                vec
            }
        };

        let s = Self {
            field_path: info.field_path,
            vec,
            data_dir,
        };

        s.commit().context("Failed to commit number field")?;

        Ok(s)
        */
        panic!("aaa")
    }

    fn commit(&self) -> Result<()> {
        // Ensure the data directory exists
        create_if_not_exists(&self.data_dir).context("Failed to create data directory")?;

        let file_path = self.data_dir.join("number_vec.bin");
        let file = std::fs::File::create(&file_path).context("Failed to create number_vec.bin")?;
        bincode::serialize_into(file, &self.tree).context("Failed to serialize number vec")?;

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
        panic!();
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_geopoint: &GeoSearchFilter,
    ) -> Box<dyn Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        match filter_geopoint {
            GeoSearchFilter::Radius(filter) => {
                Box::new(self.tree.search_by_radius(
                    Coord::new(filter.coordinates.lat, filter.coordinates.lon),
                    filter.value.to_meter(filter.unit),
                    haversine_distance,
                    filter.inside,
                ).copied())
            },
            GeoSearchFilter::Polygon(filter) => {
                let iter = self.tree.search_by_polygon(
                    filter.coordinates.iter().map(|g| Coord::new(g.lat, g.lon)).collect(),
                    false,
                ).copied();
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
