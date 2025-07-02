use bkd::{haversine_distance, BKDTree, Coord, Point};
use serde::Serialize;

use crate::{
    collection_manager::sides::write::index::GeoPoint,
    types::{DocumentId, GeoSearchFilter},
};

#[derive(Debug)]
pub struct UncommittedGeoPointFilterField {
    field_path: Box<[String]>,
    count: usize,
    inner: BKDTree<f32, DocumentId>,
}

impl UncommittedGeoPointFilterField {
    pub fn empty(field_path: Box<[String]>) -> Self {
        Self {
            field_path,
            count: 0,
            inner: BKDTree::new(),
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.inner = BKDTree::new();
        self.count = 0;
    }

    pub fn insert(&mut self, doc_id: DocumentId, value: GeoPoint) {
        self.count += 1;
        self.inner
            .insert(Point::new(Coord::new(value.lat, value.lon), doc_id));
    }

    pub fn inner(&self) -> BKDTree<f32, DocumentId> {
        self.inner.clone()
    }

    pub fn iter<'s>(&'s self) -> impl Iterator<Item = &'s Point<f32, DocumentId>> + 's {
        self.inner.iter()
    }

    pub fn filter<'s, 'iter>(
        &'s self,
        filter_geopoint: &GeoSearchFilter,
    ) -> Box<dyn Iterator<Item = DocumentId> + 'iter>
    where
        's: 'iter,
    {
        self.inner.display(0);

        match filter_geopoint {
            GeoSearchFilter::Radius(filter) => Box::new(
                self.inner
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
                    .inner
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

    pub fn stats(&self) -> UncommittedGeoPointFieldStats {
        UncommittedGeoPointFieldStats { count: self.len() }
    }
}

#[derive(Serialize, Debug)]
pub struct UncommittedGeoPointFieldStats {
    pub count: usize,
}
