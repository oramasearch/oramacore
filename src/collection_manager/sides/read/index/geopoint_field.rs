use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use oramacore_fields::geopoint::{
    GeoFilterOp, GeoPoint as FieldsGeoPoint, GeoPointStorage, GeoPolygon,
    IndexedValue as GeoPointIndexedValue, Threshold,
};
use oramacore_lib::bkd::BKDTree;
use oramacore_lib::fs::BufferedFile;
use serde::Serialize;
use tracing::info;

use crate::collection_manager::sides::write::index::GeoPoint as WriteSideGeoPoint;
use crate::types::{DocumentId, GeoSearchFilter};

use super::committed_field::GeoPointFieldInfo;

/// Maximum number of segments before partial merge during compaction.
const MAX_SEGMENTS: usize = 10;

/// Wrapper around `oramacore_fields::geopoint::GeoPointStorage` that provides
/// a unified geopoint field with built-in persistence.
/// Replaces the old UncommittedGeoPointFilterField + CommittedGeoPointField pair.
pub struct GeoPointFieldStorage {
    field_path: Box<[String]>,
    base_path: PathBuf,
    storage: GeoPointStorage,
}

impl std::fmt::Debug for GeoPointFieldStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeoPointFieldStorage")
            .field("field_path", &self.field_path)
            .field("base_path", &self.base_path)
            .finish()
    }
}

#[derive(Serialize, Debug)]
pub struct GeoPointFieldStorageStats {
    pub count: usize,
}

impl GeoPointFieldStorage {
    /// Creates a new GeoPointFieldStorage at the given path.
    pub fn new(field_path: Box<[String]>, base_path: PathBuf) -> Result<Self> {
        let storage = GeoPointStorage::new(base_path.clone(), Threshold::default(), MAX_SEGMENTS)
            .context("Failed to create GeoPointStorage")?;
        Ok(Self {
            field_path,
            base_path,
            storage,
        })
    }

    /// Loads an existing GeoPointFieldStorage from a GeoPointFieldInfo,
    /// migrating from old format if necessary.
    pub fn try_load(info: GeoPointFieldInfo) -> Result<Self> {
        let base_path = info.data_dir.clone();
        let old_file = base_path.join("geopoint_tree.bin");

        if old_file.exists() {
            // Migration path: read old bincode format (BKDTree<f32, DocumentId>)
            let old_tree: BKDTree<f32, DocumentId> = BufferedFile::open(&old_file)
                .context("Cannot open old geopoint_tree.bin")?
                .read_bincode_data()
                .context("Cannot deserialize old geopoint_tree.bin")?;

            let storage =
                GeoPointStorage::new(base_path.clone(), Threshold::default(), MAX_SEGMENTS)
                    .context("Failed to create GeoPointStorage for migration")?;

            // Insert all points from old BKDTree, converting f32 -> f64
            for point in old_tree.iter() {
                let lat = point.coords.lat as f64;
                let lon = point.coords.lon as f64;
                if let Ok(geo_point) = FieldsGeoPoint::new(lat, lon) {
                    storage.insert(GeoPointIndexedValue::Plain(geo_point), point.data.0);
                }
            }

            storage
                .compact(1)
                .context("Failed to compact migrated GeoPointStorage")?;
            storage.cleanup();
            std::fs::remove_file(&old_file)
                .context("Failed to remove old geopoint_tree.bin after migration")?;
            info!("Migrated geopoint field from geopoint_tree.bin to GeoPointStorage");

            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        } else {
            // New format or already migrated: GeoPointStorage loads its own CURRENT/versions/ structure
            let storage =
                GeoPointStorage::new(base_path.clone(), Threshold::default(), MAX_SEGMENTS)
                    .context("Failed to load GeoPointStorage")?;
            Ok(Self {
                field_path: info.field_path,
                base_path,
                storage,
            })
        }
    }

    pub fn field_path(&self) -> &[String] {
        &self.field_path
    }

    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Inserts a document with the given write-side geopoint value (f32 coords).
    /// Converts f32 lat/lon to f64 for oramacore_fields.
    pub fn insert(&self, doc_id: DocumentId, value: WriteSideGeoPoint) -> Result<()> {
        let geo_point = FieldsGeoPoint::new(value.lat as f64, value.lon as f64)
            .context("Invalid geopoint coordinates")?;
        self.storage
            .insert(GeoPointIndexedValue::Plain(geo_point), doc_id.0);
        Ok(())
    }

    /// Inserts a document using an already-extracted `GeoPointIndexedValue`.
    /// This accepts both `Plain` and `Array` variants.
    pub fn insert_indexed(&self, doc_id: DocumentId, value: &GeoPointIndexedValue) {
        self.storage.insert(value.clone(), doc_id.0);
    }

    /// Deletes a document from the geopoint index.
    pub fn delete(&self, doc_id: DocumentId) {
        self.storage.delete(doc_id.0);
    }

    /// Returns true if there are pending operations that need compaction.
    pub fn has_pending_ops(&self) -> bool {
        match self.storage.info() {
            Ok(info) => info.pending_inserts > 0 || info.pending_deletes > 0,
            // Before first compact (no CURRENT file), info() may fail.
            // In that case, we assume there are pending ops.
            Err(_) => true,
        }
    }

    /// Compacts the storage at the given version number.
    pub fn compact(&self, version: u64) -> Result<()> {
        self.storage
            .compact(version)
            .context("Failed to compact GeoPointStorage")
    }

    /// Cleans up old version directories.
    pub fn cleanup(&self) {
        self.storage.cleanup();
    }

    /// Returns stats about this geopoint field.
    pub fn stats(&self) -> GeoPointFieldStorageStats {
        let count = match self.storage.info() {
            Ok(info) => info.total_points + info.pending_inserts,
            Err(_) => 0,
        };
        GeoPointFieldStorageStats { count }
    }

    /// Returns the current version number.
    pub fn current_version_id(&self) -> u64 {
        self.storage.current_version_id()
    }

    // =========================================================================
    // Filter support
    // =========================================================================

    /// Converts a GeoSearchFilter into a GeoFilterOp and runs the query.
    /// Returns matching document IDs.
    pub fn filter(&self, filter_param: &GeoSearchFilter) -> Result<Vec<DocumentId>> {
        let op = geo_search_filter_to_op(filter_param)?;
        let filter_data = self.storage.filter(op);
        Ok(filter_data.iter().map(DocumentId).collect())
    }

    /// Returns metadata for DumpV1 serialization.
    pub fn metadata(&self) -> GeoPointFieldInfo {
        GeoPointFieldInfo {
            field_path: self.field_path.clone(),
            data_dir: self.base_path.clone(),
        }
    }
}

/// Converts an oramacore `GeoSearchFilter` to an `oramacore_fields` `GeoFilterOp`.
fn geo_search_filter_to_op(filter: &GeoSearchFilter) -> Result<GeoFilterOp> {
    match filter {
        GeoSearchFilter::Radius(r) => {
            let center = FieldsGeoPoint::new(r.coordinates.lat as f64, r.coordinates.lon as f64)
                .context("Invalid radius center coordinates")?;
            let radius_meters = r.value.to_meter(r.unit) as f64;
            if r.inside {
                Ok(GeoFilterOp::Radius {
                    center,
                    radius_meters,
                })
            } else {
                Ok(GeoFilterOp::OutsideRadius {
                    center,
                    radius_meters,
                })
            }
        }
        GeoSearchFilter::Polygon(p) => {
            let vertices: Result<Vec<FieldsGeoPoint>> = p
                .coordinates
                .iter()
                .map(|g| {
                    FieldsGeoPoint::new(g.lat as f64, g.lon as f64)
                        .context("Invalid polygon vertex coordinates")
                })
                .collect();
            let polygon = GeoPolygon::new(vertices?).context("Failed to create GeoPolygon")?;
            if p.inside {
                Ok(GeoFilterOp::Polygon { polygon })
            } else {
                Ok(GeoFilterOp::OutsidePolygon { polygon })
            }
        }
    }
}
