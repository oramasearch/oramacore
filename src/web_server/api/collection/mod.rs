use std::sync::Arc;

use axum::Router;

use crate::collection_manager::sides::{document_storage::DocumentStorage, read::CollectionsReader, write::CollectionsWriter};

mod admin;
mod search;

pub fn apis(
    writers: Option<Arc<CollectionsWriter>>,
    readers: Option<Arc<CollectionsReader>>,
    doc: Option<Arc<dyn DocumentStorage>>,
) -> Router {
    let collection_router = Router::new();

    let collection_router = if let Some(writers) = writers {
        collection_router.nest("/", admin::apis(writers))
    } else {
        collection_router
    };

    if let Some(readers) = readers {
        let doc = doc.expect("Document storage is required for search");
        collection_router.nest("/", search::apis(readers, doc))
    } else {
        collection_router
    }
}
