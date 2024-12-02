use std::sync::Arc;

use axum::Router;

use crate::collection_manager::sides::{read::CollectionsReader, write::CollectionsWriter};

mod admin;
mod search;

pub fn apis(
    writers: Option<Arc<CollectionsWriter>>,
    readers: Option<Arc<CollectionsReader>>,
) -> Router {
    let collection_router = Router::new();

    let collection_router = if let Some(writers) = writers {
        collection_router.nest("/", admin::apis(writers))
    } else {
        collection_router
    };

    if let Some(readers) = readers {
        collection_router.nest("/", search::apis(readers))
    } else {
        collection_router
    }
}
