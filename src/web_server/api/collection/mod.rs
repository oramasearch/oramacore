use std::sync::Arc;

use axum::Router;

use crate::collection_manager::CollectionManager;

mod admin;
mod search;

pub fn apis() -> Router<Arc<CollectionManager>> {
    let collection_router = Router::<Arc<CollectionManager>>::new();

    collection_router
        .nest("/", admin::apis())
        .nest("/", search::apis())
}
