use std::sync::Arc;

use axum::Router;

use crate::collection_manager::sides::{ReadSide, WriteSide};

mod actions;
mod admin;
mod answer;
mod hooks;
mod search;

pub fn apis(write_side: Option<Arc<WriteSide>>, read_side: Option<Arc<ReadSide>>) -> Router {
    let collection_router = Router::new();

    let collection_router = if let Some(write_side) = write_side {
        collection_router
            .merge(hooks::apis(write_side.clone()))
            .merge(admin::apis(write_side))
    } else {
        collection_router
    };

    if let Some(read_side) = read_side {
        collection_router
            .merge(search::apis(read_side.clone()))
            .merge(answer::apis(read_side))
    } else {
        collection_router
    }
}
