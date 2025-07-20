use std::sync::Arc;

use axum::Router;

use crate::collection_manager::sides::{read::ReadSide, write::WriteSide};

mod actions;
mod admin;
mod answer;
mod generate;
mod hooks;
mod search;
mod system_prompts;
mod tools;

pub fn apis(write_side: Option<Arc<WriteSide>>, read_side: Option<Arc<ReadSide>>) -> Router {
    let collection_router = Router::new();

    let collection_router = if let Some(write_side) = write_side {
        collection_router
            .merge(hooks::write_apis(write_side.clone()))
            .merge(admin::apis(write_side.clone()))
            .merge(tools::write_apis(write_side.clone()))
            .merge(system_prompts::write_apis(write_side))
    } else {
        collection_router
    };

    if let Some(read_side) = read_side {
        collection_router
            .merge(search::apis(read_side.clone()))
            .merge(actions::apis(read_side.clone()))
            .merge(answer::apis(read_side.clone()))
            .merge(generate::apis(read_side.clone()))
            .merge(tools::read_apis(read_side.clone()))
            .merge(system_prompts::read_apis(read_side))
    } else {
        collection_router
    }
}
