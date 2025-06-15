use std::sync::Arc;

use axum::Router;

use crate::collection_manager::sides::{read::ReadSide, write::WriteSide};

mod actions;
mod admin;
mod answer;
mod hooks;
mod mpc;
mod search;
mod segments;
mod system_prompts;
mod tools;
mod triggers;

pub fn apis(write_side: Option<Arc<WriteSide>>, read_side: Option<Arc<ReadSide>>) -> Router {
    let collection_router = Router::new();

    let collection_router = if let Some(write_side) = write_side {
        collection_router
            .merge(hooks::apis(write_side.clone()))
            .merge(admin::apis(write_side.clone()))
            .merge(segments::write_apis(write_side.clone()))
            .merge(triggers::write_apis(write_side.clone()))
            .merge(tools::write_apis(write_side.clone()))
            .merge(mpc::write_apis(write_side.clone()))
            .merge(system_prompts::write_apis(write_side))
    } else {
        collection_router
    };

    if let Some(read_side) = read_side {
        collection_router
            .merge(search::apis(read_side.clone()))
            .merge(actions::apis(read_side.clone()))
            .merge(answer::apis(read_side.clone()))
            .merge(segments::read_apis(read_side.clone()))
            .merge(triggers::read_apis(read_side.clone()))
            .merge(tools::read_apis(read_side.clone()))
            .merge(mpc::read_apis(read_side.clone()))
            .merge(system_prompts::read_apis(read_side))
    } else {
        collection_router
    }
}
