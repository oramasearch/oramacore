use std::sync::Arc;

use axum::Router;

use crate::collection_manager::sides::{read::ReadSide, write::WriteSide};

mod actions;
mod admin;
mod analytics;
mod answer;
mod generate;
mod hooks;
mod mcp;
mod pin_rules;
mod search;
mod system_prompts;
mod tools;
mod training_sets;

pub fn apis(write_side: Option<Arc<WriteSide>>, read_side: Option<Arc<ReadSide>>) -> Router {
    let collection_router = Router::new();

    let collection_router = if let Some(write_side) = write_side {
        collection_router
            .merge(hooks::write_apis(write_side.clone()))
            .merge(admin::apis(write_side.clone()))
            .merge(tools::write_apis(write_side.clone()))
            .merge(system_prompts::write_apis(write_side.clone()))
            .merge(training_sets::write_apis(write_side.clone()))
            .merge(pin_rules::write_apis(write_side.clone()))
            .merge(mcp::write_apis(write_side.clone()))
    } else {
        collection_router
    };

    if let Some(read_side) = read_side {
        collection_router
            .merge(analytics::read_apis(read_side.clone()))
            .merge(search::apis(read_side.clone()))
            .merge(actions::apis(read_side.clone()))
            .merge(answer::apis(read_side.clone()))
            .merge(generate::apis(read_side.clone()))
            .merge(tools::read_apis(read_side.clone()))
            .merge(system_prompts::read_apis(read_side.clone()))
            .merge(training_sets::read_apis(read_side.clone()))
            .merge(mcp::apis(read_side.clone()))
            .merge(pin_rules::read_apis(read_side.clone()))
    } else {
        collection_router
    }
}
