use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::State,
    response::IntoResponse,
    Json, Router,
};
use axum_openapi3::{utoipa::{PartialSchema, ToSchema}, *};
use hook_storage::HookType;
use serde::Deserialize;
use serde_json::json;
use axum_openapi3::utoipa;

use crate::{
    collection_manager::sides::write::{WriteError, WriteSide},
    types::{CollectionId, WriteApiKey},
};

pub fn write_apis(write_side: Arc<WriteSide>) -> Router {
    Router::new()
        .add(set_hook_v0())
        .add(delete_hook_v0())
        .add(list_hook_v0())
        .with_state(write_side)
}

#[derive(Deserialize, ToSchema)]
pub struct NewHookPostParams {
    name: HookTypeWrapper,
    code: String,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/hooks/set",
    description = "Add a new JavaScript hook"
)]
async fn set_hook_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<NewHookPostParams>,
) -> Result<impl IntoResponse, WriteError> {
    let NewHookPostParams { name, code } = params;

    let hook = write_side.get_hooks_storage(write_api_key, collection_id).await?;
    hook.insert_hook(name.0, code).await?;

    Ok(Json(json!({ "success": true })))
}

#[derive(Deserialize, ToSchema)]
struct DeleteHookPostParams {
    name_to_delete: HookTypeWrapper,
}

#[endpoint(
    method = "POST",
    path = "/v1/collections/{collection_id}/hooks/delete",
    description = "Delete new JavaScript hook"
)]
async fn delete_hook_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
    Json(params): Json<DeleteHookPostParams>,
) -> Result<impl IntoResponse, WriteError> {
    let DeleteHookPostParams { name_to_delete } = params;

    let hook = write_side.get_hooks_storage(write_api_key, collection_id).await?;
    hook.delete_hook(name_to_delete.0).await?;

    Ok(Json(json!({ "success": true })))
}

#[endpoint(
    method = "GET",
    path = "/v1/collections/{collection_id}/hooks/list",
    description = "List new JavaScript hooks"
)]
async fn list_hook_v0(
    collection_id: CollectionId,
    write_side: State<Arc<WriteSide>>,
    write_api_key: WriteApiKey,
) -> Result<impl IntoResponse, WriteError> {
    let hook = write_side.get_hooks_storage(write_api_key, collection_id).await?;
    let hooks = hook.list_hooks()?;

    let output: HashMap<_, _> = hooks.into_iter()
        .collect();

    Ok(Json(json!({ "hooks": output })))
}


#[derive(Deserialize)]
struct HookTypeWrapper(HookType);

impl PartialSchema for HookTypeWrapper {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        <String as PartialSchema>::schema()
    }
}
impl ToSchema for HookTypeWrapper {}