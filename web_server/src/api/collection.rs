use actix_web::{get, web, Responder};


#[get("/")]
async fn get_collections() -> impl Responder {
    "get collections"
}
#[get("/{collection_id}")]
async fn get_collection_by_id(path: web::Path<String>) -> impl Responder {
    format!("get collection {}", path)
}
