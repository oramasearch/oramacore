use actix_web::{get, patch, post, web, HttpResponse, Responder};
use collection_manager::{
    dto::{CreateCollectionOptionDTO, SearchParams},
    CollectionManager,
};
use serde_json::json;
use types::{CollectionId, DocumentList};

#[get("/")]
async fn get_collections(manager: web::Data<CollectionManager>) -> impl Responder {
    let collections = manager.list();
    HttpResponse::Ok().json(collections)
}
#[get("/{collection_id}")]
async fn get_collection_by_id(
    path: web::Path<String>,
    manager: web::Data<CollectionManager>,
) -> impl Responder {
    let collection_id = CollectionId(path.into_inner());
    let output = manager.get(collection_id, |collection| collection.as_dto());

    match output {
        Some(data) => HttpResponse::Ok().json(data),
        None => HttpResponse::NotFound().finish(),
    }
}
#[post("/")]
async fn create_collection(
    manager: web::Data<CollectionManager>,
    json: web::Json<CreateCollectionOptionDTO>,
) -> impl Responder {
    let collection_id = match manager.create_collection(json.into_inner()) {
        Ok(collection_id) => collection_id,
        Err(e) => {
            return HttpResponse::InternalServerError().json(json!({ "error": e.to_string() }))
        }
    };
    HttpResponse::Created().json(json!({
        "collection_id": collection_id,
    }))
}
#[patch("/{collection_id}/documents")]
async fn add_documents(
    path: web::Path<String>,
    manager: web::Data<CollectionManager>,
    json: web::Json<DocumentList>,
) -> impl Responder {
    let collection_id = CollectionId(path.into_inner());
    let output = manager.get(collection_id, |collection| {
        collection.insert_batch(json.into_inner())
    });

    match output {
        Some(Ok(())) => HttpResponse::Ok().finish(),
        Some(Err(e)) => HttpResponse::InternalServerError().json(json!({ "error": e.to_string() })),
        None => HttpResponse::NotFound().finish(),
    }
}
#[post("/{collection_id}/search")]
async fn search(
    path: web::Path<String>,
    manager: web::Data<CollectionManager>,
    json: web::Json<SearchParams>,
) -> impl Responder {
    let collection_id = CollectionId(path.into_inner());
    let output = manager.get(collection_id, |collection| {
        collection.search(json.into_inner())
    });

    match output {
        Some(Ok(data)) => HttpResponse::Ok().json(data),
        Some(Err(e)) => HttpResponse::InternalServerError().json(json!({ "error": e.to_string() })),
        None => HttpResponse::NotFound().finish(),
    }
}
