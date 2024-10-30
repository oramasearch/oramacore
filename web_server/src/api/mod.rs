use actix_web::{get, web, HttpResponse, Responder};
use collection::{add_documents, create_collection, get_collection_by_id, get_collections, search};

mod collection;

pub fn api_config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/collections")
            .service(get_collections)
            .service(get_collection_by_id)
            .service(create_collection)
            .service(add_documents)
            .service(search),
    );
    cfg.service(index);
    cfg.service(health);
}

static INDEX_MESSAGE: &str = "hi! welcome to Orama";
#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok().body(INDEX_MESSAGE)
}

static HEALTH_MESSAGE: &str = "up";
#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body(HEALTH_MESSAGE)
}
