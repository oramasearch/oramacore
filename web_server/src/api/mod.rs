use actix_web::{body::BoxBody, get, http::header::ContentType, web, HttpRequest, HttpResponse, Responder};
use collection::{get_collection_by_id, get_collections};
use serde::Serialize;

mod collection;


pub fn api_config(cfg: &mut web::ServiceConfig)
{
    cfg.service(
        web::scope("/collections")
        .service(get_collections)
        .service(get_collection_by_id)
    );
    cfg.service(index);
    cfg.service(health);
}

#[derive(Serialize)]
struct GenericMessage {
    message: &'static str,
}
impl Responder for GenericMessage {
    type Body = BoxBody;

    fn respond_to(self, _req: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        // Create response and set content type
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

static INDEX_MESSAGE: &str = "hi! welcome to Orama";
#[get("/")]
async fn index() -> impl Responder {
    GenericMessage {
        message: INDEX_MESSAGE,
    }
}

static HEALTH_MESSAGE: &str = "up";
#[get("/health")]
async fn health() -> impl Responder {
    GenericMessage {
        message: HEALTH_MESSAGE,
    }
}
