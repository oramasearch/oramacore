use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use anyhow::Result;
use collection_manager::CollectionManager;
use serde::Deserialize;
use tower_http::cors::CorsLayer;

mod api;

#[derive(Debug, Deserialize, Clone)]
pub struct HttpConfig {
    pub host: IpAddr,
    pub port: u16,
    pub allow_cors: bool,
}

pub struct WebServer {
    collection_manager: Arc<CollectionManager>,
}

impl WebServer {
    pub fn new(collection_manager: Arc<CollectionManager>) -> Self {
        Self { collection_manager }
    }

    pub async fn start(self, config: HttpConfig) -> Result<()> {
        let addr = SocketAddr::new(config.host, config.port);

        let router = api::api_config().with_state(self.collection_manager.clone());
        let router = if config.allow_cors {
            let cors_layer = CorsLayer::new()
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
                .allow_origin(tower_http::cors::Any);

            router.layer(cors_layer)
        } else {
            router
        };

        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

        println!("Started at http://{:?}", listener.local_addr().unwrap());

        let output = axum::serve(listener, router).await;

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{self, Request, StatusCode},
        Router,
    };
    use collection_manager::{
        dto::{CreateCollectionOptionDTO, Limit, SearchParams},
        CollectionManager, CollectionsConfiguration,
    };
    use serde_json::{json, Value};
    use storage::Storage;
    use tempdir::TempDir;

    use types::SearchResult;

    use crate::api;

    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use tower_http::trace::TraceLayer;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    #[tokio::test]
    async fn test_index_get() {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                    format!("{}=debug,tower_http=debug", env!("CARGO_CRATE_NAME")).into()
                }),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();

        let mut router = api::api_config()
            .with_state(Arc::new(create_manager()))
            .layer(TraceLayer::new_for_http());

        let collection_id = "test".to_string();

        let (status_code, output) = create_collection(
            &mut router,
            &CreateCollectionOptionDTO {
                id: collection_id.clone(),
                description: None,
                language: None,
                typed_fields: Default::default(),
            },
        )
        .await;

        assert_eq!(status_code, 201);
        assert_eq!(output, json!({ "collection_id": collection_id }));

        let (status_code, value) = list_collections(&mut router).await;

        assert_eq!(status_code, 200);
        let resp: Vec<collection_manager::dto::CollectionDTO> =
            serde_json::from_value(value).unwrap();
        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0].id, collection_id);
        assert_eq!(resp[0].document_count, 0);

        let (status_code, value) = add_documents(&mut router, &collection_id, &vec![
            json!({
                "id": "1",
                "title": "The Beatles",
                "content": "The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time.",
            }),
            json!({
                "id": "2",
                "title": "The Rolling Stones",
                "content": "The Rolling Stones are an English rock band formed in London in 1962. The first settled line-up consisted of Brian Jones, Ian Stewart, Mick Jagger, Keith Richards, Bill Wyman, and Charlie Watts.",
            }),
        ]).await;
        assert_eq!(status_code, 200);
        println!("{:#?}", value);

        let (status_code, value) = list_collections(&mut router).await;
        assert_eq!(status_code, 200);
        let resp: Vec<collection_manager::dto::CollectionDTO> =
            serde_json::from_value(value).unwrap();
        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0].id, collection_id);
        assert_eq!(resp[0].document_count, 2);

        let (status_code, value) = search(
            &mut router,
            &collection_id,
            &SearchParams {
                term: "beatles".to_string(),
                limit: Limit(10),
                boost: Default::default(),
                properties: Default::default(),
            },
        )
        .await;

        assert_eq!(status_code, 200);
        let resp: SearchResult = serde_json::from_value(value).unwrap();

        assert_eq!(resp.count, 1);
        assert_eq!(resp.hits.len(), 1);
        assert_eq!(resp.hits[0].id, "1");
    }

    async fn create_collection(
        router: &mut Router,
        create_collection_dto: &CreateCollectionOptionDTO,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .uri("/v0/collections")
            .method(http::Method::POST)
            .header(http::header::CONTENT_TYPE, mime::APPLICATION_JSON.as_ref())
            .body(Body::from(
                serde_json::to_string(create_collection_dto).unwrap(),
            ))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();

        let status_code = resp.status();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let output = serde_json::from_slice::<serde_json::Value>(&body).unwrap();

        (status_code, output)
    }

    async fn list_collections(router: &mut Router) -> (StatusCode, Value) {
        let req = Request::builder()
            .uri("/v0/collections")
            .method(http::Method::GET)
            .body(Body::empty())
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();

        let status_code = resp.status();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let output = serde_json::from_slice::<serde_json::Value>(&body).unwrap();

        (status_code, output)
    }

    async fn add_documents(
        router: &mut Router,
        collection_id: &str,
        docs: &Vec<serde_json::Value>,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .uri(&format!("/v0/collections/{collection_id}/documents"))
            .header(http::header::CONTENT_TYPE, mime::APPLICATION_JSON.as_ref())
            .method(http::Method::PATCH)
            .body(Body::from(serde_json::to_string(docs).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();

        let status_code = resp.status();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let output = serde_json::from_slice::<serde_json::Value>(&body).unwrap();

        (status_code, output)
    }

    async fn search(
        router: &mut Router,
        collection_id: &str,
        search_params: &SearchParams,
    ) -> (StatusCode, Value) {
        let req = Request::builder()
            .uri(&format!("/v0/collections/{collection_id}/search"))
            .header(http::header::CONTENT_TYPE, mime::APPLICATION_JSON.as_ref())
            .method(http::Method::POST)
            .body(Body::from(serde_json::to_string(search_params).unwrap()))
            .unwrap();

        let resp = router.oneshot(req).await.unwrap();

        let status_code = resp.status();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let output = serde_json::from_slice::<serde_json::Value>(&body).unwrap();

        (status_code, output)
    }

    fn create_manager() -> CollectionManager {
        let tmp_dir = TempDir::new("string_index_test").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
        let storage = Arc::new(Storage::from_path(&tmp_dir));

        CollectionManager::new(CollectionsConfiguration { storage })
    }
}
