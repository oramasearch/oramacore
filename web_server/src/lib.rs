use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use actix_web::{
    body::MessageBody,
    dev::{ServiceFactory, ServiceRequest, ServiceResponse},
    web::Data,
    App, HttpServer,
};
use anyhow::Result;
use api::api_config;
use collection_manager::CollectionManager;
use serde::Deserialize;

mod api;

#[derive(Debug, Deserialize, Clone)]
pub struct HttpConfig {
    pub host: IpAddr,
    pub port: u16,
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
        let s = Arc::new(self);
        let server = HttpServer::new(move || s.get_service()).bind(addr)?;

        for addr in server.addrs() {
            println!("Started at http://{:?}", addr);
        }

        let output = server.run().await;

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    fn get_service(
        &self,
    ) -> App<
        impl ServiceFactory<
            ServiceRequest,
            Response = ServiceResponse<impl MessageBody>,
            Config = (),
            InitError = (),
            Error = actix_web::error::Error,
        >,
    > {
        let data: Data<_> = self.collection_manager.clone().into();
        App::new().app_data(data.clone()).configure(api_config)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use collection_manager::{
        dto::{CreateCollectionOptionDTO, Limit, SearchParams},
        CollectionManager, CollectionsConfiguration,
    };
    use rocksdb::OptimisticTransactionDB;
    use serde_json::json;
    use storage::Storage;
    use tempdir::TempDir;

    use actix_web::{
        body::MessageBody,
        dev::{Service, ServiceResponse},
        http::header::ContentType,
        test,
    };
    use types::SearchResult;

    #[actix_web::test]
    async fn test_index_get() {
        let web_server = super::WebServer::new(Arc::new(create_manager()));

        let app = test::init_service(web_server.get_service()).await;

        let collection_id = "test".to_string();

        let resp = create_collection(
            &app,
            &CreateCollectionOptionDTO {
                id: collection_id.clone(),
                description: None,
                language: None,
                typed_fields: Default::default(),
            },
        )
        .await;

        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), 201);

        let resp = list_collections(&app).await;

        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0].id, collection_id);
        assert_eq!(resp[0].document_count, 0);

        let resp = add_documents(&app, &collection_id, &vec![
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
        assert!(resp.status().is_success());
        assert_eq!(resp.status().as_u16(), 200);

        let resp = list_collections(&app).await;
        assert_eq!(resp.len(), 1);
        assert_eq!(resp[0].id, collection_id);
        assert_eq!(resp[0].document_count, 2);

        let resp = search(
            &app,
            &collection_id,
            &SearchParams {
                term: "beatles".to_string(),
                limit: Limit(10),
            },
        )
        .await;
        println!("{:#?}", resp);
        assert_eq!(resp.count, 1);
        assert_eq!(resp.hits.len(), 1);
        assert_eq!(resp.hits[0].id, "1");
    }

    async fn create_collection<
        App: Service<
            actix_http::Request,
            Response = ServiceResponse<impl MessageBody>,
            Error = actix_web::error::Error,
        >,
    >(
        app: &App,
        create_collection_dto: &CreateCollectionOptionDTO,
    ) -> ServiceResponse<impl MessageBody> {
        let req = test::TestRequest::post()
            .insert_header(ContentType::json())
            .uri("/v0/collections/")
            .set_payload(serde_json::to_string(create_collection_dto).unwrap())
            .to_request();
        test::call_service(&app, req).await
    }

    async fn list_collections<
        App: Service<
            actix_http::Request,
            Response = ServiceResponse<impl MessageBody>,
            Error = actix_web::error::Error,
        >,
    >(
        app: &App,
    ) -> Vec<collection_manager::dto::CollectionDTO> {
        let req = test::TestRequest::get()
            .insert_header(ContentType::json())
            .uri("/v0/collections/")
            .to_request();
        let resp: Vec<collection_manager::dto::CollectionDTO> =
            test::call_and_read_body_json(&app, req).await;

        resp
    }

    async fn add_documents<
        App: Service<
            actix_http::Request,
            Response = ServiceResponse<impl MessageBody>,
            Error = actix_web::error::Error,
        >,
    >(
        app: &App,
        collection_id: &str,
        docs: &Vec<serde_json::Value>,
    ) -> ServiceResponse<impl MessageBody> {
        let req = test::TestRequest::patch()
            .insert_header(ContentType::json())
            .uri(&format!("/v0/collections/{}/documents", collection_id))
            .set_payload(serde_json::to_string(docs).unwrap())
            .to_request();
        test::call_service(&app, req).await
    }

    async fn search<
        App: Service<
            actix_http::Request,
            Response = ServiceResponse<impl MessageBody>,
            Error = actix_web::error::Error,
        >,
    >(
        app: &App,
        collection_id: &str,
        search_params: &SearchParams,
    ) -> SearchResult {
        let req = test::TestRequest::post()
            .insert_header(ContentType::json())
            .uri(&format!("/v0/collections/{}/search", collection_id))
            .set_payload(serde_json::to_string(search_params).unwrap())
            .to_request();
        let resp: SearchResult = test::call_and_read_body_json(&app, req).await;
        resp
    }

    fn create_manager() -> CollectionManager {
        let tmp_dir = TempDir::new("string_index_test").unwrap();
        let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
        let db = OptimisticTransactionDB::open_default(tmp_dir).unwrap();
        let storage = Arc::new(Storage::new(db));

        CollectionManager::new(CollectionsConfiguration { storage })
    }
}
