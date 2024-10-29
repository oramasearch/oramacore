use actix_web::{App, HttpServer};
use anyhow::Result;
use api::api_config;

mod api;

pub struct WebServer {

}

impl WebServer {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn start(self) -> Result<()> {
        let output = HttpServer::new(|| {
            App::new()
                .configure(api_config)
        })
        .bind(("127.0.0.1", 8080))?
        .run()
        .await;

        match output {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into())
        }
    }
}
