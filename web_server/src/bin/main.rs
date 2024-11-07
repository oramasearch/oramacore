use std::{net::IpAddr, str::FromStr, sync::Arc};

use anyhow::Result;
use collection_manager::{CollectionManager, CollectionsConfiguration};
use storage::Storage;
use tempdir::TempDir;
use web_server::{HttpConfig, WebServer};

#[tokio::main]
async fn main() -> Result<()> {
    let manager = create_manager();
    let manager = Arc::new(manager);
    let web_server = WebServer::new(manager);

    web_server
        .start(HttpConfig {
            host: IpAddr::from_str("127.0.0.1").unwrap(),
            port: 8080,
            allow_cors: true,
        })
        .await?;

    Ok(())
}

fn create_manager() -> CollectionManager {
    let tmp_dir = TempDir::new("string_index_test").unwrap();
    let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
    let storage = Arc::new(Storage::from_path(&tmp_dir));

    CollectionManager::new(CollectionsConfiguration { storage })
}
