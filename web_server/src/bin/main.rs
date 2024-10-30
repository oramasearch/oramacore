use std::{net::IpAddr, str::FromStr, sync::Arc};

use collection_manager::{CollectionManager, CollectionsConfiguration};
use rocksdb::OptimisticTransactionDB;
use storage::Storage;
use tempdir::TempDir;
use web_server::{HttpConfig, WebServer};

fn main() -> std::io::Result<()> {
    let err = ::actix_web::rt::System::new().block_on(async move {
        let manager = create_manager();
        let manager = Arc::new(manager);
        let web_server = WebServer::new(manager);

        web_server.start(HttpConfig {
            host: IpAddr::from_str("127.0.0.1").unwrap(),
            port: 8080,
        }).await
    });

    println!("{:?}", err);

    Ok(())
}

fn create_manager() -> CollectionManager {
    let tmp_dir = TempDir::new("string_index_test").unwrap();
    let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
    let db = OptimisticTransactionDB::open_default(tmp_dir).unwrap();
    let storage = Arc::new(Storage::new(db));

    CollectionManager::new(CollectionsConfiguration { storage })
}
