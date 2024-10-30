use std::sync::Arc;

use anyhow::Context;
use collection_manager::{CollectionManager, CollectionsConfiguration};
use config::Config;
use rocksdb::OptimisticTransactionDB;
use serde::Deserialize;
use storage::Storage;
use web_server::{HttpConfig, WebServer};

#[derive(Debug, Deserialize, Clone)]
struct RustoramaConfig {
    data_dir: String,
    http: HttpConfig,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "./config.json".to_string());

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Json5))
        .add_source(config::Environment::with_prefix("RUSTORAMA"))
        .build()
        .context("Failed to load configuration")?;

    let config = settings
        .try_deserialize::<RustoramaConfig>()
        .context("Failed to deserialize configuration")?;

    let manager = create_manager(config.clone());
    let manager = Arc::new(manager);
    let web_server = WebServer::new(manager);

    web_server.start(config.http).await?;

    Ok(())
}

fn create_manager(config: RustoramaConfig) -> CollectionManager {
    let db = OptimisticTransactionDB::open_default(config.data_dir).unwrap();
    let storage = Arc::new(Storage::new(db));

    CollectionManager::new(CollectionsConfiguration { storage })
}
