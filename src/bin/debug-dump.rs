use std::{path::PathBuf, sync::Arc};

use anyhow::Context;
use clap::{Parser, Subcommand};
use config::Config;
use oramacore::{
    ai::{llms::LLMService, AIService},
    collection_manager::sides::read::{
        document_storage::{DocumentStorage, DocumentStorageConfig},
        Index,
    },
    nlp::NLPService,
    types::{DocumentId, IndexId},
    OramacoreConfig,
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    orama_config: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    OpenReadIndex {
        collection_id: String,
        index_id: String,

        #[command(subcommand)]
        command: OpenReadIndex,
    },
    OpenReadDocumentStorage {
        #[command(subcommand)]
        command: OpenReadDocumentStorage,
    },
}

#[derive(Subcommand)]
enum OpenReadDocumentStorage {
    ListDocumentIds,
    GetDocuments { document_ids: Vec<u64> },
    ZeboDebug,
}

#[derive(Subcommand)]
enum OpenReadIndex {
    ListDocument,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let config = load_config(cli.orama_config).context("Failed to load Oramacore configuration")?;

    match cli.command {
        Commands::OpenReadIndex {
            collection_id,
            index_id,
            command,
        } => {
            let path = config
                .reader_side
                .config
                .data_dir
                .join("collections")
                .join(&collection_id)
                .join("indexes")
                .join(&index_id);
            let index_id = IndexId::try_new(index_id)
                .map_err(|e| anyhow::anyhow!("Failed to create IndexId: {}", e))?;

            let ai_service = AIService::new(config.ai_server.clone());
            let ai_service = Arc::new(ai_service);

            let llm_service = match LLMService::try_new(
                config.ai_server.llm,
                config.ai_server.remote_llms.clone(),
            ) {
                Ok(service) => Arc::new(service),
                Err(err) => {
                    anyhow::bail!(
                        "Failed to create LLMService: {}. Please check your configuration.",
                        err
                    );
                }
            };

            let nlp_service = Arc::new(NLPService::new());

            let index = Index::try_load(index_id, path, nlp_service, llm_service, ai_service)
                .context("Failed to load index")?;

            match command {
                OpenReadIndex::ListDocument => {
                    let document_storage = DocumentStorage::try_new(DocumentStorageConfig {
                        data_dir: config.reader_side.config.data_dir.join("docs"),
                    })
                    .await
                    .context("Cannot create document storage")?;

                    let document_ids = index
                        .get_all_document_ids()
                        .await
                        .context("Failed to list document IDs")?;

                    println!("Found {} documents:", document_ids.len());

                    let output = document_storage
                        .get_documents_by_ids(document_ids.clone())
                        .await
                        .context("Failed to get documents by IDs")?;
                    for (d, doc_id) in output.into_iter().zip(document_ids) {
                        match d {
                            None => {
                                println!("{:?}: not found", doc_id);
                            }
                            Some(doc) => {
                                println!("{:?}: {:?} -> {}", doc_id, doc.id, doc.inner.get());
                            }
                        }
                    }
                }
            }
        }

        Commands::OpenReadDocumentStorage { command } => {
            let document_storage = DocumentStorage::try_new(DocumentStorageConfig {
                data_dir: config.reader_side.config.data_dir.join("docs"),
            })
            .await
            .context("Cannot create document storage")?;

            match command {
                OpenReadDocumentStorage::ListDocumentIds => {
                    let zebo_info = document_storage
                        .get_zebo_info()
                        .await
                        .context("Failed to get Zebo info")?;
                    let document_ids = zebo_info
                        .page_headers
                        .into_iter()
                        .flat_map(|h| h.index.into_iter().map(|x| DocumentId(x.0)))
                        .collect::<Vec<_>>();

                    println!("Found {} documents", document_ids.len());

                    for doc_id in document_ids {
                        println!("{:?}", doc_id);
                    }
                }
                OpenReadDocumentStorage::GetDocuments { document_ids } => {
                    let document_ids = document_ids.into_iter().map(DocumentId).collect::<Vec<_>>();
                    let output = document_storage
                        .get_documents_by_ids(document_ids.clone())
                        .await
                        .context("Failed to get documents by IDs")?;

                    for (d, doc_id) in output.into_iter().zip(document_ids) {
                        match d {
                            None => {
                                println!("{:?}: not found", doc_id);
                            }
                            Some(doc) => {
                                println!("{:?}: {:?} -> {}", doc_id, doc.id, doc.inner.get());
                            }
                        }
                    }
                }
                OpenReadDocumentStorage::ZeboDebug => {
                    let zebo_info = document_storage
                        .get_zebo_info()
                        .await
                        .context("Failed to get Zebo info")?;

                    println!("Zebo Info: {:#?}", zebo_info);
                }
            }
        }
    }

    Ok(())
}

fn load_config(config_path: PathBuf) -> anyhow::Result<OramacoreConfig> {
    let config_path = std::fs::canonicalize(&config_path)?;
    let config_path: String = config_path.to_string_lossy().into();

    let settings = Config::builder()
        .add_source(config::File::with_name(&config_path).format(config::FileFormat::Yaml))
        .add_source(config::Environment::with_prefix("ORAMACORE"))
        .build()
        .context("Failed to load configuration")?;

    let oramacore_config = settings
        .try_deserialize::<OramacoreConfig>()
        .context("Failed to deserialize configuration")?;

    Ok(oramacore_config)
}
