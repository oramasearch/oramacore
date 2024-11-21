use std::{fs, sync::Arc};

use anyhow::Context;
use documentation::parse_documentation;
use example::parse_example;
use rustorama::{
    collection_manager::{
        dto::{CreateCollectionOptionDTO, Limit, SearchParams, TypedField},
        CollectionManager, CollectionsConfiguration,
    }, embeddings::{EmbeddingConfig, EmbeddingService}, types::CodeLanguage, web_server::{HttpConfig, WebServer}
};

mod documentation;
mod example;
mod fs_utils;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage_dir = "./tanstack";
    let _ = fs::remove_dir_all(storage_dir);

    let embedding_service = EmbeddingService::try_new(EmbeddingConfig {
        cache_path: std::env::temp_dir().to_str().unwrap().to_string(),
        hugging_face: None,
        preload_all: false,
    })
        .with_context(|| "Failed to initialize the EmbeddingService")?;
    let embedding_service = Arc::new(embedding_service);
    let manager = CollectionManager::new(CollectionsConfiguration {
        embedding_service,
    });

    let collection_id = manager
        .create_collection(CreateCollectionOptionDTO {
            id: "tanstack".to_string(),
            description: None,
            language: None,
            typed_fields: vec![("code".to_string(), TypedField::Code(CodeLanguage::TSX))]
                .into_iter()
                .collect(),
        })
        .await
        .expect("unable to create collection");

    let orama_documentation_documents =
        parse_documentation("/Users/allevo/repos/rustorama/tanstack_example/tanstack_table/docs");
    let orama_example_documents = parse_example(
        "/Users/allevo/repos/rustorama/tanstack_example/tanstack_table/examples/react",
    )
    .await;

    let orama_documents = orama_documentation_documents
        .into_iter()
        .chain(orama_example_documents)
        .collect::<Vec<_>>();

    let collection = manager.get(collection_id).await.unwrap();
    collection
        .insert_batch(orama_documents.try_into().unwrap())
        .await
        .unwrap();

    collection
        .search(SearchParams {
            term: r###"columnHelper.accessor('firstName')

// OR

{
  accessorKey: 'firstName',
}"###
                .to_string(),
            limit: Limit(3),
            boost: Default::default(),
            properties: Some(vec!["code".to_string()]),
            where_filter: Default::default(),
            facets: Default::default(),
        })
        .await
        .unwrap();

    drop(collection);

    let web_server = WebServer::new(Arc::new(manager));

    web_server
        .start(HttpConfig {
            port: 8080,
            host: "127.0.0.1".parse().unwrap(),
            allow_cors: true,
        })
        .await?;

    Ok(())
}
