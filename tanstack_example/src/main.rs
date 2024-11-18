use std::{fs, sync::Arc};

use collection_manager::{
    dto::{CreateCollectionOptionDTO, Limit, SearchParams, TypedField},
    CollectionManager, CollectionsConfiguration,
};
use documentation::parse_documentation;
use example::parse_example;
use storage::Storage;

use types::CodeLanguage;
use web_server::{HttpConfig, WebServer};

mod documentation;
mod example;
mod fs_utils;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let storage_dir = "./tanstack";
    let _ = fs::remove_dir_all(storage_dir);

    let storage = Arc::new(Storage::from_path(storage_dir));

    let manager = CollectionManager::new(CollectionsConfiguration { storage });

    let collection_id = manager
        .create_collection(CreateCollectionOptionDTO {
            id: "tanstack".to_string(),
            description: None,
            language: None,
            typed_fields: vec![("code".to_string(), TypedField::Code(CodeLanguage::TSX))]
                .into_iter()
                .collect(),
        })
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

    manager.get(collection_id.clone(), |collection| {
        collection.insert_batch(orama_documents.try_into().unwrap())
    });

    let output = manager.get(collection_id, |collection| {
        collection.search(SearchParams {
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
        })
    });

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
