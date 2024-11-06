use std::{
    collections::HashMap, fs, sync::Arc
};

use code_parser::CodeParser;
use collection_manager::{
    dto::{CreateCollectionOptionDTO, Limit, SearchParams, TypedField},
    CollectionManager, CollectionsConfiguration,
};
#[allow(unused_imports)]
use documentation::parse_documentation;
use example::parse_example;
use fs_utils::get_files;
use itertools::Itertools;
use ptrie::Trie;
use serde_json::Value;
use storage::Storage;

use types::{CodeLanguage, DocumentId};

mod documentation;
mod example;
mod fs_utils;

fn main() -> anyhow::Result<()> {
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

    let orama_documentation_documents = parse_documentation("/Users/allevo/repos/rustorama/tanstack_example/tanstack_table/docs");
    let orama_example_documents = parse_example("/Users/allevo/repos/rustorama/tanstack_example/tanstack_table/examples/react");

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
        })
    });

    println!("{:#?}", output);

    /*
    println!("orama_documents: {:#?}", orama_documents.len());

    #[derive(Hash, PartialEq, Eq, Debug, Clone)]
    struct Position(usize);

    /// Token <-> DocumentId
    #[derive(Debug, Clone)]
    struct CodePosting {
        token: String,
        document_id: DocumentId,
        positions: HashMap<Position, Vec<TokenType>>,
    }

    #[derive(Debug, Clone)]
    enum TokenType {
        Token,
        ComposedSubwords(u8),
        Subword,
    }

    let mut tree = Trie::<u8, HashMap<DocumentId, CodePosting>>::new();
    let code_parser = CodeParser::from_language(CodeLanguage::TSX);
    for (doc_id, doc) in orama_documents.iter().enumerate() {
        if !(doc_id == 0 || doc_id == 40) {
            // continue;
        }

        let doc_id = DocumentId(doc_id as u64);

        let code: &str = doc["code"].as_str().unwrap();
        let tokens = code_parser.tokenize_and_stem(code).unwrap();

        for (i, (token, stemmeds)) in tokens.into_iter().enumerate() {
            if let Some(posting) = tree.get_mut(token.bytes()) {
                let r = posting.entry(doc_id)
                    .or_insert(CodePosting {
                        token: token.clone(),
                        document_id: doc_id,
                        positions: HashMap::new(),
                    });
                r.positions.entry(Position(i))
                    .or_default()
                    .push(TokenType::Token);
            } else {
                let code_posting = CodePosting {
                    token: token.clone(),
                    document_id: doc_id,
                    positions: vec![
                        (Position(i), vec![TokenType::Token])
                    ].into_iter().collect(),
                };
                tree.insert(token.bytes(), vec![(doc_id, code_posting)].into_iter().collect());
            }

            /*
            let perms = generate_permutations(stemmeds.clone());
            for (perm, n) in perms {
                let n = n as u8;
                let stemmed = perm.join("");
                if let Some(posting) = tree.get_mut(stemmed.bytes()) {
                    let e = posting.positions.entry(Position(i));
                    e.or_default().push(TokenType::ComposedSubwords(n));
                } else {
                    tree.insert(stemmed.bytes(), CodePosting {
                        token: stemmed.clone(),
                        document_id: DocumentId(doc_id as u64),
                        positions: vec![
                            (Position(i), vec![TokenType::ComposedSubwords(n)])
                        ].into_iter().collect(),
                    });
                }
            }
            */

            for stemmed in stemmeds.iter() {
                if let Some(posting) = tree.get_mut(stemmed.bytes()) {
                    let r = posting.entry(doc_id)
                        .or_insert(CodePosting {
                            token: stemmed.clone(),
                            document_id: doc_id,
                            positions: HashMap::new(),
                        });
                    r.positions.entry(Position(i))
                        .or_default()
                        .push(TokenType::Token);
                } else {
                    let code_posting = CodePosting {
                        token: stemmed.clone(),
                        document_id: doc_id,
                        positions: vec![
                            (Position(i), vec![TokenType::Token])
                        ].into_iter().collect(),
                    };
                    tree.insert(stemmed.bytes(), vec![(doc_id, code_posting)].into_iter().collect());
                }
            }
        }
    }

    let term = r###"columnHelper.accessor('firstName')

// OR

{
  accessorKey: 'firstName',
}"###;
    let output = search(&code_parser, &tree, term);
    for (doc_id, score) in output.into_iter().take(3) {
        println!("doc_id: {:#?}, score: {:#?}, doc: {:?}", doc_id, score, orama_documents[doc_id.0 as usize]);
    }

    /*
    let term = "ColumnOrder";
    let output = search(&code_parser, &tree, term);
    for (doc_id, score) in output.into_iter().take(3) {
        println!("doc_id: {:#?}, score: {:#?}, doc: {:?}", doc_id, score, orama_documents[doc_id.0 as usize]);
    }
    */

    */

    Ok(())
}

/*
fn generate_permutations(input: Vec<String>) -> Vec<(Vec<String>, usize)> {
    let mut result = Vec::new();

    if input.len() > 5 {
        println!("input: {:#?}", input);
    }
    
    // Generate all subsets of the input vector
    for len in 2..=input.len().min(4) {
        for subset in input.iter().combinations(len) {
            // Generate all permutations of each subset
            for permutation in subset.into_iter().cloned().permutations(len) {
                result.push((permutation, len));
            }
        }
    }
    
    result
}
*/

