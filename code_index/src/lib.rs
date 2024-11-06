use std::{collections::HashMap, sync::RwLock};

use anyhow::Result;
use code_parser::CodeParser;
use ptrie::Trie;
use types::{CodeLanguage, DocumentId, FieldId};

pub struct CodeIndex {
    tree: RwLock<Trie<u8, HashMap<DocumentId, HashMap<FieldId, CodePosting>>>>,
}

impl CodeIndex {
    pub fn new() -> Self {
        Self {
            tree: RwLock::new(Trie::<u8, HashMap<DocumentId, HashMap<FieldId, CodePosting>>>::new()),
        }
    }

    pub fn insert_multiple(&self, documents: HashMap<DocumentId, Vec<(FieldId, String)>>) -> Result<()> {
        let mut tree = self.tree.write().unwrap();

        let code_parser = CodeParser::from_language(CodeLanguage::TSX);

        for (doc_id, properties) in documents {

            for (field_id, code) in properties {
                let tokens = code_parser.tokenize_and_stem(&code).unwrap();

                for (i, (token, _)) in tokens.into_iter().enumerate() {
                    if let Some(posting) = tree.get_mut(token.bytes()) {
                        let r = posting.entry(doc_id)
                            .or_default();
                        let r = r.entry(field_id)
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
                        let mut field_postings = HashMap::new();
                        field_postings.insert(field_id, code_posting);
                        tree.insert(token.bytes(), vec![(doc_id, field_postings)].into_iter().collect());
                    }
                }
            }
        }

        Ok(())
    }

    pub fn search(
        &self, 
        term: String,
        field_ids: Option<Vec<FieldId>>,
        boost: HashMap<FieldId, f32>,
    ) -> HashMap<DocumentId, f32> {

        let code_parser = CodeParser::from_language(CodeLanguage::TSX);
        let tokens = code_parser.tokenize_and_stem(&term).unwrap();

        let tree = self.tree.read().unwrap();
    
        println!("tokenize_and_stem {term}: {:#?}", tokens);
    
        let mut output = HashMap::<DocumentId, f32>::new();
    
        for (token, stemmeds) in tokens.into_iter() {
            let (exact_match, _) = tree.find_postfixes_with_current(token.bytes());
    
            println!("exact_match: {:#?}", exact_match);
    
            if let Some(c) = exact_match {
                for (doc_id, code_posting) in c {
                    let v = output.entry(*doc_id)
                        .or_default();

                    let position_len = if let Some(field_ids) = field_ids.as_ref() {
                        field_ids.into_iter()
                            .filter_map(|field_id| {
                                let code_posting = code_posting.get(field_id)?;
                                Some(code_posting.positions.len())
                            })
                            .sum::<usize>()
                    } else {
                        code_posting.values()
                            .map(|code_posting| code_posting.positions.len())
                            .sum::<usize>()
                    };

                    *v = 10.0 * position_len as f32;
                }
            }
        }

        output
    }
}


#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct Position(usize);

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
