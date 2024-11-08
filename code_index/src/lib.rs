use std::{collections::HashMap, sync::RwLock};

use anyhow::Result;
use code_parser::{treesitter::CodeToken, CodeParser};
use code_parser::treesitter::{FunctionDeclaration, ImportedTokens, JsxElement, NewParser};
use ptrie::Trie;
use types::{CodeLanguage, DocumentId, FieldId};

pub struct CodeIndex {
    tree: RwLock<Trie<u8, HashMap<DocumentId, HashMap<FieldId, CodePosting>>>>,
    new_parser: NewParser,
}

impl CodeIndex {
    pub fn new() -> Self {
        let new_parser = NewParser::new();
        Self {
            tree: RwLock::new(Trie::<u8, HashMap<DocumentId, HashMap<FieldId, CodePosting>>>::new()),
            new_parser,
        }
    }

    pub fn insert_multiple(&self, documents: HashMap<DocumentId, Vec<(FieldId, String)>>) -> Result<()> {
        let mut tree = self.tree.write().unwrap();

        for (doc_id, properties) in documents {

            let mut a = HashMap::<String, HashMap<FieldId, CodePosting>>::new();

            for (field_id, code) in properties {
                let extracted_tokens = self.new_parser.parse(CodeLanguage::TSX, &code).unwrap();
                let token_with_weight = get_tokens_from_code_tokens(extracted_tokens);

                for (token, weight) in token_with_weight {
                    let entry = a.entry(token)
                        .or_default();
                    let code_posting = entry.entry(field_id)
                        .or_insert(CodePosting {
                            weight: 0.0,
                        });
                    code_posting.weight += weight.0;
                }
            }

            for (token, field_postings) in a {
                let mut b = HashMap::<DocumentId, HashMap<FieldId, CodePosting>>::new();
                b.insert(doc_id, field_postings);
                tree.insert(token.bytes(), b);
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

struct TokenWeight(f32);

fn get_tokens_from_code_tokens(code_tokens: Vec<CodeToken>) -> Vec<(String, TokenWeight)> {
    code_tokens.into_iter()
        .flat_map(|code_token| match code_token {
            CodeToken::Comment(comment) => {
                vec![(comment, TokenWeight(0.5))]
            },
            CodeToken::Imported(imported) => {
                let ImportedTokens{ package, identifiers } = imported;

                vec![(package, TokenWeight(0.5))]
                    .into_iter()
                    .chain(identifiers.into_iter().map(|identifier| (identifier, TokenWeight(0.5))))
                    .collect()
            },
            CodeToken::GlobalIdentifier(global_identifier) => {
                vec![(global_identifier, TokenWeight(0.5))]
            },
            CodeToken::GlobalJsx(global_jsx) => {
                let JsxElement{ tag, attribute_keys } = global_jsx;
                vec![(tag, TokenWeight(0.5))]
                    .into_iter()
                    .chain(attribute_keys.into_iter().map(|attribute_key| (attribute_key, TokenWeight(0.5))))
                    .collect()
            },
            CodeToken::FunctionDeclaration(function_declaration) => {
                let FunctionDeclaration { name, params, jsx, comments, identifiers } = function_declaration;
                vec![(name, TokenWeight(0.5))]
                    .into_iter()
                    .chain(params.into_iter().map(|param| (param, TokenWeight(0.5))))
                    .chain(jsx.into_iter().flat_map(|jsx| {
                        let JsxElement{ tag, attribute_keys } = jsx;
                        vec![(tag, TokenWeight(0.5))]
                            .into_iter()
                            .chain(attribute_keys.into_iter().map(|attribute_key| (attribute_key, TokenWeight(0.5))))
                    }))
                    .chain(comments.into_iter().map(|comment| (comment, TokenWeight(0.5))))
                    .chain(identifiers.into_iter().map(|identifier| (identifier, TokenWeight(0.5))))
                    .collect()
            },
        })
        .collect()
}


#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct Position(usize);

#[derive(Debug, Clone)]
struct CodePosting {
    weight: f32,
}

#[derive(Debug, Clone)]
enum TokenType {
    Token,
    ComposedSubwords(u8),
    Subword,
}
