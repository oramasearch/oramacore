use std::collections::HashSet;
use std::{collections::HashMap, sync::RwLock};

use anyhow::Result;
use ptrie::Trie;
use regex::Regex;

use crate::code_parser::{
    CodeLanguage, CodeToken, FunctionDeclaration, ImportedTokens, JsxElement, NewParser,
};
use crate::collection_manager::dto::FieldId;
use crate::document_storage::DocumentId;
use crate::nlp::tokenizer::Tokenizer;

pub struct CodeIndex {
    tree: RwLock<Trie<u8, HashMap<DocumentId, HashMap<FieldId, CodePosting>>>>,
    new_parser: NewParser,
    english_tokenizer: Tokenizer,
}

impl Default for CodeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeIndex {
    pub fn new() -> Self {
        let new_parser = NewParser::new();
        Self {
            tree: RwLock::new(
                Trie::<u8, HashMap<DocumentId, HashMap<FieldId, CodePosting>>>::new(),
            ),
            new_parser,
            english_tokenizer: Tokenizer::english(),
        }
    }

    pub fn insert_multiple(
        &self,
        documents: HashMap<DocumentId, Vec<(FieldId, String)>>,
    ) -> Result<()> {
        let mut tree = self.tree.write().unwrap();

        for (doc_id, properties) in documents {
            let mut a = HashMap::<String, HashMap<FieldId, CodePosting>>::new();

            for (field_id, code) in properties {
                let extracted_tokens = self.new_parser.parse(CodeLanguage::TSX, &code).unwrap();
                let token_with_weight = get_tokens_from_code_tokens(extracted_tokens);

                for (token, weight) in token_with_weight {
                    let entry = a.entry(token).or_default();
                    let code_posting = entry.entry(field_id).or_insert(CodePosting { weight: 0.0 });
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
        _field_ids: Option<Vec<FieldId>>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let tokens = self.new_parser.parse(CodeLanguage::TSX, &term)?;
        let tokens = tokens
            .into_iter()
            .flat_map(|code_token| get_tokens_from_code_tokens(vec![code_token]))
            .map(|(token, _)| token)
            .chain(self.english_tokenizer.tokenize(&term))
            .collect::<Vec<_>>();

        let tree = self.tree.read().unwrap();

        let mut output = HashMap::<DocumentId, f32>::new();

        for token in tokens {
            let (exact_match, postfix_matches) = tree.find_postfixes_with_current(token.bytes());

            if let Some(c) = exact_match {
                for (doc_id, code_posting) in c {
                    if let Some(filtered_doc_ids) = filtered_doc_ids {
                        if !filtered_doc_ids.contains(doc_id) {
                            continue;
                        }
                    }

                    let v = output.entry(*doc_id).or_default();

                    for (field_id, code_posting) in code_posting {
                        let boost = boost.get(field_id).unwrap_or(&1.0);
                        *v += code_posting.weight * boost;
                    }
                }
            }

            for c in postfix_matches {
                for (doc_id, code_posting) in c {
                    let v = output.entry(*doc_id).or_default();

                    for (field_id, code_posting) in code_posting {
                        let boost = boost.get(field_id).unwrap_or(&1.0);
                        *v += code_posting.weight * boost * 0.1;
                    }
                }
            }
        }

        Ok(output)
    }
}

struct TokenWeight(f32);

fn get_tokens_from_code_tokens(code_tokens: Vec<CodeToken>) -> Vec<(String, TokenWeight)> {
    let re = Regex::new(r"[A-Z][a-z]*|[a-z]+").unwrap();

    code_tokens
        .into_iter()
        .flat_map(|code_token| match code_token {
            CodeToken::Comment(comment) => {
                vec![(comment, TokenWeight(0.5))]
            }
            CodeToken::Imported(imported) => {
                let ImportedTokens {
                    package,
                    identifiers,
                } = imported;

                let single = identifiers
                    .iter()
                    .flat_map(|identifier| {
                        return re
                            .find_iter(identifier)
                            .map(|m| (m.as_str().to_string(), TokenWeight(0.1)));
                    })
                    .collect::<Vec<_>>();

                vec![(package, TokenWeight(1.0))]
                    .into_iter()
                    .chain(
                        identifiers
                            .into_iter()
                            .map(|identifier| (identifier, TokenWeight(0.7))),
                    )
                    .chain(single)
                    .collect()
            }
            CodeToken::GlobalIdentifier(global_identifier) => {
                vec![(global_identifier.clone(), TokenWeight(0.5))]
                    .into_iter()
                    .chain(
                        re.find_iter(&global_identifier)
                            .map(|m| (m.as_str().to_string(), TokenWeight(0.1))),
                    )
                    .collect()
            }
            CodeToken::GlobalJsx(global_jsx) => {
                let JsxElement {
                    tag,
                    attribute_keys,
                } = global_jsx;
                vec![(tag, TokenWeight(0.5))]
                    .into_iter()
                    .chain(
                        attribute_keys
                            .into_iter()
                            .map(|attribute_key| (attribute_key, TokenWeight(0.5))),
                    )
                    .collect()
            }
            CodeToken::FunctionDeclaration(function_declaration) => {
                let FunctionDeclaration {
                    name,
                    params,
                    jsx,
                    comments,
                    identifiers,
                } = function_declaration;
                vec![(name, TokenWeight(0.3))]
                    .into_iter()
                    .chain(params.into_iter().map(|param| (param, TokenWeight(0.5))))
                    .chain(jsx.into_iter().flat_map(|jsx| {
                        let JsxElement {
                            tag,
                            attribute_keys,
                        } = jsx;

                        let has_uppercase = tag.chars().any(|c| c.is_uppercase());
                        let weight = if has_uppercase { 0.5 } else { 0.1 };
                        vec![(tag, TokenWeight(weight))].into_iter().chain(
                            attribute_keys
                                .into_iter()
                                .map(|attribute_key| (attribute_key, TokenWeight(0.5))),
                        )
                    }))
                    .chain(
                        comments
                            .into_iter()
                            .map(|comment| (comment, TokenWeight(0.2))),
                    )
                    .chain(identifiers.into_iter().map(|identifier| {
                        let has_uppercase = identifier.chars().any(|c| c.is_uppercase());
                        let weight = if has_uppercase { 0.5 } else { 0.1 };
                        (identifier, TokenWeight(weight))
                    }))
                    .collect()
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
struct CodePosting {
    weight: f32,
}
