use anyhow::Result;
use dashmap::DashMap;
use tree_sitter::{Node, Parser};
use tree_sitter_typescript::{LANGUAGE_TSX, LANGUAGE_TYPESCRIPT};
use types::CodeLanguage;

#[derive(Default)]
pub struct NewParser {
    parsers: DashMap<CodeLanguage, Parser>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ImportedTokens {
    pub package: String,
    pub identifiers: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct JsxElement {
    pub tag: String,
    pub attribute_keys: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionDeclaration {
    pub name: String,
    pub comments: Vec<String>,
    pub params: Vec<String>,
    pub jsx: Vec<JsxElement>,
    pub identifiers: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CodeToken {
    Comment(String),
    Imported(ImportedTokens),
    FunctionDeclaration(FunctionDeclaration),
    GlobalIdentifier(String),
    GlobalJsx(JsxElement),
}

const TSX_COMMENT: u16 = 113;
const TSX_IMPORT_STATEMENT: u16 = 183;
const TSX_FUNCTION_DECLARATION: u16 = 242;

const TSX_STRING_FRAGMENT: u16 = 169;
const TSX_IDENTIFIER: u16 = 1;
const TSX_FORMAL_PARAMETERS: u16 = 275;
const TSX_SHORTHAND_PROPERTY_IDENTIFIER_PATTERN: u16 = 401;

const TSX_JSX_OPENING_ELEMENT: u16 = 231;
const TSX_JSX_SELF_CLOSING_ELEMENT: u16 = 235;
const TSX_PROPERTY_IDENTIFIER: u16 = 399;

const TSX_VARIABLE_DECLARATOR: u16 = 194;
const TSX_NEW_EXPRESSION: u16 = 250;
const TSX_CALL_EXPRESSION: u16 = 249;
const TSX_ARGUMENTS: u16 = 270;
const TSX_LEXICAL_DECLARATION: u16 = 193;
const TSX_JSX_ATTRIBUTE: u16 = 236;

impl NewParser {
    pub fn new() -> Self {
        Self {
            parsers: DashMap::new(),
        }
    }

    // The parse algorithm is inneficient because it traverses the tree multiple times
    // and allocates a lot of memory
    // This is mainly because the tree-sitter API is not very ergonomic
    // but, now, we don't care about performance
    // TODO: care about performance
    pub fn parse(&self, language: CodeLanguage, code: &str) -> Result<Vec<CodeToken>> {
        let tree = self.get(language, |parser| parser.parse(code, None))?;
        let tree = match tree {
            Some(tree) => tree,
            None => return Err(anyhow::anyhow!("Failed to parse the code.")),
        };

        let nodes = flat(tree.root_node(), |node| {
            match node.kind_id() {
                TSX_COMMENT => TraverseOption::KeepAndStopTraverse,
                TSX_IMPORT_STATEMENT => TraverseOption::KeepAndStopTraverse,
                TSX_FUNCTION_DECLARATION => TraverseOption::KeepAndStopTraverse,
                TSX_LEXICAL_DECLARATION => TraverseOption::KeepAndStopTraverse,
                TSX_JSX_OPENING_ELEMENT => TraverseOption::KeepAndStopTraverse,
                TSX_JSX_SELF_CLOSING_ELEMENT => TraverseOption::KeepAndStopTraverse,
                _ => {
                    // println!("Node: {:#?} {} text: {}", node.kind(), node.kind_id(), &code[node.start_byte()..node.end_byte()]);
                    TraverseOption::SkipAndTraverse
                }
            }
        });

        let mut tokens = vec![];
        for node in nodes {
            match node.kind_id() {
                TSX_COMMENT => {
                    handle_comment(node, code, &mut tokens);
                }
                TSX_IMPORT_STATEMENT => {
                    handle_import_statement(node, code, &mut tokens);
                }
                TSX_FUNCTION_DECLARATION => {
                    handle_function_declaration(node, code, &mut tokens);
                }
                TSX_LEXICAL_DECLARATION => {
                    handle_global_variable_declaration(node, code, &mut tokens);
                }
                TSX_JSX_OPENING_ELEMENT | TSX_JSX_SELF_CLOSING_ELEMENT => {
                    handle_global_jsx_element(node, code, &mut tokens);
                }
                _ => {}
            }
        }

        Ok(tokens)
    }

    pub fn parse_text(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }

    fn get<R>(&self, language: CodeLanguage, f: impl FnOnce(&mut Parser) -> R) -> Result<R> {
        let v = self.parsers.get_mut(&language);
        match v {
            Some(mut parser) => return Ok(f(parser.value_mut())),
            None => {
                let mut parser = Parser::new();

                match language {
                    CodeLanguage::TSX => {
                        parser.set_language(&LANGUAGE_TSX.into())?;
                    }
                    CodeLanguage::TypeScript => {
                        parser.set_language(&LANGUAGE_TYPESCRIPT.into())?;
                    }
                    _ => return Err(anyhow::anyhow!("Language not supported yet.")),
                }

                let output = f(&mut parser);

                self.parsers.insert(language, parser);

                Ok(output)
            }
        }
    }
}

fn handle_global_jsx_element(node: Node<'_>, code: &str, nodes: &mut Vec<CodeToken>) {
    let html_tag = flat(node, |node| {
        if node.kind_id() == TSX_IDENTIFIER {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let html_tag = html_tag
        .into_iter()
        .map(|node| {
            let t = &code[node.start_byte()..node.end_byte()];
            t.to_string()
        })
        .collect::<Vec<String>>();

    let jsx_attributes = flat(node, |node| {
        if node.kind_id() == TSX_JSX_ATTRIBUTE {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let attribute_keys = jsx_attributes
        .into_iter()
        .filter_map(|node| {
            let identifiers = flat(node, |node| {
                if node.kind_id() == TSX_PROPERTY_IDENTIFIER {
                    TraverseOption::KeepAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            // The first identifier is the key of the attribute
            if let Some(identifier) = identifiers.first() {
                let t = &code[identifier.start_byte()..identifier.end_byte()];
                Some(t.to_string())
            } else {
                None
            }
        })
        .collect::<Vec<String>>();

    let jsx_element = JsxElement {
        tag: html_tag[0].clone(),
        attribute_keys,
    };

    nodes.push(CodeToken::GlobalJsx(jsx_element));
}

fn handle_global_variable_declaration(node: Node<'_>, code: &str, nodes: &mut Vec<CodeToken>) {
    let identifiers = flat(node, |node| {
        if node.kind_id() == TSX_IDENTIFIER {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let identifiers = identifiers
        .into_iter()
        .map(|node| {
            let t = &code[node.start_byte()..node.end_byte()];
            t.to_string()
        })
        .collect::<Vec<String>>();

    for identifier in identifiers {
        nodes.push(CodeToken::GlobalIdentifier(identifier));
    }
}

fn handle_function_declaration(node: Node<'_>, code: &str, nodes: &mut Vec<CodeToken>) {
    let mut function_name_seen = false;
    let function_name = flat(node, move |node| {
        if node.kind_id() == TSX_IDENTIFIER {
            if function_name_seen {
                TraverseOption::SkipAndStopTraverse
            } else {
                function_name_seen = true;
                TraverseOption::KeepAndStopTraverse
            }
        } else {
            TraverseOption::SkipAndStopTraverse
        }
    });
    let function_name = function_name
        .into_iter()
        .map(|node| {
            let t = &code[node.start_byte()..node.end_byte()];
            t.to_string()
        })
        .collect::<String>();

    let comments = flat(node, |node| {
        if node.kind_id() == TSX_COMMENT {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let comments = comments
        .into_iter()
        .map(|node| {
            let comment = &code[node.start_byte()..node.end_byte()];
            clean_up_comment(comment)
        })
        .collect::<Vec<String>>();

    let params = flat(node, |node| {
        if node.kind_id() == TSX_FORMAL_PARAMETERS {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let params = flat(params[0], |node| {
        // This is limited feature: we keep the node only if there's the destructuring pattern
        // Otherwise we skip the node
        // This is suboptimal because not every function use this pattern
        // Anyway, in React world, this is quite common
        // TODO: Improve this to handle the other cases
        if node.kind_id() == TSX_SHORTHAND_PROPERTY_IDENTIFIER_PATTERN {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let params = params
        .into_iter()
        .map(|node| {
            let t = &code[node.start_byte()..node.end_byte()];
            t.to_string()
        })
        .collect::<Vec<String>>();

    let all_jsx = flat(node, |node| {
        let kind_id = node.kind_id();
        if kind_id == TSX_JSX_OPENING_ELEMENT || kind_id == TSX_JSX_SELF_CLOSING_ELEMENT {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let jsxs: Vec<_> = all_jsx
        .into_iter()
        .filter_map(|jsx| {
            let html_tag = flat(jsx, |node| {
                if node.kind_id() == TSX_IDENTIFIER {
                    TraverseOption::KeepAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            let html_tag = html_tag
                .into_iter()
                .map(|node| {
                    let t = &code[node.start_byte()..node.end_byte()];
                    t.to_string()
                })
                .collect::<Vec<String>>();

            let attributes = flat(jsx, |node| {
                if node.kind_id() == TSX_PROPERTY_IDENTIFIER {
                    TraverseOption::KeepAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            let attributes = attributes
                .into_iter()
                .map(|node| {
                    let t = &code[node.start_byte()..node.end_byte()];
                    t.to_string()
                })
                .collect::<Vec<String>>();

            if let Some(tag) = html_tag.first() {
                Some(JsxElement {
                    tag: tag.clone(),
                    attribute_keys: attributes,
                })
            } else {
                println!(
                    "--- No tag found {}",
                    &code[jsx.start_byte()..jsx.end_byte()]
                );
                None
            }
        })
        .collect();

    let variable_declarators = flat(node, move |node| {
        if node.kind_id() == TSX_VARIABLE_DECLARATOR {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });

    let identifiers: Vec<_> = variable_declarators
        .into_iter()
        .flat_map(|variable_declarator| {
            let identifiers = flat(variable_declarator, move |node| {
                if node.kind_id() == TSX_IDENTIFIER {
                    TraverseOption::KeepAndStopTraverse
                } else if node.kind_id() == TSX_CALL_EXPRESSION
                    || node.kind_id() == TSX_NEW_EXPRESSION
                {
                    // Handled later
                    TraverseOption::SkipAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            identifiers.into_iter().map(|node| {
                let t = &code[node.start_byte()..node.end_byte()];
                t.to_string()
            })
        })
        .collect();

    let new_expressions = flat(node, move |node| {
        if node.kind_id() == TSX_NEW_EXPRESSION {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });

    let other_identifiers: Vec<_> = new_expressions
        .into_iter()
        .flat_map(|new_expression| {
            let new_expression_identifiers = flat(new_expression, move |node| {
                if node.kind_id() == TSX_IDENTIFIER {
                    TraverseOption::KeepAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            let new_expression_identifiers = new_expression_identifiers
                .into_iter()
                .map(|node| {
                    let t = &code[node.start_byte()..node.end_byte()];
                    t.to_string()
                })
                .collect::<Vec<String>>();

            let params = flat(new_expression, move |node| {
                if node.kind_id() == TSX_PROPERTY_IDENTIFIER {
                    TraverseOption::KeepAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            let params = params
                .into_iter()
                .map(|node| {
                    let t = &code[node.start_byte()..node.end_byte()];
                    t.to_string()
                })
                .collect::<Vec<String>>();

            new_expression_identifiers.into_iter().chain(params)
        })
        .collect();

    let call_expressions = flat(node, move |node| {
        if node.kind_id() == TSX_CALL_EXPRESSION {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let call_expressions: Vec<_> = call_expressions
        .into_iter()
        .flat_map(|call_expression| {
            let fn_name_or_argument = flat(call_expression, move |node| {
                if node.kind_id() == TSX_IDENTIFIER || node.kind_id() == TSX_ARGUMENTS {
                    TraverseOption::KeepAndStopTraverse
                } else {
                    TraverseOption::SkipAndTraverse
                }
            });
            let fn_name: Vec<_> = fn_name_or_argument
                .iter()
                .filter(|node| node.kind_id() == TSX_IDENTIFIER)
                .map(|node| {
                    let t = &code[node.start_byte()..node.end_byte()];
                    t.to_string()
                })
                .collect();
            let arguments: Vec<_> = fn_name_or_argument
                .iter()
                .filter(|node| node.kind_id() == TSX_ARGUMENTS)
                .flat_map(|arguments| {
                    let argument_identifiers = flat(*arguments, move |node| {
                        if node.kind_id() == TSX_PROPERTY_IDENTIFIER {
                            TraverseOption::KeepAndStopTraverse
                        } else {
                            TraverseOption::SkipAndTraverse
                        }
                    });
                    argument_identifiers.into_iter().map(|node| {
                        let t = &code[node.start_byte()..node.end_byte()];
                        t.to_string()
                    })
                })
                .collect();

            fn_name.into_iter().chain(arguments)
        })
        .collect();

    let identifiers = identifiers
        .into_iter()
        .chain(other_identifiers)
        .chain(call_expressions)
        .collect();

    let function_declaration = FunctionDeclaration {
        name: function_name,
        comments,
        params,
        jsx: jsxs,
        identifiers,
    };

    nodes.push(CodeToken::FunctionDeclaration(function_declaration));
}

fn handle_comment(node: Node<'_>, code: &str, nodes: &mut Vec<CodeToken>) {
    let comment = &code[node.start_byte()..node.end_byte()];
    let comment = clean_up_comment(comment);
    nodes.push(CodeToken::Comment(comment));
}

fn handle_import_statement(node: Node<'_>, code: &str, nodes: &mut Vec<CodeToken>) {
    let import_identifiers = flat(node, |node| {
        if node.kind_id() == TSX_IDENTIFIER {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let import_identifiers = import_identifiers
        .into_iter()
        .map(|node| {
            let t = &code[node.start_byte()..node.end_byte()];
            t.to_string()
        })
        .collect::<Vec<String>>();

    let imported_package = flat(node, |node| {
        if node.kind_id() == TSX_STRING_FRAGMENT {
            TraverseOption::KeepAndStopTraverse
        } else {
            TraverseOption::SkipAndTraverse
        }
    });
    let imported_package = imported_package
        .into_iter()
        .map(|node| {
            let t = &code[node.start_byte()..node.end_byte()];
            t.to_string()
        })
        .collect::<Vec<String>>();

    nodes.push(CodeToken::Imported(ImportedTokens {
        package: imported_package[0].clone(),
        identifiers: import_identifiers,
    }));
}

#[allow(clippy::enum_variant_names)]
enum TraverseOption {
    SkipAndTraverse,
    SkipAndStopTraverse,
    KeepAndStopTraverse,
    #[allow(dead_code)]
    KeepAndTraverse,
}

fn flat<MatchFn>(node: Node<'_>, mut match_fn: MatchFn) -> Vec<Node<'_>>
where
    MatchFn: FnMut(&Node) -> TraverseOption,
{
    fn traverse<'cursor, 'node, MatchFn>(
        cursor: &mut tree_sitter::TreeCursor<'cursor>,
        nodes: &mut Vec<Node<'node>>,
        match_fn: &mut MatchFn,
    ) where
        'cursor: 'node,
        MatchFn: FnMut(&Node) -> TraverseOption,
    {
        if cursor.goto_first_child() {
            loop {
                let node = cursor.node();

                match match_fn(&node) {
                    TraverseOption::KeepAndStopTraverse => {
                        nodes.push(node);
                    }
                    TraverseOption::KeepAndTraverse => {
                        nodes.push(node);
                        traverse(cursor, nodes, match_fn);
                    }
                    TraverseOption::SkipAndTraverse => {
                        traverse(cursor, nodes, match_fn);
                    }
                    TraverseOption::SkipAndStopTraverse => {}
                }

                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }

    let mut nodes = vec![];
    let mut cursor = node.walk();
    traverse(&mut cursor, &mut nodes, &mut match_fn);

    nodes
}

fn clean_up_comment(comment: &str) -> String {
    let comment = comment.trim();
    let comment = comment.trim_start_matches("//");
    let comment = comment.trim();
    comment.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    use pretty_assertions::assert_eq;

    #[test]
    fn test_parser_simple1() {
        let code = r###"
// This is a comment
'use client'

import {
    foo,
    foo2
} from 'foo-package'

function myFunction() {
    if (foo) {
        return console.log("wow")
    } else {
        // Another comment
        return console.log("nope")
    }
}

export default function MyComponent({ children, todo }) {
    const [state, setState] = useState(0)
    return <div id="my-id">{children}</div>
}
"###;
        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![
                CodeToken::Comment("This is a comment".to_string()),
                CodeToken::Imported(ImportedTokens {
                    package: "foo-package".to_string(),
                    identifiers: vec!["foo".to_string(), "foo2".to_string()]
                }),
                CodeToken::FunctionDeclaration(FunctionDeclaration {
                    name: "myFunction".to_string(),
                    comments: vec!["Another comment".to_string()],
                    params: vec![],
                    jsx: vec![],
                    identifiers: vec!["console".to_string(), "console".to_string(),]
                }),
                CodeToken::FunctionDeclaration(FunctionDeclaration {
                    name: "MyComponent".to_string(),
                    comments: vec![],
                    params: vec!["children".to_string(), "todo".to_string()],
                    jsx: vec![JsxElement {
                        tag: "div".to_string(),
                        attribute_keys: vec!["id".to_string()]
                    }],
                    identifiers: vec![
                        "state".to_string(),
                        "setState".to_string(),
                        "useState".to_string()
                    ]
                })
            ]
        );
    }

    #[test]
    fn test_parser_simple2() {
        let code = r#"
// In Next.js, this file would be called: app/layout.jsx
import Providers from './providers'

export default function RootLayout({ children }) {
    return (
    <html lang="en">
        <head />
        <body>
        <Providers>{children}</Providers>
        </body>
    </html>
    )
}"#;

        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![
                CodeToken::Comment(
                    "In Next.js, this file would be called: app/layout.jsx".to_string(),
                ),
                CodeToken::Imported(ImportedTokens {
                    package: "./providers".to_string(),
                    identifiers: vec!["Providers".to_string(),],
                },),
                CodeToken::FunctionDeclaration(FunctionDeclaration {
                    name: "RootLayout".to_string(),
                    comments: vec![],
                    params: vec!["children".to_string(),],
                    jsx: vec![
                        JsxElement {
                            tag: "html".to_string(),
                            attribute_keys: vec!["lang".to_string(),],
                        },
                        JsxElement {
                            tag: "head".to_string(),
                            attribute_keys: vec![],
                        },
                        JsxElement {
                            tag: "body".to_string(),
                            attribute_keys: vec![],
                        },
                        JsxElement {
                            tag: "Providers".to_string(),
                            attribute_keys: vec![],
                        },
                    ],
                    identifiers: vec![],
                },),
            ]
        );
    }

    #[test]
    fn test_parser_simple3() {
        let code = r#"
function makeQueryClient() {
    return new QueryClient({
        defaultOptions: {
            queries: {
                staleTime: 60 * 1000,
            },
        },
    })
}
"#;
        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![CodeToken::FunctionDeclaration(FunctionDeclaration {
                name: "makeQueryClient".to_string(),
                comments: vec![],
                params: vec![],
                jsx: vec![],
                identifiers: vec![
                    "QueryClient".to_string(),
                    "defaultOptions".to_string(),
                    "queries".to_string(),
                    "staleTime".to_string(),
                ],
            },),]
        )
    }

    #[test]
    fn test_parser_simple4() {
        let code = r#"
function getQueryClient() {
    if (isServer) {
        return makeQueryClient()
    } else {
        if (!browserQueryClient) browserQueryClient = makeQueryClient({ foo: 'bar'})
        return browserQueryClient
    }
}"#;
        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![CodeToken::FunctionDeclaration(FunctionDeclaration {
                name: "getQueryClient".to_string(),
                comments: vec![],
                params: vec![],
                jsx: vec![],
                identifiers: vec![
                    "makeQueryClient".to_string(),
                    "makeQueryClient".to_string(),
                    "foo".to_string(),
                ],
            },),]
        );
    }

    #[test]
    fn test_parser_simple5() {
        let code = r#"
const a = <th
    key={header.id}
    colSpan={header.colSpan}
    style={{ width: `${header.getSize()}px` }}
/>"#;

        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![
                CodeToken::GlobalIdentifier("a".to_string()),
                CodeToken::GlobalIdentifier("th".to_string()),
                CodeToken::GlobalIdentifier("header".to_string()),
                CodeToken::GlobalIdentifier("header".to_string()),
                CodeToken::GlobalIdentifier("header".to_string()),
            ]
        );
    }

    #[test]
    fn test_parser_simple6() {
        let code = r#"
<th
    key={header.id}
    colSpan={header.colSpan}
    style={{ width: `${header.getSize()}px` }}
>"#;

        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![CodeToken::GlobalJsx(JsxElement {
                tag: "th".to_string(),
                attribute_keys: vec![
                    "key".to_string(),
                    "colSpan".to_string(),
                    "style".to_string(),
                ],
            }),]
        );
    }

    #[test]
    fn test_parser_simple7() {
        let code = r#"
<tr id="foo">
    <th scope="col">Player</th>
</tr>"#;

        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(
            output,
            vec![
                CodeToken::GlobalJsx(JsxElement {
                    tag: "tr".to_string(),
                    attribute_keys: vec!["id".to_string(),],
                }),
                CodeToken::GlobalJsx(JsxElement {
                    tag: "th".to_string(),
                    attribute_keys: vec!["scope".to_string(),],
                }),
            ]
        );
    }

    #[test]
    fn test_parser_simple8() {
        let code = r#"useQuery"#;

        let parser = NewParser::new();
        let output: Vec<CodeToken> = parser.parse(CodeLanguage::TSX, code).unwrap();

        assert_eq!(output, vec![]);
    }
}
