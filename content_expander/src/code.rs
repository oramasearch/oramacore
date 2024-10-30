use crate::prompts::{get_prompt, Prompts};
use html_parser::{Dom, Node};
use mistralrs::{IsqType, TextMessageRole, TextMessages, TextModelBuilder};

type CodeBlockDescriptions = Vec<String>;

#[derive(Debug)]
pub enum TextFormat {
    Markdown,
    HTML,
    Plaintext,
}

const TEXT_MODEL_ID: &str = "microsoft/Phi-3.5-mini-instruct";

pub async fn describe_code_blocks(
    text: String,
    format: TextFormat,
) -> Option<CodeBlockDescriptions> {
    let model = TextModelBuilder::new(TEXT_MODEL_ID)
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .build()
        .await
        .unwrap();

    let code_blocks = capture_code_blocks(text, format)?;

    let futures: Vec<_> = code_blocks
        .into_iter()
        .map(|code| format!("### Code\n\n{}", code))
        .map(|code| async {
            let messages = TextMessages::new()
                .add_message(TextMessageRole::System, get_prompt(Prompts::CodeDescriptor))
                .add_message(TextMessageRole::User, code);

            match model.send_chat_request(messages).await {
                Ok(response) => response
                    .choices
                    .first()
                    .and_then(|choice| choice.message.content.clone()),
                Err(_) => None,
            }
        })
        .collect();

    let results = futures_util::future::join_all(futures).await;

    let descriptions: Vec<String> = results.into_iter().flatten().collect();

    if descriptions.is_empty() {
        None
    } else {
        Some(descriptions)
    }
}

fn capture_code_blocks(text: String, format: TextFormat) -> Option<CodeBlockDescriptions> {
    match format {
        TextFormat::Markdown => capture_code_blocks_markdown(text),
        TextFormat::HTML => capture_code_blocks_html(text),
        TextFormat::Plaintext => Some(vec![text]),
    }
}

fn capture_code_blocks_markdown(text: String) -> Option<CodeBlockDescriptions> {
    let mut blocks = Vec::new();
    let mut current_block = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        if line.trim().starts_with("```") {
            if in_code_block {
                blocks.push(current_block.trim().to_string());
                current_block.clear();
                in_code_block = false;
            } else {
                in_code_block = true;
                if line.trim() != "```" {
                    let code_content = line.trim_start_matches('`').trim();
                    if !code_content.is_empty() {
                        current_block.push_str(code_content);
                        current_block.push('\n');
                    }
                }
            }
        } else if in_code_block {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    if !current_block.is_empty() {
        blocks.push(current_block.trim().to_string());
    }

    if blocks.is_empty() {
        None
    } else {
        Some(blocks)
    }
}

fn capture_code_blocks_html(text: String) -> Option<CodeBlockDescriptions> {
    let dom = Dom::parse(&text).ok()?;
    let blocks = extract_code_blocks_from_dom(&dom);

    if blocks.is_empty() {
        None
    } else {
        Some(blocks)
    }
}

#[allow(clippy::only_used_in_recursion)]
fn extract_code_blocks_from_dom(dom: &Dom) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current_block = String::new();

    #[allow(clippy::if_same_then_else)]
    fn process_node(node: &Node, current_block: &mut String, blocks: &mut Vec<String>) {
        match node {
            Node::Element(element) => {
                if element.name == "pre" || element.name == "code" {
                    for child in &element.children {
                        process_node(child, current_block, blocks);
                    }
                } else {
                    for child in &element.children {
                        process_node(child, current_block, blocks);
                    }
                }
            }
            Node::Text(text) => {
                current_block.push_str(&html_escape::decode_html_entities(text));
                current_block.push('\n');
            }
            _ => {}
        }
    }

    for node in &dom.children {
        process_node(node, &mut current_block, &mut blocks);
    }

    if !current_block.is_empty() {
        blocks.push(current_block.trim().to_string());
    }

    blocks
        .into_iter()
        .filter(|block| !block.trim().is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_blocks() {
        let text = r#"
            Here's an example of a code block:

            ```js
            import { create } from '@orama/orama'

            const db = create({ schema: { title: 'string' } })
            ```

            And that was it.
        "#;

        let code_blocks = capture_code_blocks(text.to_string(), TextFormat::Markdown).unwrap();
        assert_eq!(code_blocks.len(), 1);
        assert!(code_blocks[0].contains("import { create }"));
        assert!(code_blocks[0].contains("const db = create"));
    }

    #[test]
    fn test_multiple_code_blocks() {
        let text = r#"
            First block:
            ```rust
            fn main() {
                println!("Hello");
            }
            ```
            Second block:
            ```python
            def hello():
                print("World")
            ```
        "#;

        let code_blocks = capture_code_blocks(text.to_string(), TextFormat::Markdown).unwrap();
        assert_eq!(code_blocks.len(), 2);
        assert!(code_blocks[0].contains("println"));
        assert!(code_blocks[1].contains("print"));
    }

    #[test]
    fn test_inline_code_exclusion() {
        let text = r#"
            Here;s how to use `orama`'s methods:

            ```js
            import { create } from '@orama/orama'

            const db = create({ schema: { title: 'string' } })
            ```

            And that's it.
        "#;

        let code_blocks = capture_code_blocks(text.to_string(), TextFormat::Markdown).unwrap();
        assert_eq!(code_blocks.len(), 1);
        assert!(code_blocks[0].contains("import { create }"));
        assert!(code_blocks[0].contains("const db = create"));
    }
}
