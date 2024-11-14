use crate::LocalLLM;
use anyhow::Result;
use mistralrs::{Function, Model, RequestBuilder, TextMessageRole, Tool, ToolChoice, ToolType};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::string::ToString;
use std::sync::Arc;
use textwrap::dedent;
use utils::parse_json_safely;

pub struct QueryTranslator {
    pub llm: Arc<Model>,
}

const TRANSLATE_FUNCTION_NAME: &str = "generate_query";

impl QueryTranslator {
    pub async fn try_new() -> Result<Self> {
        let llm = LocalLLM::Phi3_5MiniInstruct.try_new().await?;

        Ok(QueryTranslator { llm })
    }

    pub async fn translate(
        &self,
        query: String,
        index_schema: Option<String>,
    ) -> Result<Option<String>> {
        let system_prompt = dedent(
            r#"
        	You are a tool used to translate text into an Orama query. Orama is a full-text, vector, and hybrid search engine.

	        Let me show you what you need to do with some examples.

            Example:
                - Query: `"What are the red wines that cost less than 20 dollars?"`
                - Schema: `{ name: 'string', content: 'string', price: 'number', tags: 'enum[]' }`
                - Generated query: `{ "term": "", "where": { "tags": { "containsAll": ["red", "wine"] }, "price": { "lt": 20 } } }`
        
            Another example:
                - Query: `"Show me 5 prosecco wines good for aperitif"`
                - Schema: `{ name: 'string', content: 'string', price: 'number', tags: 'enum[]' }`
                - Generated query: `{ "term": "prosecco aperitif", "limit": 5 }`
                
            One example without schema:
            - Query: `"What are the best headphones under $200 for listening to hi-fi music?"`
            - Schema: There is no schema for this query.
            - Generated query: `{ "term": "best headphones hi-fi under $200" }`
        
            One last example:
                - Query: `"Show me some wine reviews with a score greater than 4.5 and less than 5.0."`
                - Schema: `{ title: 'string', content: 'string', reviews: { score: 'number', text: 'string' } }]`
                - Generated query: `{ "term": "", "where": { "reviews.score": { "between": [4.5, 5.0] } } }`
        
            The rules to generate the query are:
        
            - Never use the "embedding" field.
            - Every query has a "term" field that is a string. It represents the full-text search terms. Can be empty (will match all documents).
            - You can use a "where" field that is an object. It represents the filters to apply to the documents. Its keys and values depend on the schema of the database:
                - If the field is a "string", you should not use operators. Example: `{ "where": { "title": "champagne" } }`.
                - If the field is a "number", you can use the following operators: "gt", "gte", "lt", "lte", "eq", "between". Example: `{ "where": { "price": { "between": [20, 100] } } }`. Another example: `{ "where": { "price": { "lt": 20 } } }`.
                - If the field is an "enum", you can use the following operators: "eq", "in", "nin". Example: `{ "where": { "tags": { "containsAll": ["red", "wine"] } } }`.
                - If the field is an "string[]", it's gonna be just like the "string" field, but you can use an array of values. Example: `{ "where": { "title": ["champagne", "montagne"] } }`.
                - If the field is a "boolean", you can use the following operators: "eq". Example: `{ "where": { "isAvailable": true } }`. Another example: `{ "where": { "isAvailable": false } }`.
                - If the field is a "enum[]", you can use the following operators: "containsAll". Example: `{ "where": { "tags": { "containsAll": ["red", "wine"] } } }`.
                - Nested properties are supported. Just translate them into dot notation. Example: `{ "where": { "author.name": "John" } }`.
                - Array of numbers are not supported.
                - Array of booleans are not supported.
            - If there is no schema, just use the "term" field.
        
            Reply with the generated query in a valid JSON format only. Nothing else.
        "#,
        );

        let json_schema: HashMap<String, Value> = serde_json::from_value(json!({
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "required": ["term"],
          "additionalProperties": false,
          "properties": {
            "term": {
              "type": "string",
              "description": "The full-text search terms. Can be empty."
            },
            "where": {
              "type": "object",
              "description": "The filters to apply to the documents. Keys and values depend on the database schema.",
              "additionalProperties": {
                "oneOf": [
                  {
                    "type": "string"
                  },
                  {
                    "type": "array",
                    "items": {
                      "type": "string"
                    }
                  },
                  {
                    "type": "object",
                    "properties": {
                      "gt": {"type": "number"},
                      "gte": {"type": "number"},
                      "lt": {"type": "number"},
                      "lte": {"type": "number"},
                      "eq": {
                        "oneOf": [
                          {"type": "number"},
                          {"type": "string"},
                          {"type": "boolean"}
                        ]
                      },
                      "between": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                      },
                      "in": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        }
                      },
                      "nin": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        }
                      },
                      "containsAll": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        }
                      }
                    },
                    "additionalProperties": false
                  }
                ]
              }
            }
          }
        }))?;

        let tools = vec![Tool {
            tp: ToolType::Function,
            function: Function {
                name: TRANSLATE_FUNCTION_NAME.to_string(),
                parameters: Some(json_schema),
                description: Some("Generates an Orama query based on the user question and the schema of the database, following the rules described in the prompt.".to_string()),
            }
        }];

        let messages = RequestBuilder::new()
            .add_message(TextMessageRole::System, system_prompt)
            .add_message(
                TextMessageRole::User,
                QueryTranslator::generate_user_prompt(query, index_schema),
            )
            .set_tools(tools)
            .set_tool_choice(ToolChoice::Auto);

        let response = self.llm.send_chat_request(messages).await?;
        let message = &response.choices[0].message;

        match message.clone().content {
            Some(content) => {
                let json_value = parse_json_safely(content)?;
                Ok(Some(json_value.to_string()))
            }
            None => Ok(None),
        }
    }

    fn generate_user_prompt(query: String, schema: Option<String>) -> String {
        let mut prompt = format!("Return a query for: {}\n", query);
        if let Some(schema) = schema {
            prompt += &format!("The schema is: {}", schema);
        } else {
            prompt += "There is no schema for this query.";
        };

        prompt
    }
}
