use crate::ai::answer::AnswerEvent;
use crate::ai::RemoteLLMProvider;
use crate::types::{CollectionId, Interaction, InteractionLLMConfig, InteractionMessage, Role};
use anyhow::{anyhow, bail, Result};
use async_openai::types::ChatCompletionRequestMessage;
use serde::Deserialize;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIChatRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub model: Option<String>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

#[derive(Debug)]
pub enum OpenAIStreamEvent {
    Chunk(String),
    Done,
}

#[derive(Debug)]
pub struct ResponseAccumulator {
    content: String,
    model: String,
    search_results: Option<Vec<String>>,
}

impl ResponseAccumulator {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            model: String::new(),
            search_results: None,
        }
    }

    pub fn add_event(&mut self, event: AnswerEvent) {
        match event {
            AnswerEvent::SelectedLLM(config) => {
                self.model = config.model;
            }
            AnswerEvent::SearchResults(results) => {
                self.search_results = Some(results.iter().map(|r| r.id.clone()).collect());
            }
            AnswerEvent::AnswerResponse(chunk) => {
                if !chunk.is_empty() {
                    self.content.push_str(&chunk);
                }
            }
            _ => {}
        }
    }

    pub fn to_openai_response(self, request_id: &str) -> String {
        let created = current_timestamp();

        let response = serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        });

        serde_json::to_string(&response).expect("Failed to serialize OpenAI response")
    }
}

pub fn openai_request_to_interaction(
    request: OpenAIChatRequest,
    _collection_id: CollectionId,
) -> Result<Interaction> {
    let query = extract_last_user_message(&request.messages)?;

    let messages: Vec<InteractionMessage> = request
        .messages
        .iter()
        .filter_map(openai_message_to_interaction_message)
        .collect();

    let interaction_id = format!("openai_{}", cuid2::create_id()); // @todo: we may want to pass this via API later
    let visitor_id = format!("openai_visitor_{}", cuid2::create_id()); // @todo: we may want to pass this via API later
    let conversation_id = format!("openai_conv_{}", cuid2::create_id()); // @todo: we may want to pass this via API later

    let llm_config = request
        .model
        .as_ref()
        .and_then(|model_str| parse_provider_and_model(model_str).ok());

    // note: OpenAI parameters like temperature, max_tokens, etc. are not passed to the search
    // they would be used by the LLM provider but OramaCore's Answer handles that internally
    let interaction = Interaction {
        interaction_id,
        system_prompt_id: None, // Use default system prompt. @todo: allow custom system prompts later
        query,
        visitor_id,
        conversation_id,
        related: None, // note: OpenAI doesn't have related queries in request
        messages,
        llm_config,
        min_similarity: None,                    // use default (0.5)
        max_documents: None,                     // use default (5)
        ragat_notation: None,                    // note: not supported in OpenAI format
        search_mode: Some("hybrid".to_string()), // default to hybrid search mode. @todo: allow custom later, also allow to force a specific mode
    };

    Ok(interaction)
}

fn extract_last_user_message(messages: &[ChatCompletionRequestMessage]) -> Result<String> {
    for msg in messages.iter().rev() {
        if let Some(content) = get_message_content(msg) {
            if matches!(msg, ChatCompletionRequestMessage::User(_)) {
                return Ok(content);
            }
        }
    }

    bail!("No user message found in messages array")
}

fn get_message_content(msg: &ChatCompletionRequestMessage) -> Option<String> {
    use async_openai::types::*;

    match msg {
        ChatCompletionRequestMessage::System(m) => match &m.content {
            ChatCompletionRequestSystemMessageContent::Text(text) => Some(text.clone()),
            ChatCompletionRequestSystemMessageContent::Array(parts) => {
                let text: Vec<String> = parts
                    .iter()
                    .filter_map(|part| match part {
                        ChatCompletionRequestSystemMessageContentPart::Text(t) => {
                            Some(t.text.clone())
                        }
                        _ => None,
                    })
                    .collect();
                if text.is_empty() {
                    None
                } else {
                    Some(text.join(" "))
                }
            }
        },
        ChatCompletionRequestMessage::User(m) => match &m.content {
            ChatCompletionRequestUserMessageContent::Text(text) => Some(text.clone()),
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                let text: Vec<String> = parts
                    .iter()
                    .filter_map(|part| match part {
                        ChatCompletionRequestUserMessageContentPart::Text(t) => {
                            Some(t.text.clone())
                        }
                        _ => None,
                    })
                    .collect();
                if text.is_empty() {
                    None
                } else {
                    Some(text.join(" "))
                }
            }
        },
        ChatCompletionRequestMessage::Assistant(m) => match &m.content {
            Some(ChatCompletionRequestAssistantMessageContent::Text(text)) => Some(text.clone()),
            Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => {
                let text: Vec<String> = parts
                    .iter()
                    .filter_map(|part| match part {
                        ChatCompletionRequestAssistantMessageContentPart::Text(t) => {
                            Some(t.text.clone())
                        }
                        _ => None,
                    })
                    .collect();
                if text.is_empty() {
                    None
                } else {
                    Some(text.join(" "))
                }
            }
            None => None,
        },
        ChatCompletionRequestMessage::Tool(m) => match &m.content {
            ChatCompletionRequestToolMessageContent::Text(text) => Some(text.clone()),
            ChatCompletionRequestToolMessageContent::Array(parts) => {
                let text: Vec<String> = parts
                    .iter()
                    .filter_map(|part| match part {
                        ChatCompletionRequestToolMessageContentPart::Text(t) => {
                            Some(t.text.clone())
                        }
                        _ => None,
                    })
                    .collect();
                if text.is_empty() {
                    None
                } else {
                    Some(text.join(" "))
                }
            }
        },
        ChatCompletionRequestMessage::Function(m) => m.content.clone(),
        ChatCompletionRequestMessage::Developer(m) => match &m.content {
            ChatCompletionRequestDeveloperMessageContent::Text(text) => Some(text.clone()),
            ChatCompletionRequestDeveloperMessageContent::Array(parts) => {
                let text: Vec<String> = parts.iter().filter_map(|part| None).collect();
                if text.is_empty() {
                    Some(String::new())
                } else {
                    Some(text.join(" "))
                }
            }
        },
    }
}

fn openai_message_to_interaction_message(
    msg: &ChatCompletionRequestMessage,
) -> Option<InteractionMessage> {
    let content = get_message_content(msg)?;

    let role = match msg {
        ChatCompletionRequestMessage::System(_) => Role::System,
        ChatCompletionRequestMessage::User(_) => Role::User,
        ChatCompletionRequestMessage::Assistant(_) => Role::Assistant,
        ChatCompletionRequestMessage::Tool(_) => Role::User, // map tool messages to user
        ChatCompletionRequestMessage::Function(_) => Role::Assistant, // map function to assistant
        ChatCompletionRequestMessage::Developer(_) => Role::System, // ,ap developer to system
    };

    Some(InteractionMessage { role, content })
}

// Parse provider and model from "provider/model" format
// Examples: "openai/gpt-4", "orama/gpt-oss-120b", "anthropic/claude-3-opus"
fn parse_provider_and_model(model_str: &str) -> Result<InteractionLLMConfig> {
    if let Some((provider_str, model_name)) = model_str.split_once('/') {
        let provider = provider_str
            .parse::<RemoteLLMProvider>()
            .map_err(|e| anyhow::anyhow!("Invalid provider '{}': {}", provider_str, e))?;

        Ok(InteractionLLMConfig {
            provider,
            model: model_name.to_string(),
        })
    } else {
        bail!("No provider prefix in model string: {}", model_str)
    }
}

pub fn answer_event_to_openai_chunk(
    event: AnswerEvent,
    stream_id: &str,
) -> Option<OpenAIStreamEvent> {
    let created = current_timestamp();

    match event {
        // skip internal events
        AnswerEvent::Acknowledged => None,
        AnswerEvent::OptimizeingQuery(_) => None,
        AnswerEvent::RelatedQueries(_) => None,
        AnswerEvent::ResultAction { .. } => None,

        // first chunk with model selection
        AnswerEvent::SelectedLLM(config) => {
            let chunk = serde_json::json!({
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": config.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": null
                }]
            });

            Some(OpenAIStreamEvent::Chunk(
                serde_json::to_string(&chunk).expect("Failed to serialize chunk"),
            ))
        }

        AnswerEvent::SearchResults(_results) => {
            // for now, skip search results in streaming output
            // @todo: could include as tool_calls with document information
            None
        }

        AnswerEvent::AnswerResponse(content) => {
            if content.is_empty() {
                Some(OpenAIStreamEvent::Done)
            } else {
                let chunk = serde_json::json!({
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "",
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": content
                        },
                        "finish_reason": null
                    }]
                });

                Some(OpenAIStreamEvent::Chunk(
                    serde_json::to_string(&chunk).expect("Failed to serialize chunk"),
                ))
            }
        }

        AnswerEvent::FailedToRunPrompt(err)
        | AnswerEvent::FailedToFetchAnswer(err)
        | AnswerEvent::FailedToRunRelatedQuestion(err)
        | AnswerEvent::FailedToFetchRelatedQuestion(err) => {
            let chunk = serde_json::json!({
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": format!("[Error: {}]", err)
                    },
                    "finish_reason": "error"
                }]
            });

            Some(OpenAIStreamEvent::Chunk(
                serde_json::to_string(&chunk).expect("Failed to serialize chunk"),
            ))
        }

        AnswerEvent::FailedToGenerateTitle(_err) => None,
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_provider_and_model() {
        let result = parse_provider_and_model("openai/gpt-4").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::OpenAI));
        assert_eq!(result.model, "gpt-4");

        let result = parse_provider_and_model("openai/gpt-3.5-turbo").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::OpenAI));
        assert_eq!(result.model, "gpt-3.5-turbo");

        let result = parse_provider_and_model("orama/gpt-oss-120b").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::OramaCore));
        assert_eq!(result.model, "gpt-oss-120b");

        let result = parse_provider_and_model("oramacore/custom-model").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::OramaCore));
        assert_eq!(result.model, "custom-model");

        let result = parse_provider_and_model("anthropic/claude-3-opus").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::Anthropic));
        assert_eq!(result.model, "claude-3-opus");

        let result = parse_provider_and_model("google/gemini-pro").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::GoogleVertex));
        assert_eq!(result.model, "gemini-pro");

        let result = parse_provider_and_model("vertex/gemini-pro").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::GoogleVertex));
        assert_eq!(result.model, "gemini-pro");

        let result = parse_provider_and_model("groq/llama-70b").unwrap();
        assert!(matches!(result.provider, RemoteLLMProvider::Groq));
        assert_eq!(result.model, "llama-70b");

        assert!(parse_provider_and_model("gpt-4").is_err());
        assert!(parse_provider_and_model("unknown-model").is_err());

        assert!(parse_provider_and_model("invalid/model").is_err());
    }

    #[test]
    fn test_extract_last_user_message() {
        use async_openai::types::{
            ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        };

        let messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Hello")
                .build()
                .unwrap()
                .into(),
        ];

        let result = extract_last_user_message(&messages);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello");
    }

    #[test]
    fn test_extract_last_user_message_multiple_user_messages() {
        use async_openai::types::ChatCompletionRequestUserMessageArgs;

        let messages = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("First message")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Last message")
                .build()
                .unwrap()
                .into(),
        ];

        let result = extract_last_user_message(&messages);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Last message");
    }

    #[test]
    fn test_openai_message_to_interaction_message() {
        use async_openai::types::{
            ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        };

        let msg = ChatCompletionRequestSystemMessageArgs::default()
            .content("System message")
            .build()
            .unwrap()
            .into();
        let result = openai_message_to_interaction_message(&msg);
        assert!(result.is_some());
        let interaction_msg = result.unwrap();
        assert!(matches!(interaction_msg.role, Role::System));
        assert_eq!(interaction_msg.content, "System message");

        let msg = ChatCompletionRequestUserMessageArgs::default()
            .content("User message")
            .build()
            .unwrap()
            .into();
        let result = openai_message_to_interaction_message(&msg);
        assert!(result.is_some());
        let interaction_msg = result.unwrap();
        assert!(matches!(interaction_msg.role, Role::User));
        assert_eq!(interaction_msg.content, "User message");
    }

    #[test]
    fn test_response_accumulator() {
        use crate::types::{InteractionLLMConfig, SearchResultHit};

        let mut accumulator = ResponseAccumulator::new();

        accumulator.add_event(AnswerEvent::SelectedLLM(InteractionLLMConfig {
            provider: RemoteLLMProvider::OpenAI,
            model: "gpt-4".to_string(),
        }));

        accumulator.add_event(AnswerEvent::AnswerResponse("Hello, ".to_string()));
        accumulator.add_event(AnswerEvent::AnswerResponse("world!".to_string()));

        let response = accumulator.to_openai_response("test-id");
        assert!(response.contains("\"id\":\"test-id\""));
        assert!(response.contains("\"model\":\"gpt-4\""));
        assert!(response.contains("Hello, world!"));
    }

    #[test]
    fn test_answer_event_to_openai_chunk_selected_llm() {
        use crate::types::InteractionLLMConfig;

        let event = AnswerEvent::SelectedLLM(InteractionLLMConfig {
            provider: RemoteLLMProvider::OpenAI,
            model: "gpt-4".to_string(),
        });

        let result = answer_event_to_openai_chunk(event, "test-stream");
        assert!(result.is_some());

        match result.unwrap() {
            OpenAIStreamEvent::Chunk(json) => {
                assert!(json.contains("\"model\":\"gpt-4\""));
                assert!(json.contains("\"role\":\"assistant\""));
            }
            _ => panic!("Expected Chunk event"),
        }
    }

    #[test]
    fn test_answer_event_to_openai_chunk_answer_response() {
        let event = AnswerEvent::AnswerResponse("Test content".to_string());
        let result = answer_event_to_openai_chunk(event, "test-stream");
        assert!(result.is_some());

        match result.unwrap() {
            OpenAIStreamEvent::Chunk(json) => {
                assert!(json.contains("\"content\":\"Test content\""));
            }
            _ => panic!("Expected Chunk event"),
        }
    }

    #[test]
    fn test_answer_event_to_openai_chunk_empty_signals_done() {
        let event = AnswerEvent::AnswerResponse("".to_string());
        let result = answer_event_to_openai_chunk(event, "test-stream");
        assert!(result.is_some());

        match result.unwrap() {
            OpenAIStreamEvent::Done => {} // Expected
            _ => panic!("Expected Done event for empty AnswerResponse"),
        }
    }

    #[test]
    fn test_answer_event_to_openai_chunk_skips_internal_events() {
        // Test that internal events are skipped
        let event = AnswerEvent::Acknowledged;
        assert!(answer_event_to_openai_chunk(event, "test").is_none());

        let event = AnswerEvent::OptimizeingQuery("query".to_string());
        assert!(answer_event_to_openai_chunk(event, "test").is_none());

        let event = AnswerEvent::RelatedQueries("queries".to_string());
        assert!(answer_event_to_openai_chunk(event, "test").is_none());
    }
}
