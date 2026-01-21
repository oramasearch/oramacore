use crate::tests::utils::{create_ai_server_mock, create_oramacore_config, init_log, TestContext};
use crate::types::{Document, DocumentList};
use crate::web_server::api::collection::openai_chat::conversions::{
    answer_event_to_openai_chunk, openai_request_to_interaction, OpenAIChatRequest,
    OpenAIStreamEvent, ResponseAccumulator,
};
use anyhow::Result;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::sleep;

async fn setup_test_collection(
    test_context: &TestContext,
) -> Result<(crate::types::CollectionId, crate::types::ReadApiKey)> {
    let collection_client = test_context.create_collection().await?;
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key.clone();
    let index_client = collection_client.create_index().await?;

    let docs = vec![
        json!({
            "id": "1",
            "name": "I'm John, a software developer"
        }),
        json!({
            "id": "2",
            "name": "I'm Mark, a product manager"
        }),
        json!({
            "id": "3",
            "name": "I'm Sarah, a data scientist working on machine learning"
        }),
    ];
    let docs: Vec<Document> = docs
        .into_iter()
        .map(|d| serde_json::from_value(d).unwrap())
        .collect();

    index_client.insert_documents(DocumentList(docs)).await?;

    sleep(Duration::from_millis(500)).await;

    Ok((collection_id, read_api_key))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_request_conversion() {
    init_log();

    let completition_mock = Arc::new(RwLock::new(vec![vec![
        "Based on the context, ".to_string(),
        "John is a software developer.".to_string(),
    ]]));
    let completition_req = Arc::new(RwLock::new(vec![]));

    let output = create_ai_server_mock(completition_mock, completition_req.clone())
        .await
        .unwrap();
    let mut config = create_oramacore_config();
    config.ai_server.llm.port = Some(output.port());

    let test_context = TestContext::new_with_config(config).await;
    let (collection_id, _read_api_key) = setup_test_collection(&test_context).await.unwrap();

    let openai_request = OpenAIChatRequest {
        messages: vec![ChatCompletionRequestUserMessageArgs::default()
            .content("Who is John?")
            .build()
            .unwrap()
            .into()],
        stream: false,
        temperature: Some(0.7),
        max_tokens: Some(100),
        top_p: Some(0.9),
        model: Some("gpt-3.5-turbo".to_string()),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
    };

    let interaction = openai_request_to_interaction(openai_request, collection_id).unwrap();

    assert_eq!(interaction.query, "Who is John?");
    assert!(interaction.interaction_id.starts_with("openai_"));
    assert!(interaction.visitor_id.starts_with("openai_visitor_"));
    assert!(interaction.conversation_id.starts_with("openai_conv_"));
    assert_eq!(interaction.search_mode, Some("hybrid".to_string()));
    assert_eq!(interaction.messages.len(), 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_response_accumulator_with_events() {
    init_log();

    use crate::ai::answer::AnswerEvent;
    use crate::types::{InteractionLLMConfig, SearchResultHit};

    let mut accumulator = ResponseAccumulator::new();

    accumulator.add_event(AnswerEvent::SelectedLLM(InteractionLLMConfig {
        provider: crate::ai::RemoteLLMProvider::OpenAI,
        model: "gpt-4".to_string(),
    }));

    let search_results = vec![
        SearchResultHit {
            id: "1".to_string(),
            score: 0.95,
            document: None,
        },
        SearchResultHit {
            id: "2".to_string(),
            score: 0.85,
            document: None,
        },
    ];
    accumulator.add_event(AnswerEvent::SearchResults(search_results));
    accumulator.add_event(AnswerEvent::AnswerResponse("Based on ".to_string()));
    accumulator.add_event(AnswerEvent::AnswerResponse("the context, ".to_string()));
    accumulator.add_event(AnswerEvent::AnswerResponse(
        "Mark is a product manager.".to_string(),
    ));

    let response_json = accumulator.to_openai_response("test-id");

    assert!(response_json.contains("\"id\":\"test-id\""));
    assert!(response_json.contains("\"object\":\"chat.completion\""));
    assert!(response_json.contains("\"model\":\"gpt-4\""));
    assert!(response_json.contains("Based on the context, Mark is a product manager."));
    assert!(response_json.contains("\"role\":\"assistant\""));
    assert!(response_json.contains("\"finish_reason\":\"stop\""));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_streaming_event_conversion() {
    init_log();

    use crate::ai::answer::AnswerEvent;
    use crate::types::{InteractionLLMConfig, SearchResultHit};

    let stream_id = "test-stream-id";
    let mut chunks = vec![];

    let event = AnswerEvent::SelectedLLM(InteractionLLMConfig {
        provider: crate::ai::RemoteLLMProvider::OpenAI,
        model: "gpt-4".to_string(),
    });
    if let Some(chunk) = answer_event_to_openai_chunk(event, stream_id) {
        chunks.push(chunk);
    }

    let search_results = vec![SearchResultHit {
        id: "1".to_string(),
        score: 0.95,
        document: None,
    }];
    let event = AnswerEvent::SearchResults(search_results);
    if let Some(chunk) = answer_event_to_openai_chunk(event, stream_id) {
        chunks.push(chunk);
    }

    for text in &["Sarah ", "is ", "a ", "data ", "scientist."] {
        let event = AnswerEvent::AnswerResponse(text.to_string());
        if let Some(chunk) = answer_event_to_openai_chunk(event, stream_id) {
            chunks.push(chunk);
        }
    }

    let event = AnswerEvent::AnswerResponse("".to_string());
    if let Some(chunk) = answer_event_to_openai_chunk(event, stream_id) {
        chunks.push(chunk);
    }

    assert!(!chunks.is_empty(), "Should have received chunks");

    let has_content_chunk = chunks.iter().any(|chunk| {
        if let OpenAIStreamEvent::Chunk(json) = chunk {
            json.contains("\"delta\"") && json.contains("\"content\"")
        } else {
            false
        }
    });

    assert!(has_content_chunk, "Should have at least one content chunk");

    let has_done = chunks
        .iter()
        .any(|chunk| matches!(chunk, OpenAIStreamEvent::Done));
    assert!(has_done, "Should have a DONE event");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_with_custom_model_provider() {
    init_log();

    let completition_mock = Arc::new(RwLock::new(vec![vec![
        "John is a software developer.".to_string()
    ]]));
    let completition_req = Arc::new(RwLock::new(vec![]));

    let output = create_ai_server_mock(completition_mock, completition_req.clone())
        .await
        .unwrap();
    let mut config = create_oramacore_config();
    config.ai_server.llm.port = Some(output.port());

    let test_context = TestContext::new_with_config(config).await;
    let (collection_id, _read_api_key) = setup_test_collection(&test_context).await.unwrap();

    let openai_request = OpenAIChatRequest {
        messages: vec![ChatCompletionRequestUserMessageArgs::default()
            .content("Who is John?")
            .build()
            .unwrap()
            .into()],
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        model: Some("openai/gpt-4".to_string()),
        frequency_penalty: None,
        presence_penalty: None,
    };

    let interaction = openai_request_to_interaction(openai_request, collection_id).unwrap();

    assert!(interaction.llm_config.is_some());
    let llm_config = interaction.llm_config.unwrap();
    assert!(matches!(
        llm_config.provider,
        crate::ai::RemoteLLMProvider::OpenAI
    ));
    assert_eq!(llm_config.model, "gpt-4");

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_with_multiple_message_types() {
    init_log();

    let test_context = TestContext::new().await;
    let (collection_id, _read_api_key) = setup_test_collection(&test_context).await.unwrap();

    let openai_request = OpenAIChatRequest {
        messages: vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("You are a helpful assistant that answers questions about people.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Who is Mark?")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .content("Mark is a product manager.")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("Can you tell me more about Mark?")
                .build()
                .unwrap()
                .into(),
        ],
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        model: None,
        frequency_penalty: None,
        presence_penalty: None,
    };

    let interaction = openai_request_to_interaction(openai_request, collection_id).unwrap();

    assert_eq!(interaction.query, "Can you tell me more about Mark?");
    assert_eq!(interaction.messages.len(), 4);

    use crate::types::Role;
    assert!(matches!(interaction.messages[0].role, Role::System));
    assert!(matches!(interaction.messages[1].role, Role::User));
    assert!(matches!(interaction.messages[2].role, Role::Assistant));
    assert!(matches!(interaction.messages[3].role, Role::User));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_error_no_user_message() {
    init_log();

    let test_context = TestContext::new().await;
    let (collection_id, _read_api_key) = setup_test_collection(&test_context).await.unwrap();

    let openai_request = OpenAIChatRequest {
        messages: vec![ChatCompletionRequestSystemMessageArgs::default()
            .content("You are a helpful assistant.")
            .build()
            .unwrap()
            .into()],
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        model: None,
        frequency_penalty: None,
        presence_penalty: None,
    };

    let result = openai_request_to_interaction(openai_request, collection_id);

    assert!(
        result.is_err(),
        "Should return an error when no user message is present"
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("No user message found"));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_chat_with_invalid_model_format() {
    init_log();

    let test_context = TestContext::new().await;
    let (collection_id, _read_api_key) = setup_test_collection(&test_context).await.unwrap();

    let openai_request = OpenAIChatRequest {
        messages: vec![ChatCompletionRequestUserMessageArgs::default()
            .content("Hello")
            .build()
            .unwrap()
            .into()],
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        model: Some("gpt-4".to_string()), // test case: missing provider prefix
        frequency_penalty: None,
        presence_penalty: None,
    };

    let interaction = openai_request_to_interaction(openai_request, collection_id).unwrap();

    assert!(interaction.llm_config.is_none());

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_response_accumulator_integration() {
    init_log();

    use crate::ai::answer::AnswerEvent;
    use crate::types::InteractionLLMConfig;

    let mut accumulator = ResponseAccumulator::new();

    accumulator.add_event(AnswerEvent::SelectedLLM(InteractionLLMConfig {
        provider: crate::ai::RemoteLLMProvider::OpenAI,
        model: "gpt-4".to_string(),
    }));

    accumulator.add_event(AnswerEvent::AnswerResponse("Hello, ".to_string()));
    accumulator.add_event(AnswerEvent::AnswerResponse("world!".to_string()));

    let response = accumulator.to_openai_response("test-request-id");

    assert!(response.contains("\"id\":\"test-request-id\""));
    assert!(response.contains("\"object\":\"chat.completion\""));
    assert!(response.contains("\"model\":\"gpt-4\""));
    assert!(response.contains("Hello, world!"));
    assert!(response.contains("\"role\":\"assistant\""));
    assert!(response.contains("\"finish_reason\":\"stop\""));
}
