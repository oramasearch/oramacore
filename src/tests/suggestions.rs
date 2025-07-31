use std::{sync::Arc, time::Duration};

use hook_storage::HookType;
use itertools::Itertools;
use tokio::{
    sync::{broadcast, mpsc, RwLock},
    time::sleep,
};

use crate::{
    ai::answer::{Answer, AnswerEvent},
    tests::utils::{create_ai_server_mock, create_oramacore_config, init_log, TestContext},
    types::{Document, DocumentList, Interaction},
};

#[tokio::test(flavor = "multi_thread")]
async fn test_answer() {
    init_log();

    let completition_mock = Arc::new(RwLock::new(vec![
        vec!["Explain deeply".to_string(), " how you are".to_string()],
        vec![
            "I'm tommaso,".to_string(),
            " a software developer and".to_string(),
            " bla bla bla".to_string(),
        ],
    ]));
    let completition_req = Arc::new(RwLock::new(vec![]));

    let output = create_ai_server_mock(completition_mock, completition_req.clone())
        .await
        .expect("Failed to create AI server mock");
    let mut config = create_oramacore_config();
    config.ai_server.llm.port = output.port();

    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key;
    let index_client = collection_client.create_index().await.unwrap();

    println!("collection_id: {}", collection_id);

    let docs = r#" [
        {"id":"1","name":"I'm Tommaso, a software developer"}
    ]"#;
    let docs = serde_json::from_str::<Vec<Document>>(docs).unwrap();

    index_client
        .insert_documents(DocumentList(docs))
        .await
        .unwrap();

    sleep(Duration::from_millis(500)).await;

    let interaction = Interaction {
        conversation_id: "the-conversation-id".to_string(),
        interaction_id: "the-interaction-id".to_string(),
        llm_config: None,
        max_documents: None,
        messages: vec![],
        min_similarity: None,
        query: "Who are you?".to_string(),
        system_prompt_id: None,
        ragat_notation: None,
        related: None,
        search_mode: None,
        visitor_id: "the-visitor-id".to_string(),
    };

    let answer = Answer::try_new(test_context.reader.clone(), collection_id, read_api_key)
        .await
        .unwrap();

    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();

    answer
        .answer(interaction, answer_sender, None)
        .await
        .unwrap();

    let lock = completition_req.read().await;
    let v = lock.clone();
    drop(lock);

    assert_eq!(v.len(), 2);
    // The first message contains the original question
    assert!(v[0]
        .get("messages")
        .unwrap()
        .as_array()
        .unwrap()
        .get(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("Who are you?"));

    // The second message contains the original question
    assert!(v[1]
        .get("messages")
        .unwrap()
        .as_array()
        .unwrap()
        .get(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("Who are you?"));
    // The second message contains also the search response
    assert!(v[1]
        .get("messages")
        .unwrap()
        .as_array()
        .unwrap()
        .get(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap()
        .contains(r#""id":"1""#));

    let mut buffer = vec![];
    answer_receiver.recv_many(&mut buffer, 100).await;
    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::SelectedLLM(_))));
    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::OptimizeingQuery(_))));
    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::SearchResults(_))));
    let output = buffer
        .iter()
        .filter_map(|i| match i {
            AnswerEvent::AnswerResponse(d) => Some(d.to_string()),
            _ => None,
        })
        .join("");
    assert_eq!(&output, "I'm tommaso, a software developer and bla bla bla");

    drop(test_context);
}
