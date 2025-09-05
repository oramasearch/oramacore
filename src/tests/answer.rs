use anyhow::anyhow;
use futures::FutureExt;
use itertools::Itertools;
use oramacore_lib::hook_storage::HookType;
use std::{sync::Arc, time::Duration};
use tokio::{
    sync::{broadcast, mpsc, RwLock},
    time::sleep,
};

use crate::collection_manager::sides::read::IndexFieldStatsType;
use crate::tests::utils::wait_for;
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
        .unwrap();
    let mut config = create_oramacore_config();
    config.ai_server.llm.port = Some(output.port());

    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key;
    let index_client = collection_client.create_index().await.unwrap();

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

#[tokio::test(flavor = "multi_thread")]
async fn test_answer_before_retrieval() {
    init_log();

    let completition_mock = Arc::new(RwLock::new(vec![
        vec!["Explain deeply".to_string(), " how is Tommaso".to_string()],
        vec![
            "I'm Michele, a ".to_string(),
            " very good software developer".to_string(),
            " bla bla bla".to_string(),
        ],
    ]));
    let completition_req = Arc::new(RwLock::new(vec![]));

    let output = create_ai_server_mock(completition_mock, completition_req.clone())
        .await
        .unwrap();
    let mut config = create_oramacore_config();
    config.ai_server.llm.port = Some(output.port());

    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key;
    let index_client = collection_client.create_index().await.unwrap();

    // Insert hook
    collection_client
        .insert_hook(
            HookType::BeforeRetrieval,
            r#"
async function beforeRetrieval(search_params) {
    console.log(search_params);
    await new Promise(r => setTimeout(r, 10)) // really async
    if (search_params.term === "How is Tommaso?") { // Replace Tommaso with Michele
        search_params.mode = 'fulltext'
        search_params.term = "Michele"

        return search_params
    }
}

export default { beforeRetrieval }
        "#
            .to_string(),
        )
        .await
        .unwrap();

    sleep(Duration::from_millis(200)).await;

    let docs = r#" [
        {"id":"1","name":"I'm Tommaso, a software developer"},
        {"id":"2","name":"I'm Michele, a very good software developer"}
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
        query: "How is Tommaso?".to_string(),
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
        .contains("How is Tommaso?"));

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
        .contains("How is Tommaso?"));
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
        .contains(r#""id":"2""#)); // Michele is returns. Tommaso is ignored due to the override

    let mut buffer = vec![];
    answer_receiver.recv_many(&mut buffer, 100).await;

    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::SelectedLLM(_))));
    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::OptimizeingQuery(_))));
    let Some(AnswerEvent::SearchResults(search_result)) = buffer
        .iter()
        .find(|i| matches!(&i, AnswerEvent::SearchResults(_)))
    else {
        panic!("No search result found")
    };
    assert_eq!(search_result.len(), 1);
    assert_eq!(
        search_result[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "2"
    );
    let output = buffer
        .iter()
        .filter_map(|i| match i {
            AnswerEvent::AnswerResponse(d) => Some(d.to_string()),
            _ => None,
        })
        .join("");
    assert_eq!(
        &output,
        "I'm Michele, a  very good software developer bla bla bla"
    );

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_answer_before_answer() {
    init_log();

    let completition_mock = Arc::new(RwLock::new(vec![
        vec!["Explain deeply".to_string(), " how is Tommaso".to_string()],
        vec![
            "I'm Tommaso, a ".to_string(),
            " software developer".to_string(),
            " bla bla bla".to_string(),
        ],
    ]));
    let completition_req = Arc::new(RwLock::new(vec![]));

    let output = create_ai_server_mock(completition_mock, completition_req.clone())
        .await
        .unwrap();
    let mut config = create_oramacore_config();
    config.ai_server.llm.port = Some(output.port());

    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();
    let collection_id = collection_client.collection_id;
    let read_api_key = collection_client.read_api_key;
    let index_client = collection_client.create_index().await.unwrap();

    // Insert hook
    collection_client
        .insert_hook(
            HookType::BeforeAnswer,
            r#"
async function beforeAnswer(a, b) {
    console.log(a, b);
    await new Promise(r => setTimeout(r, 10)) // really async
    return [a, b]
}

export default { beforeAnswer }
        "#
            .to_string(),
        )
        .await
        .unwrap();

    sleep(Duration::from_millis(200)).await;

    let docs = r#" [
        {"id":"1","name":"I'm Tommaso, a software developer"},
        {"id":"2","name":"I'm Michele, a very good software developer"}
    ]"#;
    let docs = serde_json::from_str::<Vec<Document>>(docs).unwrap();

    index_client
        .insert_documents(DocumentList(docs))
        .await
        .unwrap();

    wait_for(&collection_client, |collection_client| {
        async {
            let stats = collection_client.reader_stats().await.unwrap();
            let index_stats = &stats.indexes_stats[0];
            let field_stats = index_stats
                .fields_stats
                .iter()
                .find(|fs| &fs.field_path == "___orama_auto_embedding")
                .unwrap();
            let IndexFieldStatsType::UncommittedVector(field_stats) = &field_stats.stats else {
                panic!()
            };
            if field_stats.document_count == 2 {
                Ok(())
            } else {
                Err(anyhow!("embedding not yet arrived"))
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    let interaction = Interaction {
        conversation_id: "the-conversation-id".to_string(),
        interaction_id: "the-interaction-id".to_string(),
        llm_config: None,
        max_documents: None,
        messages: vec![],
        min_similarity: None,
        query: "How is Tommaso?".to_string(),
        system_prompt_id: None,
        ragat_notation: None,
        related: None,
        search_mode: None,
        visitor_id: "the-visitor-id".to_string(),
    };

    let answer = Answer::try_new(test_context.reader.clone(), collection_id, read_api_key)
        .await
        .unwrap();

    let (log_sender, mut log_receiver) = broadcast::channel(100);
    let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();

    let log_sender = Arc::new(log_sender);
    answer
        .answer(interaction, answer_sender, Some(log_sender.clone()))
        .await
        .unwrap();

    sleep(Duration::from_millis(300)).await;

    let lock = completition_req.read().await;
    let v = lock.clone();
    drop(lock);

    assert_eq!(v.len(), 2);
    // The first message contains the original question
    let message = v[0]
        .get("messages")
        .unwrap()
        .as_array()
        .unwrap()
        .get(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(
        message.contains("How is Tommaso?"),
        "Current message: {message:?}"
    );

    // The second message contains the original question
    let message = v[1]
        .get("messages")
        .unwrap()
        .as_array()
        .unwrap()
        .get(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(
        message.contains("How is Tommaso?"),
        "Current message: {message:?}"
    );

    // The second message contains also the search response
    let message = v[1]
        .get("messages")
        .unwrap()
        .as_array()
        .unwrap()
        .get(1)
        .unwrap()
        .get("content")
        .unwrap()
        .as_str()
        .unwrap();
    assert!(
        message.contains(r#""id":"1""#),
        "Current message: {message:?}"
    ); // Tommaso

    let mut buffer = vec![];
    answer_receiver.recv_many(&mut buffer, 100).await;

    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::SelectedLLM(_))));
    assert!(buffer
        .iter()
        .any(|i| matches!(&i, AnswerEvent::OptimizeingQuery(_))));
    let Some(AnswerEvent::SearchResults(search_result)) = buffer
        .iter()
        .find(|i| matches!(&i, AnswerEvent::SearchResults(_)))
    else {
        panic!("No search result found")
    };
    assert_eq!(search_result.len(), 1);
    assert_eq!(
        search_result[0]
            .document
            .as_ref()
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap(),
        "1"
    );
    let output = buffer
        .iter()
        .filter_map(|i| match i {
            AnswerEvent::AnswerResponse(d) => Some(d.to_string()),
            _ => None,
        })
        .join("");
    assert_eq!(&output, "I'm Tommaso, a  software developer bla bla bla");

    let output = log_receiver.recv().await.unwrap();
    assert!(output.1.contains("How is Tommaso?")); // the script receives the original question
    assert!(output.1.contains("\"id\":\"1\"")); // and the search result

    drop(test_context);
}
