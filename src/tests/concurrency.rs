use std::{sync::Arc, time::Duration};

use chrono::Utc;
use serde_json::json;
use tokio::{runtime::Builder, sync::Barrier, task::LocalSet, time::sleep};

use crate::{
    tests::utils::{init_log, TestContext},
    types::DocumentList,
};

#[tokio::test(flavor = "multi_thread", worker_threads = 5)]
async fn test_insert_create_collection_concurrency() {
    use std::sync::Mutex;

    #[derive(Clone, Debug)]
    enum LogType {
        InsertDocument {
            start: chrono::DateTime<Utc>,
            end: chrono::DateTime<Utc>,
        },
        CreateCollection {
            start: chrono::DateTime<Utc>,
            end: chrono::DateTime<Utc>,
        },
    }
    let events = Arc::new(Mutex::new(Vec::with_capacity(300)));

    let (sender, _) = tokio::sync::broadcast::channel(1);

    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs: Vec<_> = (0..200)
        .map(|i| json!({ "id": i.to_string(), "text": format!("text {}", i)}))
        .collect();
    let docs: DocumentList = docs.try_into().unwrap();

    let barrier = Arc::new(Barrier::new(3)); // 2 threads + main

    let barrier1 = Arc::clone(&barrier);
    let events1 = events.clone();
    let mut receiver1 = sender.subscribe();
    let insert_document_handler = std::thread::spawn(move || {
        let local = LocalSet::new();
        local.spawn_local(async move {
            println!("barrier1.wait().await");
            barrier1.wait().await;
            receiver1.recv().await.unwrap();
            println!("barrier1.wait().await DONE");
            let start = Utc::now();
            index_client.insert_documents(docs).await.unwrap();

            events1.lock().unwrap().push(LogType::InsertDocument {
                start,
                end: Utc::now(),
            });
        });

        // Let's tell the local set to run the async code and wait for it to finish.
        Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(local)
    });

    let barrier2 = Arc::clone(&barrier);
    let events2 = events.clone();
    let mut receiver2 = sender.subscribe();
    let insert_collections_handler = std::thread::spawn(move || {
        let local = LocalSet::new();
        local.spawn_local(async move {
            barrier2.wait().await;
            receiver2.recv().await.unwrap();

            for _ in 0..30 {
                let start = Utc::now();
                test_context.create_collection().await.unwrap();
                events2.lock().unwrap().push(LogType::CreateCollection {
                    start,
                    end: Utc::now(),
                });
            }

            // Leak the test_context to avoid run `drop`
            // We don't care too much in tests...
            Box::leak(Box::new(test_context));
        });

        // Let's tell the local set to run the async code and wait for it to finish.
        Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(local);
    });

    // Let's the threads start...
    sleep(Duration::from_millis(100)).await;

    println!("barrier.wait().await");
    barrier.wait().await; // start both at same time
    sender.send(()).unwrap();
    println!("barrier.wait().await DONE");

    println!("insert_document_handler.join().unwrap()");
    insert_document_handler.join().unwrap();
    insert_collections_handler.join().unwrap();
    println!("insert_document_handler.join().unwrap() DONE");

    let events = events.lock().unwrap();
    let (insert_doc_start, insert_doc_end) = events
        .iter()
        .filter_map(|ev| match ev {
            LogType::InsertDocument { start, end } => Some((start, end)),
            LogType::CreateCollection { .. } => None,
        })
        .next()
        .unwrap();

    let create_collection_during_insertion: Vec<_> = events
        .iter()
        .filter(|ev| match ev {
            LogType::InsertDocument { .. } => false,
            LogType::CreateCollection { start, end } => {
                insert_doc_start < start && insert_doc_end > end
            }
        })
        .collect();

    println!("--- {}", create_collection_during_insertion.len());
    assert!(!create_collection_during_insertion.is_empty());
}
