/// Binary to load synthetic transcripts from a JSONL file into OramaCore.
///
/// Reads a large JSONL file line by line (streaming, not loading all into memory),
/// batches documents, and inserts them in parallel into an OramaCore index.
///
/// Usage:
///   cargo run --bin load_transcripts -- [OPTIONS] <file>
///
/// Options:
///   --batch-size <N>       Number of documents per batch (default: 200)
///   --parallelism <N>      Number of concurrent insert requests (default: 4)
///   --collection-id <ID>   Collection ID (default: "transcripts")
///   --index-id <ID>        Index ID (default: "main")
use std::env;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Instant;

use reqwest::Client;
use serde_json::Value;
use tokio::sync::Semaphore;

const SERVER_URL: &str = "http://localhost:8080";
const MASTER_API_KEY: &str = "my-master-api-key";
const READ_API_KEY: &str = "read";
const WRITE_API_KEY: &str = "write";

struct Config {
    file_path: String,
    batch_size: usize,
    parallelism: usize,
    collection_id: String,
    index_id: String,
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut config = Config {
        file_path: String::new(),
        batch_size: 200,
        parallelism: 4,
        collection_id: "transcripts".to_string(),
        index_id: "main".to_string(),
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--batch-size" => {
                i += 1;
                config.batch_size = args[i]
                    .parse()
                    .expect("--batch-size requires a positive integer");
            }
            "--parallelism" => {
                i += 1;
                config.parallelism = args[i]
                    .parse()
                    .expect("--parallelism requires a positive integer");
            }
            "--collection-id" => {
                i += 1;
                config.collection_id = args[i].clone();
            }
            "--index-id" => {
                i += 1;
                config.index_id = args[i].clone();
            }
            other if other.starts_with("--") => {
                eprintln!("Unknown option: {other}");
                std::process::exit(1);
            }
            _ => {
                config.file_path = args[i].clone();
            }
        }
        i += 1;
    }

    if config.file_path.is_empty() {
        eprintln!(
            "Usage: load_transcripts [--batch-size N] [--parallelism N] [--collection-id ID] [--index-id ID] <file.jsonl>"
        );
        std::process::exit(1);
    }

    config
}

/// Flatten a nested JSON object into a single-level object with dot-separated keys.
/// Arrays are kept as-is, only nested objects are flattened.
fn flatten_json(value: &Value) -> Value {
    let mut result = serde_json::Map::new();
    if let Value::Object(map) = value {
        flatten_into(&mut result, map, "");
    }
    Value::Object(result)
}

fn flatten_into(result: &mut serde_json::Map<String, Value>, map: &serde_json::Map<String, Value>, prefix: &str) {
    for (key, value) in map {
        let full_key = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Value::Object(inner) => {
                flatten_into(result, inner, &full_key);
            }
            other => {
                result.insert(full_key, other.clone());
            }
        }
    }
}

async fn create_collection(client: &Client, collection_id: &str) -> anyhow::Result<()> {
    let url = format!("{SERVER_URL}/v1/collections/create");
    let body = serde_json::json!({
        "id": collection_id,
        "description": "Synthetic transcripts collection",
        "read_api_key": READ_API_KEY,
        "write_api_key": WRITE_API_KEY,
        "language": "english",
    });

    let resp = client
        .post(&url)
        .bearer_auth(MASTER_API_KEY)
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("Failed to create collection (HTTP {status}): {text}");
    }
    println!("Collection '{collection_id}' created: {text}");
    Ok(())
}

async fn create_index(client: &Client, collection_id: &str, index_id: &str) -> anyhow::Result<()> {
    let url = format!("{SERVER_URL}/v1/collections/{collection_id}/indexes/create");
    // No embedding field — use "none"
    let body = serde_json::json!({
        "id": index_id,
        "embedding": "none",
    });

    let resp = client
        .post(&url)
        .bearer_auth(WRITE_API_KEY)
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("Failed to create index (HTTP {status}): {text}");
    }
    println!("Index '{index_id}' created in collection '{collection_id}': {text}");
    Ok(())
}

async fn insert_batch(
    client: &Client,
    collection_id: &str,
    index_id: &str,
    batch: Vec<Value>,
) -> anyhow::Result<()> {
    let url = format!("{SERVER_URL}/v1/collections/{collection_id}/indexes/{index_id}/insert");

    let resp = client
        .post(&url)
        .bearer_auth(WRITE_API_KEY)
        .json(&batch)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await?;
        anyhow::bail!("Insert failed (HTTP {status}): {text}");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = parse_args();

    println!(
        "Loading transcripts from: {}",
        config.file_path
    );
    println!(
        "Batch size: {}, Parallelism: {}, Collection: {}, Index: {}",
        config.batch_size, config.parallelism, config.collection_id, config.index_id
    );

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    // Step 1: Create collection and index
    create_collection(&client, &config.collection_id).await?;
    create_index(&client, &config.collection_id, &config.index_id).await?;

    // Step 2: Read file line by line and send batches
    let file = std::fs::File::open(&config.file_path)?;
    let reader = BufReader::new(file);

    let semaphore = Arc::new(Semaphore::new(config.parallelism));
    let client = Arc::new(client);
    let collection_id = Arc::new(config.collection_id);
    let index_id = Arc::new(config.index_id);

    let mut batch: Vec<Value> = Vec::with_capacity(config.batch_size);
    let mut total_docs: usize = 0;
    let mut total_batches: usize = 0;
    let mut tasks = Vec::new();
    let start = Instant::now();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Error reading line: {e}");
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let doc: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error parsing JSON line: {e}");
                continue;
            }
        };

        // Flatten nested objects (e.g. metadata.account_info.*)
        let flat_doc = flatten_json(&doc);
        batch.push(flat_doc);

        if batch.len() >= config.batch_size {
            total_docs += batch.len();
            total_batches += 1;
            let batch_to_send = std::mem::replace(&mut batch, Vec::with_capacity(config.batch_size));

            let permit = semaphore.clone().acquire_owned().await?;
            let client = client.clone();
            let collection_id = collection_id.clone();
            let index_id = index_id.clone();
            let batch_num = total_batches;

            tasks.push(tokio::spawn(async move {
                let result =
                    insert_batch(&client, &collection_id, &index_id, batch_to_send).await;
                drop(permit);
                match result {
                    Ok(()) => {
                        if batch_num % 50 == 0 {
                            println!("Batch {batch_num} inserted successfully");
                        }
                    }
                    Err(e) => {
                        eprintln!("Batch {batch_num} failed: {e}");
                    }
                }
            }));
        }
    }

    // Send remaining documents
    if !batch.is_empty() {
        total_docs += batch.len();
        total_batches += 1;
        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let collection_id = collection_id.clone();
        let index_id = index_id.clone();
        let batch_num = total_batches;

        tasks.push(tokio::spawn(async move {
            let result = insert_batch(&client, &collection_id, &index_id, batch).await;
            drop(permit);
            match result {
                Ok(()) => println!("Final batch {batch_num} inserted successfully"),
                Err(e) => eprintln!("Final batch {batch_num} failed: {e}"),
            }
        }));
    }

    // Wait for all tasks to complete
    let mut errors = 0;
    for task in tasks {
        if let Err(e) = task.await {
            eprintln!("Task join error: {e}");
            errors += 1;
        }
    }

    let elapsed = start.elapsed();
    println!("\n--- Summary ---");
    println!("Total documents: {total_docs}");
    println!("Total batches: {total_batches}");
    println!("Errors: {errors}");
    println!("Elapsed: {elapsed:.2?}");
    println!(
        "Throughput: {:.0} docs/sec",
        total_docs as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}
