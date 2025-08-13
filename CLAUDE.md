# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build Commands
```bash
cargo build
```

### Testing
```bash
# Download test model first (required for tests)
bash download-test-model.sh

# Run all tests
cargo test --all

# Run one test
cargo test test_name

```

### Code Quality
```bash
cargo check --all-features

# Format code
cargo fmt

# Run clippy (linter)
cargo clippy --fix --allow-dirty --allow-staged
```

### Running the Application
```bash
# Run with default config
cargo run --bin oramacore
```

## Architecture Overview

OramaCore is an AI runtime with full-text search engine, vector database, and LLM interface capabilities, structured around a two-sided architecture:

### Core Architecture
- **Write Side** (`WriteSide`): Handles data ingestion, indexing, and write operations
- **Read Side** (`ReadSide`): Handles search queries, AI services, and read operations
- **Channel Communication**: Write and read sides communicate via configurable channels (RabbitMQ streams supported)

### Key Components

#### Collection Manager (`src/collection_manager/`)
- **Generic KV Store**: Key-value storage abstraction
- **Read Operations**: Search, analytics, document storage access
- **Write Operations**: Document insertion, index creation, JWT management
- **Index Management**: Both committed and uncommitted field indexes for strings, numbers, dates, geopoints, vectors

#### AI Services (`src/ai/`)
- **LLM Service**: Local GPU management and remote LLM integration
- **Embeddings**: Automatic embeddings generation and selection
- **Answer Generation**: Conversational AI capabilities with context evaluation
- **State Machines**: Advanced auto-query and answer generation workflows

#### Search Infrastructure (`src/indexes/`)
- **HNSW**: Vector similarity search (both HNSW and HNSW2 implementations)
- **FST**: Finite State Transducers for text search
- **Radix Tree**: Prefix-based indexing
- **BM25**: Relevance scoring implementation

#### Web Server (`src/web_server/`)
- Built on Axum framework
- OpenAPI 3 documentation support
- Collection management APIs
- Search and analytics endpoints
- Answer session management

### Configuration
- Uses `config.yaml` by default (override with `CONFIG_PATH` env var)
- Supports environment variable overrides with `ORAMACORE_` prefix
- Features can be toggled: `reader`, `writer`

### Workspace Structure
- **Main crate**: Core application logic
- **Sub-crates**: `hnsw2`, `filters`, `bkd`, `nlp`, `hook_storage`, `fs`
- **AI Server**: Python-based gRPC service for embeddings and LLM inference (`src/ai_server/`)

### Testing Architecture

Tests are organized in `src/tests/`:
- **Search functionality**: fulltext, vector, hybrid, geosearch, facets, sorting
- **Collection lifecycle**: creation, deletion, index management, document operations  
- **Concurrent operations**: multi-threaded access, race condition testing
- **AI features**: answer generation, embeddings, state machines
- **System integrity**: migrations, data consistency, filesystem behavior
- **Error scenarios**: edge cases, validation, resource limits

#### Test Infrastructure

**Core Test Classes:**
- `TestContext`: Main orchestration - sets up read/write sides, config, API keys
- `TestCollectionClient`: Collection-level operations (search, stats, index management)
- `TestIndexClient`: Index-level operations (document CRUD, deletion)

**Key Utilities:**
- `init_log()`: Initialize logging for test debugging
- `wait_for()`: Async utility with 40-second timeout, 20ms polling intervals
- `commit_all()`: Force commit on both reader and writer sides
- `generate_new_path()`: Create isolated temporary directories per test
- `reload()`: Restart system while preserving filesystem state

#### Testing Patterns

**Standard Test Structure:**
```rust
#[tokio::test(flavor = "multi_thread")]
async fn test_feature_name() {
    init_log();
    
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    
    // Test operations
    let documents: DocumentList = json!([/* test data */]).try_into().unwrap();
    index_client.insert_documents(documents).await.unwrap();
    
    // Verify results
    let results = collection_client.search(/* params */).await.unwrap();
    assert_eq!(results.count, expected_count);
    
    // Optional: Force commit for filesystem testing
    test_context.commit_all().await.unwrap();
}
```

**Async Coordination:**
```rust
// Wait for eventual consistency
wait_for(client, |c| {
    let reader = c.reader.clone();
    async move {
        let stats = reader.collection_stats(/* params */).await?;
        if stats.document_count != expected_count {
            bail!("Document count mismatch");
        }
        Ok(())
    }.boxed()
}).await?;
```

#### Test Configuration

**Test Data Generation:**
- `json!()` macro for test documents with `.try_into().unwrap()` conversion
- Isolated collections and indexes per test to prevent interference

## Development constraints

- Tests must pass
- Prefer `expect` with a good message over `unwrap`.
- Prefer `anyhow::anyhow!` for error handling over `expect`.
- Prefer creating a custom error enum with `thiserror` crate for complex error handling.
- Comment the code to explain the intent and reason behind it.
- Prefer import the structures and enums on the top of the file instead of use the full name inside the code

### Test development
- Tests always using an complete e2e test and put that test under `src/tests/` folder.
- No hardcoded paths (use `std::env::temp_dir()` for temporary files)
- Use descriptive names for test files and functions.
