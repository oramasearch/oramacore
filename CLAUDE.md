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

### Testing
Tests are organized in `src/tests/` with comprehensive coverage for:
- Search functionality (fulltext, vector, hybrid, geosearch)
- Collection lifecycle management
- Concurrent operations
- Answer generation
- Migration and data integrity

## Development constraints

- Tests must pass
- Prefer `expect` with a good message over `unwrap`.
- Prefer `anyhow::anyhow!` for error handling over `expect`.
- Prefer creating a custom error enum with `thiserror` crate for complex error handling.
- Comment the code to explain the intent and reason behind it.

### Test development
- Tests always using an complete e2e test and put that test under `src/tests/` folder.
- No hardcoded paths (use `std::env::temp_dir()` for temporary files)
- Use descriptive names for test files and functions.
