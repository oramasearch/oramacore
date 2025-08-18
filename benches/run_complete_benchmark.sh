#!/bin/bash
set -e

echo "Complete OramaCore Performance Benchmark Suite"
echo "=============================================="

# Change to project root if running from benches directory
if [ "$(basename "$PWD")" = "benches" ]; then
    cd ..
fi

# Download test model if not available
if [ ! -d ".custom_models" ]; then
    echo "Downloading test model..."
    bash download-test-model.sh
fi

# Check if games.json exists
if [ ! -f "benches/games.json" ]; then
    echo "Error: benches/games.json not found!"
    echo "Please ensure the games.json file is in the benches/ directory"
    exit 1
fi

echo "Using $(grep -c '"title"' benches/games.json) games from benches/games.json"
echo ""

echo "Running Filter Operations Benchmark..."
echo "======================================"
cargo test test_filter_operations_benchmark -- --nocapture --test-threads=1

echo ""
echo ""
echo "Running Combined Fulltext + Filter Benchmark..."
echo "==============================================="
cargo test test_quick_fulltext_benchmark -- --nocapture --test-threads=1