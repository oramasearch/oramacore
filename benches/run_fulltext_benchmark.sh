#!/bin/bash
set -e

echo "Fulltext Search Performance Benchmark"
echo "======================================"

# Change to project root if running from benches directory
if [ "$(basename "$PWD")" = "benches" ]; then
    cd ..
fi

# Download test model if not available
if [ ! -d ".custom_models" ]; then
    echo "Downloading test model..."
    bash download-test-model.sh
fi

echo "Running fulltext search benchmark with games.json data..."
echo "This will test performance at multiple scales (500, 1K, 1.5K games)"
echo ""

# Check if games.json exists
if [ ! -f "benches/games.json" ]; then
    echo "Error: benches/games.json not found!"
    echo "Please ensure the games.json file is in the benches/ directory"
    exit 1
fi

echo "Using $(grep -c '"title"' benches/games.json) games from benches/games.json"
echo ""

# Run the benchmark test
cargo test test_quick_fulltext_benchmark -- --nocapture --test-threads=1
