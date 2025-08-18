# Fulltext Search Benchmarks

This directory contains comprehensive benchmarks for OramaCore's fulltext search functionality, designed to track performance across iterations and detect regressions.

## Available Benchmarks

### `fulltext_simple.rs`
A lightweight benchmark focused on essential fulltext search operations:

- **Data Scales**: 1K, 5K documents
- **Query Types**: Single word, multi-word, with limits
- **Index States**: Both uncommitted and committed indexes
- **Execution Time**: ~3 seconds per test group

**Usage:**
```bash
cargo bench --bench fulltext_simple
```

### `fulltext_search.rs` 
A comprehensive benchmark suite covering advanced search scenarios:

- **Data Scales**: 1K, 10K documents  
- **Query Types**: Single word, multi-word, phrases, exact matches, fuzzy search, threshold filtering
- **Index States**: Uncommitted, committed, pagination, concurrent queries
- **Complex Scenarios**: Memory patterns, long queries, edge cases
- **Execution Time**: ~10-15 seconds per test group

**Usage:**
```bash
cargo bench --bench fulltext_search
```

## Benchmark Structure

### Data Generation
- **Realistic Content**: Documents contain varied technical vocabulary across domains (technology, business, science, engineering)
- **Scalable Patterns**: Content complexity scales with document ID to create diverse datasets
- **Consistent Structure**: All documents follow the same schema for reliable comparison

### Query Types Tested

1. **Single Word**: Simple term searches ("technology")
2. **Multi-Word**: Complex term combinations ("artificial intelligence machine learning") 
3. **Phrase Matching**: Quoted phrase searches with exact/fuzzy modes
4. **Exact Matching**: Precise term matching with exact=true
5. **Fuzzy Search**: Typo-tolerant searches with configurable tolerance
6. **Threshold Filtering**: Relevance threshold-based filtering
7. **Pagination**: Offset/limit combinations for result set navigation
8. **Complex Queries**: Long multi-term queries with various parameters

### Performance Metrics

- **Throughput**: Operations per second across different data scales
- **Latency**: Response time distribution for individual queries
- **Memory Usage**: Result set size impact on performance
- **Concurrent Load**: Multi-threaded query performance
- **Index State**: Performance comparison between uncommitted and committed indexes

## Running Benchmarks

### Prerequisites
```bash
# Required for all benchmarks
bash download-test-model.sh
```

### Basic Usage
```bash
# Run all fulltext benchmarks
cargo bench --bench fulltext_simple --bench fulltext_search

# Run with specific filters
cargo bench --bench fulltext_simple -- single_word
cargo bench --bench fulltext_search -- uncommitted

# Generate HTML reports
cargo bench -- --output-format html
```

### Environment Variables
```bash
# Enable detailed logging
LOG=info cargo bench --bench fulltext_simple

# Control benchmark precision
CRITERION_SAMPLE_SIZE=20 cargo bench --bench fulltext_simple
```

## Interpreting Results

### Key Metrics to Monitor

1. **Search Latency**: Time per query across different scales
   - Target: Sub-millisecond for small datasets, <10ms for large datasets
   - Watch for: Linear vs exponential scaling with data size

2. **Throughput**: Queries per second
   - Target: >1000 QPS for simple queries, >100 QPS for complex queries
   - Watch for: Degradation with concurrent load

3. **Memory Efficiency**: Performance with large result sets
   - Target: Consistent performance regardless of result set size
   - Watch for: Memory allocation spikes with large limits

4. **Index State Performance**: Committed vs uncommitted
   - Committed indexes should show more consistent performance
   - Uncommitted may be faster for writes but slower for reads

### Performance Regression Detection

Compare benchmark results across iterations:

```bash
# Baseline measurement
cargo bench --bench fulltext_simple > baseline.txt

# After changes
cargo bench --bench fulltext_simple > modified.txt

# Compare results (manual review of timing differences)
diff baseline.txt modified.txt
```

### Scaling Behavior

Monitor how performance scales with data size:
- **Linear scaling**: Expected for most operations
- **Sub-linear scaling**: Ideal (indicates algorithmic optimizations)
- **Super-linear scaling**: Concerning (indicates potential bottlenecks)

## Customization

### Adding New Query Types
1. Add enum variant to `QueryType` in `fulltext_search.rs`
2. Implement query generation in `create_search_query()`
3. Add to test vectors in benchmark functions

### Adjusting Data Scales
```rust
// In benchmark files
const SCALES: &[usize] = &[1_000, 10_000, 50_000]; // Modify as needed
```

### Performance Tuning
```rust
// Adjust sample sizes for faster/more precise benchmarks
group.sample_size(10);           // Fewer samples = faster
group.measurement_time(Duration::from_secs(5)); // Shorter time = faster
```

## Troubleshooting

### Common Issues

1. **AI Server Connection**: Ensure the test model is downloaded
   ```bash
   bash download-test-model.sh
   ```

2. **Memory Constraints**: Reduce scale sizes for resource-limited environments
3. **Timeout Issues**: Increase timeout values in benchmark configuration
4. **Compilation Time**: Use `fulltext_simple` for quicker iteration

### Performance Debugging

Enable detailed logging to understand bottlenecks:
```bash
RUST_LOG=oramacore=debug cargo bench --bench fulltext_simple
```

Monitor system resources during benchmarks:
```bash
# Run benchmark in background and monitor
cargo bench --bench fulltext_simple &
top -p $(pgrep cargo)
```

## Integration with CI/CD

The benchmarks are designed to be integrated into continuous integration pipelines:

```yaml
# Example GitHub Actions step
- name: Run Fulltext Benchmarks
  run: |
    bash download-test-model.sh
    cargo bench --bench fulltext_simple -- --output-format json | tee benchmark_results.json
```

This enables automated performance regression detection across pull requests and releases.