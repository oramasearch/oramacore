use anyhow::Result;
use embeddings::{load_models, OramaModels};
use serde::Deserialize;
use std::fs;
use std::time::Instant;
use types::DocumentId;
use vector_index::{VectorIndex, VectorIndexConfig};

#[derive(Deserialize, Debug)]
struct Movie {
    title: String,
    plot: String,
}

fn read_json_file() -> Result<Vec<Movie>> {
    let data = fs::read_to_string("./src/bin/imdb_top_1000_tv_series.json")?;
    let movies: Vec<Movie> = serde_json::from_str(&data)?;
    Ok(movies)
}

fn main() -> Result<()> {
    let config = VectorIndexConfig {
        embeddings_model: OramaModels::GTESmall,
    };

    let mut idx = VectorIndex::new(config);
    let models = load_models();
    let dataset = read_json_file()?;

    let start_embeddings = Instant::now();
    let embeddings = dataset
        .iter()
        .enumerate()
        .filter_map(|(i, movie)| {
            let str = format!("{}. {}", movie.title, movie.plot);
            let embedding = models
                .embed(OramaModels::GTESmall, vec![str], Some(1))
                .unwrap()
                .first()
                .cloned()?;
            Some((DocumentId(i as u64), embedding))
        })
        .collect::<Vec<(DocumentId, Vec<f32>)>>();

    let duration_embeddings = start_embeddings.elapsed();
    println!(
        "- {} embeddings generated in {}s\n",
        embeddings.len(),
        duration_embeddings.as_secs()
    );

    let start_indexing = Instant::now();
    for chunk in embeddings.chunks(100) {
        let batch = chunk
            .iter()
            .map(|(id, embedding)| (*id, embedding.as_slice()))
            .collect::<Vec<(DocumentId, &[f32])>>();
        idx.insert_batch(batch).unwrap();
    }

    let duration_indexing = start_indexing.elapsed();
    println!(
        "- {} vectors indexed in {}ms\n",
        embeddings.len(),
        duration_indexing.as_millis()
    );

    let search_query = "A movie about superheroes".to_string();
    let start_query_embedding = Instant::now();
    let query_embedding_result =
        models.embed(OramaModels::GTESmall, vec![search_query.clone()], Some(1))?;
    let query_embedding = query_embedding_result.first().unwrap();
    let duration_query_embedding = start_query_embedding.elapsed();
    println!(
        "- Query embedding generated in {}ms\n",
        duration_query_embedding.as_millis()
    );

    let start_knn_search = Instant::now();
    let neighbors = idx.search(query_embedding, 10);
    let duration_knn_search = start_knn_search.elapsed();
    println!(
        "- KNN search completed in {}μs\n",
        duration_knn_search.as_micros()
    );

    let retrieved_documents: Vec<&Movie> = neighbors
        .iter()
        .filter_map(|id| dataset.get(id.0 as usize))
        .collect();

    println!(
        "Matching documents for query: \"{}\"\n",
        search_query.clone()
    );
    for doc in &retrieved_documents {
        println!("Title: {}\nPlot: {}\n", doc.title, doc.plot);
    }

    Ok(())
}
