use anyhow::Result;
use embeddings::{load_models, OramaModels};
use serde::Deserialize;
use std::fs;
use std::time::Instant;
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
            Some((i.to_string(), embedding))
        })
        .collect::<Vec<(String, Vec<f32>)>>();

    let duration_embeddings = start_embeddings.elapsed();
    println!(
        "{} embeddings generated in {}s",
        embeddings.len(),
        duration_embeddings.as_secs()
    );

    let start_indexing = Instant::now();
    for chunk in embeddings.chunks(100) {
        let batch = chunk
            .iter()
            .map(|(id, embedding)| (id.clone(), embedding.as_slice()))
            .collect::<Vec<(String, &[f32])>>();
        idx.insert_batch(batch).unwrap();
    }

    let duration_indexing = start_indexing.elapsed();
    println!(
        "{} vectors indexed in {}ms",
        embeddings.len(),
        duration_indexing.as_millis()
    );

    Ok(())
}
