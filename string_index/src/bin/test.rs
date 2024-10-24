use std::fs::File;

use itertools::Itertools;
use string_index::{DocumentId, FieldId, StringIndex};

#[derive(Debug, serde::Deserialize)]
struct Record {
    title: String,
    tag: String,
    artist: String,
    year: String,
    views: String,
    features: String,
    lyrics: String,
    // id: String,
    language_cld3: String,
    language_ft: String,
    language: String,
}

const BATCH_SIZE: usize = 10_000;

fn main() {
    let mut string_index = StringIndex::new(".".to_owned());

    let file = File::open("/Users/allevo/repos/gorama/datasets/song_lyrics.csv").unwrap();
    let mut rdr = csv::Reader::from_reader(file);

    let mut batch: Vec<(DocumentId, String)> = Vec::with_capacity(BATCH_SIZE);
    for (id, result) in rdr.deserialize().enumerate() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        let record: Record = result.unwrap();
        let doc_id = DocumentId(id);

        batch.push((doc_id, record.title));
        batch.push((doc_id, record.tag));
        batch.push((doc_id, record.artist));
        batch.push((doc_id, record.year));
        batch.push((doc_id, record.views));
        batch.push((doc_id, record.features));
        batch.push((doc_id, record.lyrics));
        batch.push((doc_id, record.language_cld3));
        batch.push((doc_id, record.language_ft));
        batch.push((doc_id, record.language));

        if (id + 1) % BATCH_SIZE == 0 {
            let now = std::time::Instant::now();
            println!("Inserting batch");
            // I don't like this
            let other = batch.drain(..);
            string_index.insert_multiple(FieldId(0), other.collect_vec()).unwrap();
            let delta = now.elapsed();
            println!("Batch insertion done: {:?}", delta);
        }
    }
}