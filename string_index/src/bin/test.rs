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

    let mut batch: Vec<(DocumentId, Vec<(FieldId, String)>)> = Vec::with_capacity(BATCH_SIZE);
    for (id, result) in rdr.deserialize().enumerate() {
        // Notice that we need to provide a type hint for automatic
        // deserialization.
        let record: Record = result.unwrap();
        let doc_id = DocumentId(id);

        batch.push((doc_id, vec![
            (FieldId(0), record.title),
            (FieldId(1), record.tag),
            (FieldId(2), record.artist),
            (FieldId(3), record.year),
            (FieldId(4), record.views),
            (FieldId(5), record.features),
            (FieldId(6), record.lyrics),
            (FieldId(7), record.language_cld3),
            (FieldId(8), record.language_ft),
            (FieldId(9), record.language),
        ]));

        if (id + 1) % BATCH_SIZE == 0 {
            let now = std::time::Instant::now();
            println!("Inserting batch");
            // I don't like this
            let other = batch.drain(..);
            string_index.insert_multiple(other.collect_vec()).unwrap();
            let delta = now.elapsed();
            println!("Batch insertion done: {:?}", delta);
        }
    }
}