use std::{fs::File, sync::Arc};

use itertools::Itertools;
use rocksdb::OptimisticTransactionDB;
use storage::Storage;
use string_index::{DocumentId, FieldId, StringIndex};
use string_utils::{Language, Parser};
use tempdir::TempDir;

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
    let tmp_dir = TempDir::new("string_index_test").unwrap();
    let tmp_dir: String = tmp_dir.into_path().to_str().unwrap().to_string();
    let db = OptimisticTransactionDB::open_default(tmp_dir).unwrap();
    let storage = Arc::new(Storage::new(db));
    let parser = Parser::from_language(Language::English);
    
    let mut string_index = StringIndex::new(storage, parser);

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
            string_index
                .insert_multiple(FieldId(0), other.collect_vec())
                .unwrap();
            let delta = now.elapsed();
            println!("Batch insertion done: {:?}", delta);
        }
    }
}
