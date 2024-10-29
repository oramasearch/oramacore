use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rocksdb::DB;

use crate::Posting;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PostingListId(usize);

pub struct PostingStorage {
    db: DB,
    id_generator_counter: AtomicUsize,
}

impl PostingStorage {
    pub fn new(base_path: String) -> Result<Self> {
        let db = DB::open_default(base_path)?;
        Ok(PostingStorage {
            db,
            id_generator_counter: AtomicUsize::new(0),
        })
    }

    pub fn get(
        &self,
        posting_list_id: PostingListId,
        freq: usize,
    ) -> Result<(Vec<Posting>, usize)> {
        let output = self.db.get_pinned(posting_list_id.0.to_be_bytes());

        let output = output
            .expect("Error on fetching posting_list_id")
            .expect("posting_list_id is unnknown");

        let posting_vec = unserialize(&output).expect("Error on deserialize Vec<Posting>");

        Ok((posting_vec, freq))
    }

    pub fn generate_new_id(&self) -> PostingListId {
        let id = self.id_generator_counter.fetch_add(1, Ordering::SeqCst);

        PostingListId(id)
    }

    pub fn add_or_create(
        &self,
        posting_list_id: PostingListId,
        postings: Vec<Vec<Posting>>,
    ) -> Result<()> {
        let key = posting_list_id.0.to_be_bytes();
        let output = self
            .db
            .get_pinned(key)
            .expect("Error on fetching posting_list_id");
        match output {
            None => {
                let value = serialize(&postings.into_iter().flatten().collect())?;
                self.db.put(key, value)?;
            }
            Some(data) => {
                let mut deserialized = unserialize(&data)?;
                deserialized.extend(postings.into_iter().flatten());
                let value = serialize(&deserialized)?;
                self.db.put(key, value)?;
            }
        }

        Ok(())
    }
}

// TODO: benchmark this and find a more performant way to serialize and deserialize
fn unserialize(input: &[u8]) -> Result<Vec<Posting>> {
    let output = bincode::deserialize(input)?;
    Ok(output)
}
fn serialize(vec: &Vec<Posting>) -> Result<Vec<u8>> {
    let output = bincode::serialize(vec)?;
    Ok(output)
}
