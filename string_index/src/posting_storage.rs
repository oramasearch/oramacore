use std::{fmt::Debug, sync::atomic::{AtomicUsize, Ordering}};

use anyhow::Result;
use rocksdb::{DBCommon, MultiThreaded, OptimisticTransactionDB, TransactionDB, DB};

use crate::Posting;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PostingListId(usize);

pub struct PostingStorage {
    db: OptimisticTransactionDB::<MultiThreaded>,
    id_generator_counter: AtomicUsize,
    rl: std::sync::RwLock<()>,
}
impl Debug for PostingStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output: Vec<_> = self.db.iterator(rocksdb::IteratorMode::Start)
            .flat_map(|r| {
                let (key, value) = r.expect("Error on iterator");

                let posting_list_id = PostingListId(usize::from_be_bytes([
                    key[0], key[1], key[2], key[3],
                    key[4], key[5], key[6], key[7],
                ]));

                let value = unserialize(&value)
                    .expect("Error on deserialize Vec<Posting>");

                Some((posting_list_id, value))
            })
            .collect();

        f.debug_struct("PostingStorage")
            .field("id_generator_counter", &self.id_generator_counter)
            .field("db_dump", &output)
            .finish()
    }
}

impl PostingStorage {
    pub fn new(base_path: String) -> Result<Self> {
        let db = OptimisticTransactionDB::<MultiThreaded>::open_default(base_path)?;
        Ok(PostingStorage {
            db,
            id_generator_counter: AtomicUsize::new(0),
            rl: std::sync::RwLock::new(()),
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

        let posting_vec = unserialize(&output)
            .expect("Error on deserialize Vec<Posting>");
        
        Ok((posting_vec, freq))
    }

    pub fn generate_new_id(
        &self,
    ) -> PostingListId {
        let id = self.id_generator_counter.fetch_add(1, Ordering::SeqCst);

        PostingListId(id)
    }

    pub fn add_or_create(
        &self,
        posting_list_id: PostingListId,
        postings: Vec<Vec<Posting>>,
    ) -> Result<()> {
        let l = self.rl.write().unwrap();

        let key = posting_list_id.0.to_be_bytes();

        let transaction = self.db.transaction();
        match transaction.get_pinned(key)
            .expect("Error on fetching posting_list_id") {
            None => {
                let value = serialize(&postings.clone().into_iter().flatten().collect()).unwrap();
                transaction.put(key, value).unwrap();
            },
            Some(data) => {
                let mut deserialized = unserialize(&data).unwrap();
                deserialized.extend(postings.clone().into_iter().flatten());
                let value = serialize(&deserialized).unwrap();
                transaction.put(key, value).unwrap();
            }
        }

        if let Err(e) = transaction.commit() {
            println!("{:?}, kind: {:?}", e, e.kind());
            return self.add_or_create(posting_list_id, postings);
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
