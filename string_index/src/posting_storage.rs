use std::sync::Arc;

use storage::{Storage, StorageError};
use thiserror::Error;

use crate::Posting;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PostingListId(usize);

#[derive(Debug, Error)]
pub enum PostingStorageError {
    #[error("PostingListId not found")]
    PostingListIdNotFound,
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),
    #[error("Serialize error")]
    SerializationError(#[from] bincode::Error),
}

const POSTING_STORAGE_TAG: u8 = 0;

pub struct PostingStorage {
    storage: Arc<Storage>,
}

impl PostingStorage {
    pub fn new(storage: Arc<Storage>) -> Self {
        PostingStorage { storage }
    }

    pub fn get(
        &self,
        posting_list_id: PostingListId,
        freq: usize,
    ) -> Result<(Vec<Posting>, usize), PostingStorageError> {
        let key = posting_list_id.0.to_be_bytes();
        let key = &[
            POSTING_STORAGE_TAG,
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
        ];
        let output = self.storage.fetch(key)?;

        let output = match output {
            None => return Err(PostingStorageError::PostingListIdNotFound),
            Some(data) => data,
        };

        let posting_vec = unserialize(&output)?;

        Ok((posting_vec, freq))
    }

    pub fn generate_new_id(&self) -> PostingListId {
        let id = self.storage.generate_new_id();
        PostingListId(id)
    }

    pub fn add_or_create(
        &self,
        posting_list_id: PostingListId,
        postings: Vec<Vec<Posting>>,
    ) -> Result<(), PostingStorageError> {
        let key = posting_list_id.0.to_be_bytes();
        let key = &[
            POSTING_STORAGE_TAG,
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
        ];

        self.storage.run_in_transaction(|transaction| {
            let pinned = transaction.get_pinned(key).unwrap();

            match pinned {
                Some(data) => {
                    let mut deserialized = unserialize(&data)?;
                    deserialized.extend(postings.into_iter().flatten());
                    let value = serialize(&deserialized)?;
                    transaction.put(key, value)?;
                }
                None => {
                    let value = serialize(&postings.into_iter().flatten().collect())?;
                    transaction.put(key, value)?;
                }
            };

            Result::<(), anyhow::Error>::Ok(())
        })?;

        Ok(())
    }
}

// TODO: benchmark this and find a more performant way to serialize and deserialize
fn unserialize(input: &[u8]) -> Result<Vec<Posting>, bincode::Error> {
    bincode::deserialize(input)
}
fn serialize(vec: &Vec<Posting>) -> Result<Vec<u8>, bincode::Error> {
    bincode::serialize(vec)
}
