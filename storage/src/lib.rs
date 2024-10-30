use std::sync::atomic::{AtomicUsize, Ordering};

use rocksdb::{DBPinnableSlice, Error, OptimisticTransactionDB};
use thiserror::Error;

pub struct Storage {
    db: OptimisticTransactionDB,
    id_generator_counter: AtomicUsize,
}

impl Storage {
    pub fn new(db: OptimisticTransactionDB) -> Self {
        Storage {
            db,
            id_generator_counter: AtomicUsize::new(0),
        }
    }

    pub fn generate_new_id(&self) -> usize {
        self.id_generator_counter.fetch_add(1, Ordering::SeqCst)
    }

    pub fn fetch(&self, key: &[u8]) -> Result<Option<DBPinnableSlice>, StorageError> {
        Ok(self.db.get_pinned(key)?)
    }

    pub fn fetch_multiple(&self, keys: Vec<Vec<u8>>) -> Result<Vec<Option<Vec<u8>>>, StorageError> {
        let output: Result<Vec<_>, Error> = self.db.multi_get(&keys).into_iter().collect();
        Ok(output?)
    }

    pub fn run_in_transaction<R, E>(
        &self,
        f: impl FnOnce(&rocksdb::Transaction<OptimisticTransactionDB>) -> anyhow::Result<R, E>,
    ) -> Result<R, StorageError>
    where
        E: std::convert::Into<anyhow::Error>,
    {
        let transaction = self.db.transaction();
        let output = match f(&transaction) {
            Ok(output) => output,
            Err(e) => {
                transaction.rollback()?;
                return Err(StorageError::GenericError(e.into()));
            }
        };
        transaction.commit()?;
        Ok(output)
    }
}

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Inner error")]
    StorageInnerError(#[from] rocksdb::Error),
    #[error("Generic error")]
    GenericError(#[from] anyhow::Error),
}
