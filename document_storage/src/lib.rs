use std::sync::{atomic::AtomicU64, Arc};

use storage::Storage;
use types::{Document, DocumentId};

pub struct DocumentStorage {
    storage: Arc<Storage>,
    document_id_generator: AtomicU64,
}

const DOCUMENT_STORAGE_TAG: u8 = 1;

impl DocumentStorage {
    pub fn new(storage: Arc<Storage>) -> Self {
        Self {
            storage,
            document_id_generator: AtomicU64::new(0),
        }
    }

    pub fn generate_document_id(&self) -> DocumentId {
        let id = self
            .document_id_generator
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DocumentId(id)
    }

    pub fn add_documents(
        &self,
        documents: Vec<(DocumentId, Document)>,
    ) -> Result<(), anyhow::Error> {
        self.storage.run_in_transaction(|transaction| {
            for (id, document) in documents {
                let value = serialize(document)?;
                let key = id.0.to_be_bytes();
                let key = &[
                    DOCUMENT_STORAGE_TAG,
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    key[5],
                    key[6],
                    key[7],
                ];
                transaction.put(key, &value)?;
            }

            Result::<(), anyhow::Error>::Ok(())
        })?;

        Ok(())
    }

    pub fn get(&self, id: DocumentId) -> Result<Option<Document>, anyhow::Error> {
        let key = id.0.to_be_bytes();
        let key = vec![
            DOCUMENT_STORAGE_TAG,
            key[0],
            key[1],
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
            key[7],
        ];

        let serialized_document = self.storage.fetch(&key)?;
        let doc = match serialized_document {
            Some(serialized_document) => dbg!(match unserialize(&serialized_document) {
                Ok(document) => Some(document),
                Err(e) => {
                    eprintln!("Failed to deserialize document: {:?}", e);
                    return Err(e.context("Failed to deserialize document"));
                }
            }),
            None => return Ok(None),
        };

        Ok(doc)
    }

    pub fn get_all(&self, ids: Vec<DocumentId>) -> Result<Vec<Option<Document>>, anyhow::Error> {
        let keys: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| {
                let key = id.0.to_be_bytes();
                vec![
                    DOCUMENT_STORAGE_TAG,
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    key[5],
                    key[6],
                    key[7],
                ]
            })
            .collect();
        let serialized_documents = self.storage.fetch_multiple(keys)?;

        let docs: Vec<_> = serialized_documents
            .into_iter()
            .map(|serialized_document| match serialized_document {
                Some(serialized_document) => match unserialize(&serialized_document) {
                    Ok(document) => Some(document),
                    Err(e) => {
                        eprintln!("Failed to deserialize document: {:?}", e);
                        None
                    }
                },
                None => None,
            })
            .collect();

        Ok(docs)
    }
}

// TODO: benchmark this and find a more performant way to serialize and deserialize
fn unserialize(input: &[u8]) -> Result<Document, anyhow::Error> {
    let inner = serde_json::from_slice(input)?;
    Ok(Document { inner })
}
fn serialize(doc: Document) -> Result<Vec<u8>, anyhow::Error> {
    Ok(serde_json::to_vec(&doc.inner)?)
}
