use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

use anyhow::Result;
use oramacore_lib::data_structures::fst::FSTIndex;
use oramacore_lib::data_structures::map::Map;
use serde::{Deserialize, Serialize};

use crate::{collection_manager::global_info::GlobalInfo, types::DocumentId};

// (exact positions, stemmed positions)
type PostingIdPosition = (Vec<usize>, Vec<usize>);

#[derive(Serialize, Debug)]
pub struct CommittedStringFieldStats {
    pub key_count: usize,
    pub global_info: GlobalInfo,
    pub loaded: AtomicBool,
}

impl Clone for CommittedStringFieldStats {
    fn clone(&self) -> Self {
        Self {
            key_count: self.key_count,
            global_info: self.global_info.clone(),
            loaded: AtomicBool::new(self.loaded.load(std::sync::atomic::Ordering::Acquire)),
        }
    }
}

const FST_INDEX_FILE_NAME: &str = "fst.map";
const POSTING_ID_INDEX_FILE_NAME: &str = "posting_id_storage.map";
const DOCUMENT_LENGTHS_PER_DOCUMENT_FILE_NAME: &str = "length_per_documents.map";

#[derive(Debug, Serialize, Deserialize)]
pub struct StringFilterFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}

#[derive(Debug)]
struct DocumentLengthsPerDocument {
    inner: Map<DocumentId, u32>,
}
impl DocumentLengthsPerDocument {
    fn from_map(inner: Map<DocumentId, u32>) -> Self {
        Self { inner }
    }

    fn load(file_path: PathBuf) -> Result<Self> {
        let inner = Map::load(file_path)?;
        Ok(Self::from_map(inner))
    }

    fn get_length(&self, doc_id: &DocumentId) -> u32 {
        self.inner.get(doc_id).copied().unwrap_or(1)
    }
}

#[derive(Debug)]
struct PostingIdStorage {
    // id -> (doc_id, (exact positions, stemmed positions))
    inner: Map<u64, Vec<(DocumentId, PostingIdPosition)>>,
}
impl PostingIdStorage {
    fn load(file_path: PathBuf) -> Result<Self> {
        Ok(Self {
            inner: Map::load(file_path)?,
        })
    }

    fn get_posting(&self, posting_id: &u64) -> Option<&Vec<(DocumentId, PostingIdPosition)>> {
        self.inner.get(posting_id)
    }
}

/// Reads old FST-format string field data and returns (doc_id, IndexedValue) pairs
/// for migration into the new StringStorage format.
///
/// The old format stores:
/// - FST: term -> posting_list_id
/// - PostingIdStorage: posting_list_id -> Vec<(DocumentId, (exact_positions, stemmed_positions))>
/// - DocumentLengthsPerDocument: DocumentId -> field_length
///
/// We group terms by document, build an IndexedValue per doc, and return them.
pub fn load_old_fst_data(
    data_dir: &Path,
) -> Result<Vec<(u64, oramacore_fields::string::IndexedValue)>> {
    use oramacore_fields::string::{IndexedValue as StringIndexedValue, TermData};

    let index = FSTIndex::load(data_dir.join(FST_INDEX_FILE_NAME))?;
    let posting_storage = PostingIdStorage::load(data_dir.join(POSTING_ID_INDEX_FILE_NAME))?;
    let document_lengths =
        DocumentLengthsPerDocument::load(data_dir.join(DOCUMENT_LENGTHS_PER_DOCUMENT_FILE_NAME))?;

    // Collect per-document term data: doc_id -> HashMap<term_string, TermData>
    let mut per_doc: HashMap<u64, HashMap<String, TermData>> = HashMap::new();

    for (term_bytes, posting_list_id) in index.iter() {
        let term_string = String::from_utf8_lossy(&term_bytes).to_string();

        let Some(postings) = posting_storage.get_posting(&posting_list_id) else {
            continue;
        };

        for (doc_id, (exact_positions, stemmed_positions)) in postings {
            let doc_terms = per_doc.entry(doc_id.0).or_default();
            let term_data = TermData::new(
                exact_positions.iter().map(|p| *p as u32).collect(),
                stemmed_positions.iter().map(|p| *p as u32).collect(),
            );
            // If the same term appears multiple times for the same doc (shouldn't happen normally),
            // we just overwrite since the FST maps term -> single posting_list_id
            doc_terms.insert(term_string.clone(), term_data);
        }
    }

    // Build IndexedValue for each document
    let mut result = Vec::with_capacity(per_doc.len());
    for (doc_id, terms) in per_doc {
        let field_length = document_lengths.get_length(&DocumentId(doc_id)) as u16;
        let indexed_value = StringIndexedValue::new(field_length, terms);
        result.push((doc_id, indexed_value));
    }

    Ok(result)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StringFieldInfo {
    pub field_path: Box<[String]>,
    pub data_dir: PathBuf,
}
