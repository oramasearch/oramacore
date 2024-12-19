use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicU32, Arc},
};

use anyhow::Result;
use dashmap::DashMap;
use posting_storage::PostingListId;
use scorer::BM25Scorer;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uncommitted::UncommittedStringFieldIndex;

use crate::{
    collection_manager::{dto::FieldId, sides::write::InsertStringTerms},
    document_storage::DocumentId,
};

pub mod posting_storage;
pub mod scorer;
mod uncommitted;
// mod committed;

pub type DocumentBatch = HashMap<DocumentId, Vec<(FieldId, Vec<(String, Vec<String>)>)>>;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Posting {
    pub document_id: DocumentId,
    pub field_id: FieldId,
    pub positions: Vec<usize>,
    /// Number of occurrences of the term in the field in the document
    pub occurrence: u32,
    pub field_length: u16,
}

#[derive(Debug, Clone)]
pub struct StringIndexValue {
    pub posting_list_id: PostingListId,
    pub term_frequency: usize,
}

pub struct StringIndex {
    uncommitted: DashMap<FieldId, UncommittedStringFieldIndex>,
    // committed: CommittedStringIndex,
    id_generator: Arc<AtomicU32>,
}

#[derive(Debug)]
pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}

impl StringIndex {
    pub fn new(id_generator: Arc<AtomicU32>) -> Self {
        StringIndex {
            uncommitted: DashMap::new(),
            // committed: CommittedStringIndex::new(PostingStorage::new(id_generator.clone())),
            id_generator,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn insert(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        terms: InsertStringTerms,
    ) -> Result<()> {
        let uncommitted = self
            .uncommitted
            .entry(field_id)
            .or_insert_with(|| UncommittedStringFieldIndex::new(self.id_generator.clone()));
        uncommitted.insert(doc_id, terms)?;

        /*
        let total_documents_length: u32 = terms
            .iter()
            .map(|(_, (_, postings))| {
                postings
                    .iter()
                    .map(|p| p.1.positions.len() as u32)
                    .sum::<u32>()
            })
            .sum();

        use std::collections::hash_map::Entry;
        let mut p: HashMap<_, _> = Default::default();

        let l = self.fst_map.read().await;
        if let Some(m) = l.as_ref() {
            debug!("Copy previous fst map");
            let mut keys = m.keys();
            while let Some(k) = keys.next() {
                p.insert(String::from_utf8_lossy(k).to_string(), m.get(k).unwrap());
            }
        } else {
            debug!("No previous fst map");
        }

        debug!("New terms: {:?}", terms);
        for (term, (freq, postings)) in terms {
            let k = term;

            match p.entry(k) {
                Entry::Occupied(mut e) => {
                    let v = e.get_mut();
                    let posting_list_id = PostingListId((*v >> 32) as u32);
                    let mut f = (*v & 0xFFFFFFFF) as u32;
                    f += freq;
                    *v = ((posting_list_id.0 as u64) << 32) | f as u64;
                    self.posting_storage.add_or_create(
                        posting_list_id,
                        vec![postings.values().cloned().collect()],
                    )?;
                }
                Entry::Vacant(e) => {
                    let posting_list_id = self.posting_storage.generate_new_id();
                    let f = freq;
                    let v: u64 = ((posting_list_id.0 as u64) << 32) | f as u64;
                    let postings: Vec<Posting> = postings.into_values().collect();
                    self.posting_storage
                        .add_or_create(posting_list_id, vec![postings])?;
                    e.insert(v);
                }
            }
        }

        let mut sorted_entries: Vec<_> = p.into_iter().collect();
        sorted_entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        debug!("building new fst map");

        let mut builder = MapBuilder::memory();

        for (key, value) in sorted_entries {
            builder.insert(key, value)?;
        }

        let fst = builder.into_map();
        drop(l);

        let mut l = self.fst_map.write().await;
        l.replace(fst);
        drop(l);

        self.total_document_length
            .fetch_add(total_documents_length as usize, Ordering::Relaxed);
        */

        Ok(())
    }

    pub async fn search(
        &self,
        tokens: Vec<String>,
        search_on: Option<&[FieldId]>,
        boost: HashMap<FieldId, f32>,
        scorer: &mut BM25Scorer<DocumentId>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
    ) -> Result<()> {
        let search_on = if let Some(v) = search_on {
            v.to_vec()
        } else {
            self.uncommitted.iter().map(|e| *e.key()).collect()
        };

        for field_id in search_on {
            let uncommitted = match self.uncommitted.get(&field_id) {
                Some(v) => v,
                None => {
                    warn!("FieldId {:?} not found in uncommitted index", field_id);
                    continue;
                }
            };

            let boost = boost.get(&field_id).copied().unwrap_or(1.0);

            uncommitted.search(&tokens, boost, scorer, filtered_doc_ids)?;
        }

        Ok(())
    }
}

/*
#[cfg(test)]
mod tests {
    use futures::{future::join_all, FutureExt};

    use crate::{
        collection_manager::dto::FieldId,
        document_storage::DocumentId,
        indexes::string::{scorer::bm25::BM25Score, StringIndex},
        nlp::{locales::Locale, TextParser},
    };
    use std::{collections::HashMap, sync::Arc};

    #[tokio::test]
    async fn test_empty_search_query() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "This is a test document.".to_string())],
        )]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(vec![], None, Default::default(), BM25Score::default(), None)
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty for empty query"
        );
    }

    #[tokio::test]
    async fn test_search_nonexistent_term() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "This is a test document.".to_string())],
        )]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["nonexistent".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty for non-existent term"
        );
    }

    #[tokio::test]
    async fn test_insert_empty_document() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(DocumentId(1), vec![(FieldId(0), "".to_string())])]
            .into_iter()
            .map(|(doc_id, fields)| {
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|(field_id, data)| {
                        let tokens = parser.tokenize_and_stem(&data);
                        (field_id, tokens)
                    })
                    .collect();
                (doc_id, fields)
            })
            .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["test".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when only empty documents are indexed"
        );
    }

    #[tokio::test]
    async fn test_search_with_field_filter() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![
                (FieldId(0), "This is a test in field zero.".to_string()),
                (FieldId(1), "Another test in field one.".to_string()),
            ],
        )]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["test".to_string()],
                Some(vec![FieldId(0)]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document when searching in FieldId(0)"
        );

        let output = string_index
            .search(
                vec!["test".to_string()],
                Some(vec![FieldId(1)]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document when searching in FieldId(1)"
        );

        let output = string_index
            .search(
                vec!["test".to_string()],
                Some(vec![FieldId(2)]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Should not find any documents when searching in non-existent FieldId"
        );
    }

    #[tokio::test]
    async fn test_search_with_boosts() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![
            (
                DocumentId(1),
                vec![(FieldId(0), "Important content in field zero.".to_string())],
            ),
            (
                DocumentId(2),
                vec![(
                    FieldId(1),
                    "Less important content in field one.".to_string(),
                )],
            ),
        ]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let mut boost = HashMap::new();
        boost.insert(FieldId(0), 2.0);

        let output = string_index
            .search(
                vec!["content".to_string()],
                None,
                boost,
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(output.len(), 2, "Should find both documents");

        let score_doc1 = output.get(&DocumentId(1)).unwrap();
        let score_doc2 = output.get(&DocumentId(2)).unwrap();

        assert!(
            score_doc1 > score_doc2,
            "Document with boosted field should have higher score"
        );
    }

    #[tokio::test]
    async fn test_insert_document_with_stop_words_only() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "the and but or".to_string())],
        )]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["the".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when only stop words are indexed"
        );
    }

    #[tokio::test]
    async fn test_search_on_empty_index() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);

        let output = string_index
            .search(
                vec!["test".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert!(
            output.is_empty(),
            "Search results should be empty when index is empty"
        );
    }

    #[tokio::test]
    async fn test_concurrent_insertions() {
        let id_geneator = Arc::new(Default::default());
        let string_index = Arc::new(StringIndex::new(id_geneator));

        let string_index_clone1 = Arc::clone(&string_index);
        let string_index_clone2 = Arc::clone(&string_index);

        let handle1 = async {
            let parser = TextParser::from_language(Locale::EN);
            let batch: HashMap<_, _> = vec![(
                DocumentId(1),
                vec![(
                    FieldId(0),
                    "Concurrent insertion test document one.".to_string(),
                )],
            )]
            .into_iter()
            .map(|(doc_id, fields)| {
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|(field_id, data)| {
                        let tokens = parser.tokenize_and_stem(&data);
                        (field_id, tokens)
                    })
                    .collect();
                (doc_id, fields)
            })
            .collect();

            string_index_clone1.insert_multiple(batch).await.unwrap();
        }
        .boxed();

        let handle2 = async {
            let parser = TextParser::from_language(Locale::EN);
            let batch: HashMap<_, _> = vec![(
                DocumentId(2),
                vec![(
                    FieldId(0),
                    "Concurrent insertion test document two.".to_string(),
                )],
            )]
            .into_iter()
            .map(|(doc_id, fields)| {
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|(field_id, data)| {
                        let tokens = parser.tokenize_and_stem(&data);
                        (field_id, tokens)
                    })
                    .collect();
                (doc_id, fields)
            })
            .collect();

            string_index_clone2.insert_multiple(batch).await.unwrap();
        }
        .boxed();

        join_all(vec![handle1, handle2]).await;

        let parser = TextParser::from_language(Locale::EN);
        let search_tokens = parser
            .tokenize_and_stem("concurrent")
            .into_iter()
            .map(|(original, _)| original)
            .collect::<Vec<_>>();

        let output = string_index
            .search(
                search_tokens,
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            2,
            "Should find both documents after concurrent insertions"
        );
    }

    #[tokio::test]
    async fn test_large_documents() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let large_text = "word ".repeat(10000);

        let batch: HashMap<_, _> = vec![(DocumentId(1), vec![(FieldId(0), large_text.clone())])]
            .into_iter()
            .map(|(doc_id, fields)| {
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|(field_id, data)| {
                        let tokens = parser.tokenize_and_stem(&data);
                        (field_id, tokens)
                    })
                    .collect();
                (doc_id, fields)
            })
            .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["word".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the large text"
        );
    }

    #[tokio::test]
    async fn test_high_term_frequency() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let repeated_word = "repeat ".repeat(1000);

        let batch: HashMap<_, _> = vec![(DocumentId(1), vec![(FieldId(0), repeated_word.clone())])]
            .into_iter()
            .map(|(doc_id, fields)| {
                let fields: Vec<_> = fields
                    .into_iter()
                    .map(|(field_id, data)| {
                        let tokens = parser.tokenize_and_stem(&data);
                        (field_id, tokens)
                    })
                    .collect();
                (doc_id, fields)
            })
            .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec!["repeat".to_string()],
                None,
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document with high term frequency"
        );
    }

    #[tokio::test]
    async fn test_term_positions() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(
                FieldId(0),
                "quick brown fox jumps over the lazy dog".to_string(),
            )],
        )]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();
    }

    #[tokio::test]
    async fn test_exact_phrase_match() {
        let id_geneator = Arc::new(Default::default());
        let string_index = StringIndex::new(id_geneator);
        let parser = TextParser::from_language(Locale::EN);

        let batch: HashMap<_, _> = vec![(
            DocumentId(1),
            vec![(FieldId(0), "5200 mAh battery in disguise".to_string())],
        )]
        .into_iter()
        .map(|(doc_id, fields)| {
            let fields: Vec<_> = fields
                .into_iter()
                .map(|(field_id, data)| {
                    let tokens = parser.tokenize_and_stem(&data);
                    (field_id, tokens)
                })
                .collect();
            (doc_id, fields)
        })
        .collect();

        string_index.insert_multiple(batch).await.unwrap();

        let output = string_index
            .search(
                vec![
                    "5200".to_string(),
                    "mAh".to_string(),
                    "battery".to_string(),
                    "in".to_string(),
                    "disguise".to_string(),
                ],
                Some(vec![FieldId(0)]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await
            .unwrap();

        assert_eq!(
            output.len(),
            1,
            "Should find the document containing the exact phrase"
        );

        assert!(
            output.contains_key(&DocumentId(1)),
            "Document with ID 1 should be found"
        );
    }
}
*/
