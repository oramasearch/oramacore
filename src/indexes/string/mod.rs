use std::{
    collections::{HashMap, HashSet}, ops::AddAssign, path::PathBuf, sync::{atomic::AtomicU64, Arc}
};

use anyhow::Result;

pub use committed::CommittedStringFieldIndex;
use dashmap::DashMap;

use scorer::BM25Scorer;
use serde::{Deserialize, Serialize};
use tracing::warn;
pub use uncommitted::UncommittedStringFieldIndex;

use crate::{
    collection_manager::{dto::FieldId, sides::write::InsertStringTerms},
    document_storage::DocumentId,
};

mod committed;

#[cfg(any(test, feature = "benchmarking"))]
pub use committed::merge;
#[cfg(not(any(test, feature = "benchmarking")))]
use committed::merge;

// pub mod posting_storage;
pub mod scorer;
mod uncommitted;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PostingListId(pub u32);

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
    committed: DashMap<FieldId, CommittedStringFieldIndex>,
    posting_id_generator: Arc<AtomicU64>,
}

#[derive(Debug, Default)]
pub struct GlobalInfo {
    pub total_documents: usize,
    pub total_document_length: usize,
}
impl AddAssign for GlobalInfo {
    fn add_assign(&mut self, rhs: Self) {
        self.total_documents += rhs.total_documents;
        self.total_document_length += rhs.total_document_length;
    }
}

impl StringIndex {
    pub fn new(posting_id_generator: Arc<AtomicU64>) -> Self {
        StringIndex {
            uncommitted: DashMap::new(),
            committed: DashMap::new(),
            posting_id_generator,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn insert(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        field_length: u16,
        terms: InsertStringTerms,
    ) -> Result<()> {
        let uncommitted = self.uncommitted.entry(field_id).or_default();
        uncommitted.insert(doc_id, field_length, terms)?;

        Ok(())
    }

    pub fn commit(&mut self, new_path: PathBuf) -> Result<()> {
        let uncommitted = std::mem::take(&mut self.uncommitted);
        let committed = std::mem::take(&mut self.committed);

        let all_fields = uncommitted
            .iter()
            .map(|e| *e.key())
            .chain(committed.iter().map(|e| *e.key()))
            .collect::<HashSet<_>>();

        for field_id in all_fields {
            let uncommitted = uncommitted.remove(&field_id);
            let committed = committed.remove(&field_id);

            let field_new_path = new_path.join(format!("{}.bin", field_id.0));

            match (uncommitted, committed) {
                (Some((_, uncommitted)), Some((_, committed))) => {
                    let committed =
                        merge::merge(self.posting_id_generator.clone(), uncommitted, committed, field_new_path)?;

                    self.committed.insert(field_id, committed);
                }
                (Some((_, uncommitted)), None) => {
                    let committed = merge::merge(
                        self.posting_id_generator.clone(),
                        uncommitted,
                        CommittedStringFieldIndex::default(),
                        field_new_path,
                    )?;
                    self.committed.insert(field_id, committed);
                }
                (None, Some((_, committed))) => {
                    self.committed.insert(field_id, committed);
                }
                (None, None) => {
                    warn!(
                        r#"No uncommitted or committed index found for field_id: {:?}.
This should never happen because the field id list is kept from uncommitted and committed field index.
So, where this field id came from?"#,
                        field_id
                    );
                }
            };
        }

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
        let search_on: HashSet<_> = if let Some(v) = search_on {
            v.iter().copied().collect()
        } else {
            self.uncommitted
                .iter()
                .map(|e| *e.key())
                .chain(self.committed.iter().map(|e| *e.key()))
                .collect()
        };

        for field_id in search_on {
            let boost = boost.get(&field_id).copied().unwrap_or(1.0);

            let uncommitted = self.uncommitted.get(&field_id);
            let committed = self.committed.get(&field_id);

            let mut global_info = committed
                .as_ref()
                .map(|c| c.get_global_info())
                .unwrap_or_default();

            // We share the global info between committed and uncommitted indexes
            // Anyway the postings aren't shared, so if a word is in both indexes:
            // - it will be scored twice
            // - the occurrence (stored in the posting) will be different
            // We can fix this, but it count be hard.
            // Anyway, the impact of this is low, so we can ignore it for now.
            // Also because soon or later, we will merge the uncommitted index into the committed one.
            // So for now, we "/10" the boost of the uncommitted index, where "10" is an arbitrary number.
            // TODO: evaluate the impact of this and fix it if needed

            if let Some(uncommitted) = uncommitted {
                global_info += uncommitted.get_global_info();
                uncommitted.search(
                    &tokens,
                    boost / 10.0,
                    scorer,
                    filtered_doc_ids,
                    &global_info,
                )?;
            }

            if let Some(committed) = committed {
                committed.search(&tokens, boost, scorer, filtered_doc_ids, &global_info)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use serde_json::json;

    use crate::{
        collection_manager::sides::write::{Term, TermStringField},
        test_utils::{create_string_index, generate_new_path},
    };

    use super::*;

    #[tokio::test]
    async fn test_indexes_string_insert_search_commit_search() -> Result<()> {
        let mut string_index = create_string_index(
            vec![(FieldId(0), "field".to_string())],
            vec![
                json!({
                    "field": "hello hello world",
                })
                .try_into()?,
                json!({
                    "field": "hello tom",
                })
                .try_into()?,
            ],
        )?;

        let mut scorer = BM25Scorer::new();
        string_index
            .search(
                vec!["hello".to_string()],
                None,
                Default::default(),
                &mut scorer,
                None,
            )
            .await?;
        let before_output = scorer.get_scores();

        string_index.commit(generate_new_path())?;

        let mut scorer = BM25Scorer::new();
        string_index
            .search(
                vec!["hello".to_string()],
                None,
                Default::default(),
                &mut scorer,
                None,
            )
            .await?;
        let after_output = scorer.get_scores();

        assert_approx_eq!(
            after_output[&DocumentId(0)] / 10.0,
            before_output[&DocumentId(0)]
        );
        assert_approx_eq!(
            after_output[&DocumentId(1)] / 10.0,
            before_output[&DocumentId(1)]
        );

        string_index.insert(
            DocumentId(2),
            FieldId(0),
            1,
            HashMap::from_iter([(
                Term("hello".to_string()),
                TermStringField { positions: vec![1] },
            )]),
        )?;

        let mut scorer = BM25Scorer::new();
        string_index
            .search(
                vec!["hello".to_string()],
                None,
                Default::default(),
                &mut scorer,
                None,
            )
            .await?;
        let after_insert_output = scorer.get_scores();

        assert_eq!(after_insert_output.len(), 3);
        assert!(after_insert_output.contains_key(&DocumentId(0)));
        assert!(after_insert_output.contains_key(&DocumentId(1)));
        assert!(after_insert_output.contains_key(&DocumentId(2)));

        string_index.commit(generate_new_path())?;

        let mut scorer = BM25Scorer::new();
        string_index
            .search(
                vec!["hello".to_string()],
                None,
                Default::default(),
                &mut scorer,
                None,
            )
            .await?;
        let after_insert_commit_output = scorer.get_scores();

        assert_eq!(after_insert_commit_output.len(), 3);
        assert!(after_insert_commit_output.contains_key(&DocumentId(0)));
        assert!(after_insert_commit_output.contains_key(&DocumentId(1)));
        assert!(after_insert_commit_output.contains_key(&DocumentId(2)));

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
